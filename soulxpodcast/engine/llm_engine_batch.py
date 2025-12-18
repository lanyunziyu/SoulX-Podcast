"""
Extended LLM Engine with Batch Support

扩展vLLM引擎以支持batch推理
"""
import os
import time
import logging
from typing import List, Dict, Any
from dataclasses import asdict

import torch
from transformers import AutoTokenizer

try:
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt
    SUPPORT_VLLM = True
except ImportError:
    SUPPORT_VLLM = False

from soulxpodcast.config import SamplingParams

logger = logging.getLogger(__name__)


class VLLMBatchEngine:
    """
    vLLM Batch推理引擎

    功能：
    1. 支持多个请求的batch推理
    2. 自动处理变长序列
    3. 返回按输入顺序排列的结果
    """

    def __init__(self, model: str = None, hf_config=None, **kwargs):
        """
        Args:
            model: 模型路径
            hf_config: HF配置对象 (兼容现有Config)
            **kwargs: vLLM配置参数
        """
        # 兼容性处理：如果传入的是完整config对象
        if model is None and 'model_path' in kwargs:
            model = kwargs['model_path']

        self.model_path = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.hf_config = hf_config

        # 默认vLLM配置
        vllm_config = {
            "enforce_eager": kwargs.get("enforce_eager", True),
            "dtype": kwargs.get("dtype", "bfloat16"),
            "max_model_len": kwargs.get("max_model_len", 8192),
            "enable_prefix_caching": kwargs.get("enable_prefix_caching", True),
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
        }

        # 添加logits processor
        logits_processors = kwargs.get("logits_processors", [
            "vllm_ras_logits_processor_fixed:FixedRASLogitsProcessor"
        ])

        logger.info(f"Initializing VLLMBatchEngine with config: {vllm_config}")

        if not SUPPORT_VLLM:
            raise ImportError("vLLM is not installed. Please install it first.")

        # 设置vLLM v1
        os.environ["VLLM_USE_V1"] = "1"

        # 初始化vLLM
        self.model = LLM(
            model=model,
            logits_processors=logits_processors,
            **vllm_config
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"VLLMBatchEngine initialized on {self.device}")

    def generate_single(
        self,
        prompt: List[int],
        sampling_params: SamplingParams,
    ) -> Dict[str, Any]:
        """
        单个请求生成（保持兼容性）

        Args:
            prompt: token ids列表
            sampling_params: 采样参数

        Returns:
            Dict: {"text": str, "token_ids": List[int]}
        """
        # 转换为batch调用
        results = self.generate_batch([prompt], [sampling_params])
        return results[0]

    def generate_batch(
        self,
        prompts: List[List[int]],
        sampling_params_list: List[SamplingParams],
    ) -> List[Dict[str, Any]]:
        """
        Batch推理

        Args:
            prompts: token ids列表的列表 [[tokens1], [tokens2], ...]
            sampling_params_list: 每个请求的采样参数列表

        Returns:
            List[Dict]: 结果列表，按输入顺序排列
                [{"text": str, "token_ids": List[int], "inference_time": float}, ...]
        """
        batch_size = len(prompts)
        logger.info(f"Starting batch generation: batch_size={batch_size}")

        start_time = time.time()

        # 转换为vLLM输入格式
        vllm_prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]

        # 转换采样参数（使用第一个作为默认）
        base_sampling_params = sampling_params_list[0] if sampling_params_list else SamplingParams()

        # 准备vLLM采样参数
        vllm_sampling_params = VllmSamplingParams(
            temperature=base_sampling_params.temperature,
            top_p=base_sampling_params.top_p,
            top_k=base_sampling_params.top_k,
            repetition_penalty=base_sampling_params.repetition_penalty,
            max_tokens=base_sampling_params.max_tokens,
            min_tokens=base_sampling_params.min_tokens,
            stop_token_ids=base_sampling_params.stop_token_ids or [],
        )

        # 添加extra_args
        if hasattr(base_sampling_params, 'extra_args') and base_sampling_params.extra_args:
            for key, value in base_sampling_params.extra_args.items():
                if hasattr(vllm_sampling_params, key):
                    setattr(vllm_sampling_params, key, value)

        # 执行batch推理
        try:
            with torch.no_grad():
                vllm_outputs = self.model.generate(
                    vllm_prompts,
                    vllm_sampling_params,
                    use_tqdm=False,
                )

            inference_time = time.time() - start_time
            logger.info(f"Batch inference completed in {inference_time:.4f}s, avg={inference_time/batch_size:.4f}s per sample")

            # 解析输出
            results = []
            for i, output in enumerate(vllm_outputs):
                token_ids = list(output.outputs[0].token_ids)
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

                results.append({
                    "text": text,
                    "token_ids": token_ids,
                    "inference_time": inference_time / batch_size,
                    "batch_index": i,
                })

            return results

        except Exception as e:
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            raise RuntimeError(f"vLLM batch inference failed: {str(e)}")

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "max_model_len": getattr(self.model.llm_engine.model_config, "max_model_len", None),
        }

    def estimate_tokens_per_second(self, batch_size: int = 8, num_tokens: int = 100) -> float:
        """
        估算吞吐量（tokens/second）

        Args:
            batch_size: 测试batch大小
            num_tokens: 生成token数量

        Returns:
            float: tokens/second
        """
        logger.info(f"Estimating throughput with batch_size={batch_size}, num_tokens={num_tokens}")

        # 创建测试prompt
        test_prompts = [[1] * 10 for _ in range(batch_size)]  # 简单的测试序列

        # 创建测试采样参数
        test_params = [
            SamplingParams(
                temperature=1.0,
                max_tokens=num_tokens,
                min_tokens=num_tokens,
            )
            for _ in range(batch_size)
        ]

        start_time = time.time()
        try:
            _ = self.generate_batch(test_prompts, test_params)
            elapsed = time.time() - start_time

            total_tokens = batch_size * num_tokens
            throughput = total_tokens / elapsed

            logger.info(f"Estimated throughput: {throughput:.2f} tokens/s")
            return throughput

        except Exception as e:
            logger.warning(f"Failed to estimate throughput: {e}")
            return 0.0


# 向后兼容的单例接口
_batch_engine = None


def get_vllm_batch_engine(model: str, **kwargs) -> VLLMBatchEngine:
    """
    获取全局vLLM Batch引擎实例（单例）

    Args:
        model: 模型路径
        **kwargs: vLLM配置参数

    Returns:
        VLLMBatchEngine
    """
    global _batch_engine

    if _batch_engine is None:
        _batch_engine = VLLMBatchEngine(model, **kwargs)

    return _batch_engine
