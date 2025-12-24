import os
from dataclasses import fields, asdict
import time
import torch

from transformers import AutoTokenizer


from vllm import SamplingParams as VllmSamplingParams
from vllm.inputs import TokensPrompt as TokensPrompt

import logging
from soulxpodcast.config import Config, SamplingParams
import asyncio
import uuid
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


class VLLMEngine:

    def __init__(self, model, **kwargs):

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        os.environ["VLLM_USE_V1"] = "1"
        engine_args = AsyncEngineArgs(model=model, 
                                      enforce_eager=False,
                                      dtype="bfloat16", 
                                      quantization="fp8",
                                      max_model_len=8192, 
                                      enable_prefix_caching=True,
                                      logits_processors=["vllm_ras_logits_processor:FixedRASLogitsProcessor"],
                                      disable_log_stats=False,
                                    #   stats_interval=5.0,
                                    #   log_format="text",
                                      gpu_memory_utilization=0.8)#,quantization="fp8",logits_processors=["/workspace/bella-infra/user/libeibei031/SoulX/SoulX-Podcast-main/vllm_ras_logits_processor.py:FixedRASLogitsProcessor"]
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

        # 异步请求跟踪
        self.pending_requests = {}  # {request_id: {'prompt': [...], 'status': 'pending'}}


    def _to_vllm_params(self, sampling_param: SamplingParams) -> VllmSamplingParams:
        """转换SamplingParams为vLLM格式"""
        vllm_params = asdict(sampling_param)
        # 过滤RAS参数（vLLM不支持）
        vllm_params.pop('use_ras', None)
        vllm_params.pop('win_size', None)
        vllm_params.pop('tau_r', None)
        return VllmSamplingParams(**vllm_params)

    async def generate_async(self, prompt_tokens: list[int], sampling_param: SamplingParams):
        """
        真正的异步生成接口。
        不再手动 add_request，而是直接 await 结果。
        """
        request_id = f"soulx_{uuid.uuid4().hex}"
        vllm_params = self._to_vllm_params(sampling_param)
        
        # vLLM V1 要求的输入格式
        inputs = {"prompt_token_ids": prompt_tokens}

        # 获取异步生成器
        results_generator = self.engine.generate(inputs, vllm_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output:
            token_ids = final_output.outputs[0].token_ids
            return {
                "token_ids": token_ids,
                "text": self.tokenizer.decode(token_ids, skip_special_tokens=True)
            }
        return None
