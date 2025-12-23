import os
import types
import atexit
from time import perf_counter
from functools import partial
from dataclasses import fields, asdict
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from transformers import EosTokenCriteria, RepetitionPenaltyLogitsProcessor
try:    
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt as TokensPrompt
    SUPPORT_VLLM = True
except ImportError:
    SUPPORT_VLLM = True

import logging
from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.models.modules.sampler import _ras_sample_hf_engine
import asyncio
import uuid
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

class HFLLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=self.device)
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        prompt: list[str],
        sampling_param: SamplingParams,
        past_key_values=None,
    ) -> dict:
        
        stopping_criteria = StoppingCriteriaList([EosTokenCriteria(eos_token_id=self.config.hf_config.eos_token_id)])
        if sampling_param.use_ras:
            sample_hf_engine_handler = partial(_ras_sample_hf_engine, 
                    use_ras=sampling_param.use_ras, 
                    win_size=sampling_param.win_size, tau_r=sampling_param.tau_r)
        else:
            sample_hf_engine_handler = None
        rep_pen_processor = RepetitionPenaltyLogitsProcessor(
            penalty=sampling_param.repetition_penalty,
            prompt_ignore_length=len(prompt)
        ) # exclude the input prompt, consistent with vLLM implementation;
        with torch.no_grad(): 
            input_len = len(prompt)
            generated_ids = self.model.generate(
                input_ids = torch.tensor([prompt], dtype=torch.int64).to(self.device),
                do_sample=True,
                top_k=sampling_param.top_k,
                top_p=sampling_param.top_p,
                min_new_tokens=sampling_param.min_tokens,
                max_new_tokens=sampling_param.max_tokens,
                temperature=sampling_param.temperature,
                stopping_criteria=stopping_criteria,
                past_key_values=past_key_values,
                custom_generate=sample_hf_engine_handler,
                use_cache=True,
                logits_processor=[rep_pen_processor]
            )
            generated_ids = generated_ids[:, input_len:].cpu().numpy().tolist()[0]
        output = {
            "text": self.tokenizer.decode(generated_ids),
            "token_ids": generated_ids,
        }
        return output

class VLLMEngine:

    def __init__(self, model, **kwargs):

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = config.hf_config.eos_token_id # speech eos token;
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        os.environ["VLLM_USE_V1"] = "1"
        if SUPPORT_VLLM:
            engine_args = AsyncEngineArgs(model=model, enforce_eager=False, dtype="bfloat16", quantization="fp8",max_model_len=8192, enable_prefix_caching=True,logits_processors=["vllm_ras_logits_processor:FixedRASLogitsProcessor"])#,quantization="fp8",logits_processors=["/workspace/bella-infra/user/libeibei031/SoulX/SoulX-Podcast-main/vllm_ras_logits_processor.py:FixedRASLogitsProcessor"]
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:#,logits_processors=["vllm_ras_logits_processor:FixedRASLogitsProcessor"]
            raise ImportError("Not Support VLLM now!!!")
        self.config = config
        self.pad_token_id = self.tokenizer.pad_token_id

        # 异步请求跟踪
        self.pending_requests = {}  # {request_id: {'prompt': [...], 'status': 'pending'}}

    def generate(
        self,
        prompt: list[str],
        sampling_param: SamplingParams,
        past_key_values=None,
    ) -> dict:
        sampling_param.stop_token_ids = [self.config.hf_config.eos_token_id]
        # Filter out custom RAS parameters that vLLM doesn't support
        vllm_params = asdict(sampling_param)
        vllm_params.pop('use_ras', None)
        vllm_params.pop('win_size', None)
        vllm_params.pop('tau_r', None)
        if isinstance(prompt[0], list):
                # 说明是 Batch 输入: [[...], [...]]
            vllm_inputs = [TokensPrompt(prompt_token_ids=p) for p in prompt]
        else:
            # 说明是单条输入: [id, id, id...]
            vllm_inputs = TokensPrompt(prompt_token_ids=prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                vllm_inputs,
                VllmSamplingParams(**vllm_params),
                use_tqdm=False,
            )
        generated_ids_list = [out.outputs[0].token_ids for out in outputs]
        # print(list(generated_ids))
        start_time = time.time()
        output = {
            "text": self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True),
            "token_ids": generated_ids_list,
        }
        batch_decode_time = time.time()
        logging.info(f"batch decode time {batch_decode_time-start_time:.4f} seconds")
        # print(output["text"])
        return output

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

    def poll_results(self, request_ids=None):
        """
        非阻塞查询结果，返回已完成的请求

        Args:
            request_ids: 要查询的请求ID列表，None表示查询所有pending请求

        Returns:
            completed: {request_id: {'token_ids': [...], 'text': '...'}}
        """
        if request_ids is None:
            request_ids = list(self.pending_requests.keys())

        completed = {}
        for req_id in request_ids:
            if req_id not in self.pending_requests:
                continue

            # 使用vLLM V1 API获取请求输出
            try:
                outputs = self.model.llm_engine.get_request_outputs(req_id)
                if outputs and outputs.finished:
                    token_ids = outputs.outputs[0].token_ids
                    completed[req_id] = {
                        'token_ids': token_ids,
                        'text': self.tokenizer.decode(token_ids, skip_special_tokens=False)
                    }
                    # 从pending中移除
                    del self.pending_requests[req_id]
                    logging.debug(f"Request {req_id} completed")
            except Exception as e:
                logging.error(f"Error polling request {req_id}: {e}")
                # 出错的请求也从pending中移除
                del self.pending_requests[req_id]

        return completed