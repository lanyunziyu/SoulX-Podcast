import time
from datetime import datetime

from tqdm import tqdm
from itertools import chain
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import s3tokenizer
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from soulxpodcast.config import Config, SamplingParams, AutoPretrainedConfig
from soulxpodcast.engine.llm_engine import (
    HFLLMEngine, VLLMEngine
)
from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec
from soulxpodcast.models.modules.hifigan import HiFTGenerator

class SoulXPodcast(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda().eval()
        if self.config.llm_engine == "hf":
            self.llm = HFLLMEngine(**self.config.__dict__)
        elif self.config.llm_engine == "vllm":
            self.llm = VLLMEngine(**self.config.__dict__)
        else:
            raise NotImplementedError

        self.use_tqdm = True

        self.flow = CausalMaskedDiffWithXvec()
        if self.config.hf_config.fp16_flow:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [INFO] - Casting flow to fp16")
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{self.config.model}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{self.config.model}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

    
    @torch.inference_mode()
    def forward_longform(
        self, prompt_mels_for_llm,
        prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]],
        text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow_ori, 
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams],
        spk_ids: list[list[int]],
        use_dialect_prompt: bool = False,
        dialect_prompt_text_tokens_for_llm: list[list[int]] = None,
        dialect_prefix: list[list[int]] = None,
        **kwargs,  # for compatibility
    ):
        s3audio_tokenization_start_time = time.time()
        prompt_size, turn_size = len(prompt_mels_for_llm), len(text_tokens_for_llm)

        # Audio tokenization
        prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )
        s3audio_tokenization_time = time.time()
        logging.info(f"s3Audio tokenization completed in {(s3audio_tokenization_time- s3audio_tokenization_start_time):.4f} seconds")

        # align speech token with speech feat as to reduce
        #    the noise ratio during the generation process.
        prompt_speech_tokens = []
        prompt_mels_for_flow, prompt_mels_lens_for_flow = [], []

        # for prompt_index in range(prompt_size):
        #     prompt_speech_token_len = prompt_speech_tokens_lens_ori[prompt_index].item()
        #     prompt_speech_token = prompt_speech_tokens_ori[prompt_index, :prompt_speech_token_len]
        #     prompt_mel = prompt_mels_for_flow_ori[prompt_index]
        #     prompt_mel_len = prompt_mel.shape[0]
        #     if prompt_speech_token_len * 2 > prompt_mel_len:
        #         prompt_speech_token = prompt_speech_token[:int(prompt_mel_len/2)]
        #         prompt_mel_len = torch.tensor([prompt_mel_len]).cuda()
        #     else:
        #         prompt_mel = prompt_mel.detach().clone()[:prompt_speech_token_len * 2].cuda()
        #         prompt_mel_len = torch.tensor([prompt_speech_token_len * 2]).cuda()
        #     prompt_speech_tokens.append(prompt_speech_token)
        #     prompt_mels_for_flow.append(prompt_mel)
        #     prompt_mels_lens_for_flow.append(prompt_mel_len)
        prompt_mels_for_flow_ori_tensor = pad_sequence(
            prompt_mels_for_flow_ori, 
            batch_first=True, 
            padding_value=0.0
        ).cuda()

        # 获取每个样本原始 Mel 的真实长度
        ori_mel_lens = torch.tensor([m.shape[0] for m in prompt_mels_for_flow_ori], device='cuda')

        # 2. 计算目标长度逻辑 (对齐 Speech Token * 2)
        # prompt_speech_tokens_lens_ori 是量化器返回的，形状为 [Batch]
        target_mel_lens = prompt_speech_tokens_lens_ori * 2

        # 3. 确定最终长度：取 target 和 original 的最小值 (对应原 if-else 逻辑)
        final_mel_lens = torch.min(target_mel_lens, ori_mel_lens)
        final_token_lens = final_mel_lens // 2

        # 4. 向量化截断 Speech Tokens
        # 这里的 prompt_speech_tokens_ori 形状通常是 [Batch, Max_Token_Len]
        batch_indices = torch.arange(prompt_speech_tokens_ori.size(1), device='cuda')[None, :]
        token_mask = batch_indices < final_token_lens[:, None]
        prompt_speech_tokens_batch = prompt_speech_tokens_ori * token_mask

        # 5. 向量化截断 Mel Features
        mel_indices = torch.arange(prompt_mels_for_flow_ori_tensor.size(1), device='cuda')[None, :]
        mel_mask = mel_indices < final_mel_lens[:, None]
        prompt_mels_for_flow_batch = prompt_mels_for_flow_ori_tensor * mel_mask.unsqueeze(-1)

        # 6. 还原为后续逻辑需要的 List 格式 (如果后续 LLM 还是逐个处理，则保持 list)
        # 如果后续 LLM/Flow 已经支持 Batch，建议直接使用 Tensor 提高效率
        prompt_speech_tokens = [t[:l] for t, l in zip(prompt_speech_tokens_batch, final_token_lens)]
        prompt_mels_for_flow = [m[:l] for m, l in zip(prompt_mels_for_flow_batch, final_mel_lens)]
        prompt_mels_lens_for_flow = final_mel_lens.unsqueeze(-1) # 保持 Tensor [Batch, 1] 格式

        
        # Prepare LLM inputs
        prompt_inputs = []
        # history_inputs = []
        
        for i in range(prompt_size):
            speech_tokens_i = [token+self.config.hf_config.speech_token_offset for token in prompt_speech_tokens[i].tolist()]
            speech_tokens_i += [self.config.hf_config.eos_token_id]
            if use_dialect_prompt and len(dialect_prompt_text_tokens_for_llm[i])>0:
                dialect_prompt_input = prompt_text_tokens_for_llm[i] + speech_tokens_i + dialect_prompt_text_tokens_for_llm[i]
                if i>0:
                    dialect_prompt_input = dialect_prefix[0] + dialect_prompt_input
                prompt_input = self.llm.generate(dialect_prompt_input, sampling_params, past_key_values=None)['token_ids']
                prompt_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
                # history_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
            else:
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )
                # history_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )

        preprocessing_time = time.time()
        logging.info(f"LLM input processing completed in {preprocessing_time-s3audio_tokenization_time:.4f} seconds")
        generated_wavs, results_dict = [], {}

        
        # LLM generation
        inputs_prompt = list(chain.from_iterable(prompt_inputs))
        
        # cache_config = AutoPretrainedConfig().from_dataclass(self.llm.config.hf_config)
        # past_key_values = DynamicCache(config=cache_config)
        # valid_turn_size = prompt_size
        for i in range(1):

            # # set ratio: reach the reset cache ratio;
            # if valid_turn_size > self.config.max_turn_size or len(inputs)>self.config.turn_tokens_threshold:
            #     assert self.config.max_turn_size >= self.config.prompt_context + self.config.history_context, "Invalid Long history size setting, "
            #     prompt_text_bound = max(self.config.prompt_context, len(history_inputs)-self.config.history_text_context-self.config.history_context)
            #     inputs = list(chain.from_iterable(
            #         history_inputs[:self.config.prompt_context]+ \
            #         history_inputs[prompt_text_bound:-self.config.history_context]+ \
            #         prompt_inputs[-self.config.history_context:]
            #     ))
            #     valid_turn_size = self.config.prompt_context + len(history_inputs) - prompt_text_bound
            #     # past_key_values = DynamicCache(config=cache_config)
            # valid_turn_size += 1

            inputs = [inputs_prompt + t for t in text_tokens_for_llm]

            # batched_inputs = [inputs for _ in range(10)]

            llm_outputs = self.llm.generate(inputs, sampling_params)

            llm_processing_time = time.time()
            logging.info(f"LLM generated input {llm_processing_time-preprocessing_time:.4f} seconds")

            # inputs.extend(llm_outputs['token_ids'])
            # prompt_inputs.append(text_tokens_for_llm[i]+llm_outputs['token_ids'])
            # history_inputs.append(text_tokens_for_llm[i][:-1]) # remove the <|audio_start|>


            
            # Prepare Flow inputs
            turn_spk = spk_ids[i]
            # generated_speech_tokens = [token - self.config.hf_config.speech_token_offset for token in  llm_outputs['token_ids'][:-1]]  # ignore last eos
            generated_speech_tokens = [
                [token - self.config.hf_config.speech_token_offset for token in tokens[:-1]]
                for tokens in llm_outputs['token_ids']
            ]
            prompt_speech_token = prompt_speech_tokens[turn_spk].tolist()
            p_t = torch.tensor(prompt_speech_token) 

            # 2. 逐样本拼接（将 List 转换为 Tensor 后再 cat）
            padding_start_indices = []
            combined_list = []
            for g_list in generated_speech_tokens:
                # 将生成的 List[int] 转为 Tensor
                g_t = torch.tensor(g_list)
                # 两个 Tensor 拼接
                combined_sample = torch.cat([p_t, g_t], dim=0)
                padding_start_indices.append(combined_sample.size(0))
                combined_list.append(combined_sample)
            
            # flow_input = torch.tensor([prompt_speech_token + generated_speech_tokens])
            flow_input = pad_sequence(combined_list, batch_first=True, padding_value=0)
            # flow_inputs_len = torch.tensor([len(prompt_speech_token) + len(generated_speech_tokens)])
            flow_inputs_len = torch.tensor([flow_input.shape[1]])

            # Flow generation and HiFi-GAN generation
            start_idx = spk_ids[i]
            prompt_mels = prompt_mels_for_flow[start_idx][None]
            prompt_mels_lens = prompt_mels_lens_for_flow[start_idx][None]
            spk_emb = spk_emb_for_flow[start_idx:start_idx+1]
            
            
            batch_flow_input = flow_input.shape[0]
            # flow_input=torch.cat([flow_input, flow_input[:, -1:]], dim=1).reshape(2,-1)
            flow_inputs_len = torch.full((batch_flow_input, 1), flow_input.shape[1], dtype=torch.long, device=flow_input.device)
            mel=prompt_mels.repeat(batch_flow_input, 1, 1)
            # mels_len = (prompt_mels_lens)
            # Flow generation
            with torch.amp.autocast("cuda", dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
                generated_mels, generated_mels_lens = self.flow(
                    flow_input.cuda(), flow_inputs_len.cuda(),
                    mel, prompt_mels_lens.expand(batch_flow_input,1), spk_emb.cuda(),
                    streaming=False, finalize=True
                )
            flow_generation_time = time.time()
            logging.info(f"Flow mel feature generation completed in {flow_generation_time-llm_processing_time:.4f} seconds")     

            # HiFi-GAN generation
            mel = generated_mels[:, :, prompt_mels_lens[0].item():generated_mels_lens[0].item()]#prompt_mels_lens
            wav, _ = self.hift(speech_feat=mel)
            
            
            
            padding_start_indices = [int((t - final_token_lens) * 1.8 * 480) for t in padding_start_indices]
            wav_list = list(wav)
            final_wavs = []
            for i in range(len(wav_list)):
                # 1. 拿到该 batch 的 Token 截断位置
                token_end_idx = padding_start_indices[i]
                trimmed_wav = wav_list[i][:token_end_idx]
                final_wavs.append(trimmed_wav)
            generated_wavs.append(wav)
            hifi_generation_time = time.time()
            logging.info(f"HIFI sampling generation completed in {hifi_generation_time-flow_generation_time:.4f} seconds")

        # Save the generated wav;
        results_dict['generated_wavs'] = generated_wavs
        s_time = time.time()
        logging.info(f"s_time {s_time:.4f} seconds")
        return results_dict

    @torch.inference_mode()
    def forward_batch(
        self, prompt_mels_for_llm,
        prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]],
        text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow_ori,
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams],
        spk_ids: list[list[int]],
        use_dialect_prompt: bool = False,
        dialect_prompt_text_tokens_for_llm: list[list[int]] = None,
        dialect_prefix: list[list[int]] = None,
        inputs_prompt: list = None,  # 支持传入已有的prompt
        return_state: bool = False,  # 是否返回状态信息
        **kwargs,  # for compatibility
    ):
        """
        批量并行前向传播，支持状态管理和多轮对话

        Args:
            inputs_prompt: 可选的已有inputs_prompt，用于多轮对话的上下文
            return_state: 如果为True，返回dict包含音频结果和状态；如果为False，只返回音频列表
            其他参数与forward_longform相同

        Returns:
            list 或 dict:
                - 如果return_state=False: 批量音频结果列表 [np.ndarray, ...]
                - 如果return_state=True: {
                    'audio_results': list,      # 音频结果列表
                    'inputs_prompt': list,      # 更新后的inputs_prompt
                    'llm_outputs': dict,        # LLM输出
                    'generation_info': dict     # 生成信息
                  }
        """
        batch_start_time = time.time()
        turn_size = len(text_tokens_for_llm)
        logging.info(f"Starting batch forward pass for {turn_size} text segments")

        # === 直接复用forward_longform的逻辑，只修改最后的返回格式 ===

        s3audio_tokenization_start_time = time.time()
        prompt_size = len(prompt_mels_for_llm)

        # Audio tokenization
        prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )
        s3audio_tokenization_time = time.time()
        logging.info(f"s3Audio tokenization completed in {(s3audio_tokenization_time- s3audio_tokenization_start_time):.4f} seconds")

        # align speech token with speech feat as to reduce
        #    the noise ratio during the generation process.
        prompt_speech_tokens = []
        prompt_mels_for_flow, prompt_mels_lens_for_flow = [], []

        prompt_mels_for_flow_ori_tensor = pad_sequence(
            prompt_mels_for_flow_ori,
            batch_first=True,
            padding_value=0.0
        ).cuda()

        # 获取每个样本原始 Mel 的真实长度
        ori_mel_lens = torch.tensor([m.shape[0] for m in prompt_mels_for_flow_ori], device='cuda')

        # 2. 计算目标长度逻辑 (对齐 Speech Token * 2)
        # prompt_speech_tokens_lens_ori 是量化器返回的，形状为 [Batch]
        target_mel_lens = prompt_speech_tokens_lens_ori * 2

        # 3. 确定最终长度：取 target 和 original 的最小值 (对应原 if-else 逻辑)
        final_mel_lens = torch.min(target_mel_lens, ori_mel_lens)
        final_token_lens = final_mel_lens // 2

        # 4. 向量化截断 Speech Tokens
        # 这里的 prompt_speech_tokens_ori 形状通常是 [Batch, Max_Token_Len]
        batch_indices = torch.arange(prompt_speech_tokens_ori.size(1), device='cuda')[None, :]
        token_mask = batch_indices < final_token_lens[:, None]
        prompt_speech_tokens_batch = prompt_speech_tokens_ori * token_mask

        # 5. 向量化截断 Mel Features
        mel_indices = torch.arange(prompt_mels_for_flow_ori_tensor.size(1), device='cuda')[None, :]
        mel_mask = mel_indices < final_mel_lens[:, None]
        prompt_mels_for_flow_batch = prompt_mels_for_flow_ori_tensor * mel_mask.unsqueeze(-1)

        # 6. 还原为后续逻辑需要的 List 格式 (如果后续 LLM 还是逐个处理，则保持 list)
        # 如果后续 LLM/Flow 已经支持 Batch，建议直接使用 Tensor 提高效率
        prompt_speech_tokens = [t[:l] for t, l in zip(prompt_speech_tokens_batch, final_token_lens)]
        prompt_mels_for_flow = [m[:l] for m, l in zip(prompt_mels_for_flow_batch, final_mel_lens)]
        prompt_mels_lens_for_flow = final_mel_lens.unsqueeze(-1) # 保持 Tensor [Batch, 1] 格式


        # Prepare LLM inputs
        prompt_inputs = []

        for i in range(prompt_size):
            speech_tokens_i = [token+self.config.hf_config.speech_token_offset for token in prompt_speech_tokens[i].tolist()]
            speech_tokens_i += [self.config.hf_config.eos_token_id]
            if use_dialect_prompt and len(dialect_prompt_text_tokens_for_llm[i])>0:
                dialect_prompt_input = prompt_text_tokens_for_llm[i] + speech_tokens_i + dialect_prompt_text_tokens_for_llm[i]
                if i>0:
                    dialect_prompt_input = dialect_prefix[0] + dialect_prompt_input
                prompt_input = self.llm.generate(dialect_prompt_input, sampling_params, past_key_values=None)['token_ids']
                prompt_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
            else:
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )

        preprocessing_time = time.time()
        logging.info(f"LLM input processing completed in {preprocessing_time-s3audio_tokenization_time:.4f} seconds")

        # LLM generation - 批量处理，支持状态管理
        current_prompt_inputs = list(chain.from_iterable(prompt_inputs))

        # 处理inputs_prompt（支持多轮对话上下文）
        if inputs_prompt is None:
            # 首次调用，使用当前prompt
            final_inputs_prompt = current_prompt_inputs
        else:
            # 多轮调用，append到之前的inputs_prompt后面
            final_inputs_prompt = list(inputs_prompt)  # 复制
            final_inputs_prompt.extend(current_prompt_inputs)  # append形式

        # 构建批量输入 - VLLM引擎接受list输入
        inputs = [final_inputs_prompt + t for t in text_tokens_for_llm]

        llm_outputs = self.llm.generate(inputs, sampling_params)  # 批量LLM生成

        llm_processing_time = time.time()
        logging.info(f"Batch LLM generated input {llm_processing_time-preprocessing_time:.4f} seconds")

        # Prepare Flow inputs - 批量处理
        generated_speech_tokens = [
            [token - self.config.hf_config.speech_token_offset for token in tokens[:-1]]
            for tokens in llm_outputs['token_ids']
        ]

        # 准备批量Flow数据
        padding_start_indices = []
        combined_list = []

        for i, g_list in enumerate(generated_speech_tokens):
            turn_spk = spk_ids[i] if isinstance(spk_ids[i], int) else spk_ids[i][0]
            prompt_speech_token = prompt_speech_tokens[turn_spk].tolist()
            p_t = torch.tensor(prompt_speech_token)
            g_t = torch.tensor(g_list)
            combined_sample = torch.cat([p_t, g_t], dim=0)
            padding_start_indices.append(combined_sample.size(0))
            combined_list.append(combined_sample)

        # Flow input preparation
        flow_input = pad_sequence(combined_list, batch_first=True, padding_value=0)
        flow_inputs_len = torch.tensor([flow_input.shape[1]])

        # Flow generation - 批量处理
        start_idx = spk_ids[0] if isinstance(spk_ids[0], int) else spk_ids[0][0]
        prompt_mels = prompt_mels_for_flow[start_idx][None]
        prompt_mels_lens = prompt_mels_lens_for_flow[start_idx][None]
        spk_emb = spk_emb_for_flow[start_idx:start_idx+1]

        batch_flow_input = flow_input.shape[0]
        flow_inputs_len = torch.full((batch_flow_input, 1), flow_input.shape[1], dtype=torch.long, device=flow_input.device)
        mel = prompt_mels.repeat(batch_flow_input, 1, 1)

        # Flow generation
        with torch.amp.autocast("cuda", dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
            generated_mels, generated_mels_lens = self.flow(
                flow_input.cuda(), flow_inputs_len.cuda(),
                mel, prompt_mels_lens.expand(batch_flow_input,1), spk_emb.cuda(),
                streaming=False, finalize=True
            )
        flow_generation_time = time.time()
        logging.info(f"Flow mel feature generation completed in {flow_generation_time-llm_processing_time:.4f} seconds")

        # HiFi-GAN generation - 批量处理
        mel = generated_mels[:, :, prompt_mels_lens[0].item():generated_mels_lens[0].item()]
        wav, _ = self.hift(speech_feat=mel)

        # 使用原有的截取逻辑 - 修复批量处理中的索引问题
        # 每个请求对应的说话人的final_token_lens
        padding_start_indices = [int((t - final_token_lens[spk_ids[i] if isinstance(spk_ids[i], int) else spk_ids[i][0]]) * 1.8 * 480) for i, t in enumerate(padding_start_indices)]
        wav_list = list(wav)
        final_wavs = []
        for i in range(len(wav_list)):
            token_end_idx = padding_start_indices[i]
            trimmed_wav = wav_list[i][:token_end_idx]
            final_wavs.append(trimmed_wav)

        hifi_generation_time = time.time()
        logging.info(f"HIFI sampling generation completed in {hifi_generation_time-flow_generation_time:.4f} seconds")

        # 转换为numpy数组列表
        batch_results = [wav.cpu().numpy() for wav in final_wavs]

        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time

        logging.info(f"Batch parallel processing completed: {len(batch_results)} results in {total_batch_time:.3f}s")

        # 根据return_state决定返回格式
        if not return_state:
            # 兼容原有接口，只返回音频结果列表
            return batch_results
        else:
            # 返回包含状态的完整信息
            # 更新inputs_prompt，添加生成的token用于下一轮
            updated_inputs_prompt = final_inputs_prompt
            for i, tokens in enumerate(llm_outputs['token_ids']):
                updated_inputs_prompt.extend(text_tokens_for_llm[i][:-1])  # 去掉最后的audio_start token
                updated_inputs_prompt.extend(tokens)  # 添加生成的tokens

            return {
                'audio_results': batch_results,
                'inputs_prompt': updated_inputs_prompt,
                'llm_outputs': llm_outputs,
                'generation_info': {
                    'total_time': total_batch_time,
                    'segments_processed': turn_size,
                    'final_prompt_length': len(updated_inputs_prompt)
                }
            }