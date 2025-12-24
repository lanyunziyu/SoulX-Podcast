import time
from datetime import datetime
from typing import Dict
import asyncio
import concurrent.futures
from tqdm import tqdm
from itertools import chain
from soulxpodcast.utils.dataloader import SPK_DICT, TEXT_START, TEXT_END, AUDIO_START
from soulxpodcast.utils.text import normalize_text
from torch.nn.utils.rnn import pad_sequence
import s3tokenizer
import torch
import logging
from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.engine.llm_engine import  VLLMEngine
from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec
from soulxpodcast.models.modules.hifigan import HiFTGenerator

class SoulXPodcast(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda().eval()
        if self.config.llm_engine == "vllm":
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
        self.audio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=30,
            thread_name_prefix="FlowHifiWorker"
        )

    
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
            else:
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )

        preprocessing_time = time.time()
        logging.info(f"LLM input processing completed in {preprocessing_time-s3audio_tokenization_time:.4f} seconds")
        generated_wavs, results_dict = [], {}

        # LLM generation
        inputs_prompt = list(chain.from_iterable(prompt_inputs))
        for i in range(1):

            inputs = [inputs_prompt + t for t in text_tokens_for_llm]
            llm_outputs = self.llm.generate(inputs, sampling_params)
            llm_processing_time = time.time()
            logging.info(f"LLM generated input {llm_processing_time-preprocessing_time:.4f} seconds")
            
            # Prepare Flow inputs
            turn_spk = spk_ids[i]
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

    def prepare_llm_inputs(
        self,
        segment: Dict,
        mode: str,
        inputs_prompt: list = None,
        preloaded_data: Dict = None
    ):
        """
        准备LLM输入（阶段1：数据预处理）

        Args:
            segment: 当前段落信息 {'text': '...', 'speaker': 0}
            mode: 模式参数
            inputs_prompt: 可选的已有inputs_prompt（多轮对话）
            preloaded_data: 可选的预加载数据，如果None则从self获取

        Returns:
            llm_inputs: LLM输入token列表
            context: flow/hifi所需上下文信息
        """
        import s3tokenizer
        from itertools import chain

        # 获取预加载数据
        if preloaded_data is None:
            from api.service import get_service
            service = get_service()
            preloaded_data = service.preloaded_data[mode]


        spk_id = segment.get('speaker', segment.get('spk_id', 0))
        text = normalize_text(segment['text'])
        text = f"{SPK_DICT[spk_id]}{TEXT_START}{text}{TEXT_END}{AUDIO_START}"
        text_tokens = self.llm.tokenizer.encode(text)

        # 准备prompt_mels数据
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(preloaded_data["log_mel_list"])
        spk_emb_for_flow = torch.tensor(preloaded_data["spk_emb_list"])
        prompt_mels_for_flow = pad_sequence(
            preloaded_data["mel_list"], batch_first=True, padding_value=0
        )

        # Audio tokenization（使用已有逻辑）
        prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )

        # 处理speech tokens和mels
        prompt_mels_for_flow_ori_tensor = prompt_mels_for_flow.cuda()
        ori_mel_lens = torch.tensor([m.shape[0] for m in preloaded_data["mel_list"]], device='cuda')
        target_mel_lens = prompt_speech_tokens_lens_ori * 2
        final_mel_lens = torch.min(target_mel_lens, ori_mel_lens)
        final_token_lens = final_mel_lens // 2

        # 截断tokens和mels
        batch_indices = torch.arange(prompt_speech_tokens_ori.size(1), device='cuda')[None, :]
        token_mask = batch_indices < final_token_lens[:, None]
        prompt_speech_tokens_batch = prompt_speech_tokens_ori * token_mask

        mel_indices = torch.arange(prompt_mels_for_flow_ori_tensor.size(1), device='cuda')[None, :]
        mel_mask = mel_indices < final_mel_lens[:, None]
        prompt_mels_for_flow_batch = prompt_mels_for_flow_ori_tensor * mel_mask.unsqueeze(-1)

        prompt_speech_tokens = [t[:l] for t, l in zip(prompt_speech_tokens_batch, final_token_lens)]
        prompt_mels_for_flow_list = [m[:l] for m, l in zip(prompt_mels_for_flow_batch, final_mel_lens)]

        # 准备LLM inputs
        prompt_inputs = []
        for i in range(len(prompt_speech_tokens)):
            speech_tokens_i = [token + self.config.hf_config.speech_token_offset
                             for token in prompt_speech_tokens[i].tolist()]
            speech_tokens_i += [self.config.hf_config.eos_token_id]
            prompt_inputs.append(preloaded_data["prompt_text_ids_list"][i] + speech_tokens_i)

        current_prompt_inputs = list(chain.from_iterable(prompt_inputs))

        # 处理inputs_prompt（支持多轮对话上下文）
        if inputs_prompt is None:
            final_inputs_prompt = current_prompt_inputs
        else:
            final_inputs_prompt = list(inputs_prompt)
            final_inputs_prompt.extend(current_prompt_inputs)

        # 构建LLM输入
        llm_inputs = final_inputs_prompt + text_tokens

        # 准备context（flow/hifi所需）
        context = {
            'prompt_speech_tokens': prompt_speech_tokens,
            'prompt_mels_for_flow': prompt_mels_for_flow_list,
            'prompt_mels_lens_for_flow': final_mel_lens.unsqueeze(-1),
            'spk_emb_for_flow': spk_emb_for_flow,
            'spk_id': spk_id,
            'final_token_lens': final_token_lens,
            'speech_token_offset': self.config.hf_config.speech_token_offset,
        }

        return llm_inputs, context

    async def process_llm_result_async(
        self,
        llm_output: Dict,
        context: Dict
    ):
        """
        处理LLM结果（阶段3：flow + hifi异步执行）

        Args:
            llm_output: LLM输出 {'token_ids': [...], 'text': '...'}
            context: 从prepare_llm_inputs返回的上下文

        Returns:
            audio: numpy音频数组

        注意：flow和hifi是GPU同步计算，我们使用线程池实现真正的异步
        """

        def _process_sync(llm_output,context):
            with torch.inference_mode():
                # 提取context信息
                prompt_speech_tokens = context['prompt_speech_tokens']
                prompt_mels_for_flow = context['prompt_mels_for_flow']
                prompt_mels_lens_for_flow = context['prompt_mels_lens_for_flow']
                spk_emb_for_flow = context['spk_emb_for_flow']
                spk_id = context['spk_id']
                final_token_lens = context['final_token_lens']
                speech_token_offset = context['speech_token_offset']
                llm_processing_time = time.time()
                # 准备flow输入
                generated_speech_tokens = [token - speech_token_offset for token in llm_output['token_ids'][:-1]]
                prompt_speech_token = prompt_speech_tokens[spk_id].tolist()
                p_t = torch.tensor(prompt_speech_token)
                g_t = torch.tensor(generated_speech_tokens)
                combined_sample = torch.cat([p_t, g_t], dim=0)
                padding_start_idx = combined_sample.size(0)

                flow_input = combined_sample.unsqueeze(0)  # [1, seq_len]
                flow_inputs_len = torch.tensor([[flow_input.shape[1]]])

                # Flow生成
                prompt_mels = prompt_mels_for_flow[spk_id][None]
                prompt_mels_lens = prompt_mels_lens_for_flow[spk_id][None]
                spk_emb = spk_emb_for_flow[spk_id:spk_id+1]

                with torch.amp.autocast("cuda", dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
                    generated_mels, generated_mels_lens = self.flow(
                        flow_input.cuda(), flow_inputs_len.cuda(),
                        prompt_mels, prompt_mels_lens, spk_emb.cuda(),
                        streaming=False, finalize=True
                    )
                flow_generation_time = time.time()
                logging.info(f"Flow mel feature generation completed in {flow_generation_time-llm_processing_time:.4f} seconds")
                # HiFi-GAN生成
                mel = generated_mels[:, :, prompt_mels_lens[0].item():generated_mels_lens[0].item()]
                wav, _ = self.hift(speech_feat=mel)

                # 截断音频
                token_end_idx = int(((padding_start_idx - final_token_lens[spk_id]) * 2 - 2)* 480)
                trimmed_wav = wav[0][:token_end_idx]
                hifi_generation_time = time.time()
                logging.info(f"HIFI sampling generation completed in {hifi_generation_time-flow_generation_time:.4f} seconds")                

                return trimmed_wav.cpu().numpy()

        # 在线程池中执行同步GPU计算
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.audio_executor, 
            _process_sync, 
            llm_output, 
            context
        )
        return result

    def update_inputs_prompt(self, inputs_prompt: list, generated_tokens: list):
        """
        更新多轮对话的inputs_prompt

        Args:
            inputs_prompt: 当前的inputs_prompt
            generated_tokens: 新生成的tokens

        Returns:
            updated_inputs_prompt: 更新后的inputs_prompt
        """
        if inputs_prompt is None:
            return generated_tokens[:]

        updated = list(inputs_prompt)
        updated.extend(generated_tokens)
        return updated