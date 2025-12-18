"""
SoulXPodcast Batch Processing Pipeline

实现真正的跨请求批处理：
- 收集所有请求的所有turns
- 一次性batch推理所有turns
- 按request_id重新组装结果
"""
import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import s3tokenizer

from soulxpodcast.config import Config, SamplingParams
from soulxpodcast.engine.llm_engine_batch import VLLMBatchEngine
from soulxpodcast.engine.batch_scheduler import BatchScheduler

logger = logging.getLogger(__name__)


@dataclass
class TurnItem:
    """单个turn的数据项"""
    request_idx: int  # 请求索引
    turn_idx: int     # Turn索引

    # LLM输入
    llm_input_tokens: List[int]

    # Flow输入
    prompt_speech_tokens: List[int]
    prompt_mel: torch.Tensor
    prompt_mel_len: torch.Tensor
    spk_emb: torch.Tensor

    # 元数据
    spk_id: int


class SoulXPodcastBatch:
    """
    SoulXPodcast批处理类

    实现真正的跨请求批处理
    """

    def __init__(self, config: Config, max_batch_size: int = 64):
        """
        Args:
            config: 配置对象
            max_batch_size: 最大batch大小
        """
        self.config = config

        # 初始化BatchScheduler
        logger.info("Initializing BatchScheduler...")
        self.scheduler = BatchScheduler(
            min_batch_size=1,
            max_batch_size=max_batch_size,
            memory_margin=0.2,
            enable_dynamic=True,
        )
        logger.info(f"BatchScheduler initialized: current_batch_size={self.scheduler.current_batch_size}")

        # 使用批处理版本的LLM引擎
        logger.info("Initializing batch LLM engine...")
        self.llm = VLLMBatchEngine(
            model=config.model,
            hf_config=config.hf_config,
            enforce_eager=config.enforce_eager,
            max_model_len=8192,
            enable_prefix_caching=True,
            logits_processors=["vllm_ras_logits_processor_fixed:FixedRASLogitsProcessor"]
        )

        # Audio tokenizer
        logger.info("Loading audio tokenizer...")
        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda().eval()

        # Flow model
        logger.info("Loading Flow model...")
        from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec
        self.flow = CausalMaskedDiffWithXvec()
        if config.hf_config.fp16_flow:
            logger.info("Casting flow to fp16")
            self.flow.half()
        self.flow.load_state_dict(
            torch.load(f"{config.model}/flow.pt", map_location="cpu", weights_only=True),
            strict=True
        )
        self.flow.cuda().eval()

        # HiFiGAN
        logger.info("Loading HiFiGAN...")
        from soulxpodcast.models.modules.hifigan import HiFTGenerator
        self.hift = HiFTGenerator()
        hift_state_dict = {
            k.replace('generator.', ''): v
            for k, v in torch.load(f"{config.model}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

        logger.info("SoulXPodcastBatch initialized successfully")

    @torch.inference_mode()
    def forward_batch(
        self,
        batch_data: List[Dict[str, Any]],
        sampling_params: SamplingParams,
    ) -> List[Dict[str, Any]]:
        """
        真正的批量处理：跨请求批处理所有turns

        Args:
            batch_data: 批量数据列表

        Returns:
            批量结果列表
        """
        total_requests = len(batch_data)
        logger.info(f"=== Starting True Batch Processing for {total_requests} requests ===")

        start_time = time.time()

        # ========== 阶段1: 准备所有请求的数据 ==========
        logger.info("[Phase 1] Preparing data for all requests...")
        prep_start = time.time()

        all_turn_items = self._prepare_all_turns(batch_data)
        total_turns = len(all_turn_items)

        prep_time = time.time() - prep_start
        logger.info(f"[Phase 1] Prepared {total_turns} turns from {total_requests} requests in {prep_time:.4f}s")

        # ========== 阶段2: Batch LLM推理所有turns ==========
        logger.info(f"[Phase 2] Batch LLM inference for {total_turns} turns...")
        llm_start = time.time()

        llm_results = self._batch_llm_inference(all_turn_items, sampling_params)

        llm_time = time.time() - llm_start
        logger.info(f"[Phase 2] Batch LLM completed in {llm_time:.4f}s, avg={llm_time/total_turns:.4f}s per turn")

        # ========== 阶段3: Batch Flow生成 ==========
        logger.info(f"[Phase 3] Batch Flow generation for {total_turns} turns...")
        flow_start = time.time()

        flow_results = self._batch_flow_generation(all_turn_items, llm_results)

        flow_time = time.time() - flow_start
        logger.info(f"[Phase 3] Batch Flow completed in {flow_time:.4f}s, avg={flow_time/total_turns:.4f}s per turn")

        # ========== 阶段4: Batch HiFiGAN生成 ==========
        logger.info(f"[Phase 4] Batch HiFiGAN generation for {total_turns} turns...")
        hifi_start = time.time()

        wav_results = self._batch_hifigan_generation(all_turn_items, flow_results)

        hifi_time = time.time() - hifi_start
        logger.info(f"[Phase 4] Batch HiFiGAN completed in {hifi_time:.4f}s, avg={hifi_time/total_turns:.4f}s per turn")

        # ========== 阶段5: 重新组装结果 ==========
        logger.info(f"[Phase 5] Reassembling results by request...")
        assemble_start = time.time()

        final_results = self._reassemble_results(batch_data, all_turn_items, wav_results)

        assemble_time = time.time() - assemble_start
        logger.info(f"[Phase 5] Reassembly completed in {assemble_time:.4f}s")

        # ========== 总结 ==========
        total_time = time.time() - start_time
        logger.info(f"=== Batch Processing Completed ===")
        logger.info(f"  Total time: {total_time:.4f}s")
        logger.info(f"  Avg per request: {total_time/total_requests:.4f}s")
        logger.info(f"  Avg per turn: {total_time/total_turns:.4f}s")
        logger.info(f"  Breakdown:")
        logger.info(f"    - Preparation: {prep_time:.4f}s ({prep_time/total_time*100:.1f}%)")
        logger.info(f"    - LLM: {llm_time:.4f}s ({llm_time/total_time*100:.1f}%)")
        logger.info(f"    - Flow: {flow_time:.4f}s ({flow_time/total_time*100:.1f}%)")
        logger.info(f"    - HiFiGAN: {hifi_time:.4f}s ({hifi_time/total_time*100:.1f}%)")
        logger.info(f"    - Assembly: {assemble_time:.4f}s ({assemble_time/total_time*100:.1f}%)")

        return final_results

    def _prepare_all_turns(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> List[TurnItem]:
        """
        准备所有请求的所有turns

        收集所有turns到一个列表，每个turn带有request_idx标记
        """
        all_turn_items = []

        for req_idx, req_data in enumerate(batch_data):
            prompt_mels_for_llm = req_data["prompt_mels_for_llm"]
            prompt_mels_lens_for_llm = req_data["prompt_mels_lens_for_llm"]
            prompt_text_tokens_for_llm = req_data["prompt_text_tokens_for_llm"]
            text_tokens_for_llm = req_data["text_tokens_for_llm"]
            prompt_mels_for_flow = req_data["prompt_mels_for_flow"]
            spk_emb_for_flow = req_data["spk_emb_for_flow"]
            spk_ids = req_data["spk_ids"]

            prompt_size = len(prompt_text_tokens_for_llm)
            turn_size = len(text_tokens_for_llm)

            # Audio tokenization
            prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
                prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
            )

            # Align speech tokens
            prompt_speech_tokens = []
            prompt_mels_for_flow_aligned = []
            prompt_mels_lens_for_flow = []

            for prompt_idx in range(prompt_size):
                prompt_speech_token_len = prompt_speech_tokens_lens_ori[prompt_idx].item()
                prompt_speech_token = prompt_speech_tokens_ori[prompt_idx, :prompt_speech_token_len]
                prompt_mel = prompt_mels_for_flow[prompt_idx]
                prompt_mel_len = prompt_mel.shape[0]

                if prompt_speech_token_len * 2 > prompt_mel_len:
                    prompt_speech_token = prompt_speech_token[:int(prompt_mel_len/2)]
                    prompt_mel_len_tensor = torch.tensor([prompt_mel_len]).cuda()
                else:
                    prompt_mel = prompt_mel.detach().clone()[:prompt_speech_token_len * 2].cuda()
                    prompt_mel_len_tensor = torch.tensor([prompt_speech_token_len * 2]).cuda()

                prompt_speech_tokens.append(prompt_speech_token)
                prompt_mels_for_flow_aligned.append(prompt_mel)
                prompt_mels_lens_for_flow.append(prompt_mel_len_tensor)

            # Prepare LLM inputs
            prompt_inputs = []
            for i in range(prompt_size):
                speech_tokens_i = [
                    token + self.config.hf_config.speech_token_offset
                    for token in prompt_speech_tokens[i].tolist()
                ]
                speech_tokens_i += [self.config.hf_config.eos_token_id]
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i)

            # 构建每个turn的输入
            llm_inputs = []
            for item in prompt_inputs:
                llm_inputs.extend(item)

            for turn_idx in range(turn_size):
                # 添加当前turn的text tokens
                llm_inputs.extend(text_tokens_for_llm[turn_idx])

                # 创建TurnItem
                turn_spk = spk_ids[turn_idx]

                turn_item = TurnItem(
                    request_idx=req_idx,
                    turn_idx=turn_idx,
                    llm_input_tokens=llm_inputs.copy(),
                    prompt_speech_tokens=prompt_speech_tokens[turn_spk].tolist(),
                    prompt_mel=prompt_mels_for_flow_aligned[turn_spk],
                    prompt_mel_len=prompt_mels_lens_for_flow[turn_spk],
                    spk_emb=spk_emb_for_flow[turn_spk:turn_spk+1],
                    spk_id=turn_spk,
                )
                all_turn_items.append(turn_item)

                # 将LLM输出添加到输入（为下一个turn做准备）
                # 注意：这里我们暂时不知道输出，所以先不添加

        logger.info(f"Collected {len(all_turn_items)} turns from {len(batch_data)} requests")
        return all_turn_items

    def _batch_llm_inference(
        self,
        turn_items: List[TurnItem],
        sampling_params: SamplingParams,
    ) -> List[List[int]]:
        """
        Batch LLM推理所有turns

        Args:
            turn_items: 所有turn项目
            sampling_params: 采样参数

        Returns:
            所有turn的LLM输出token列表
        """
        # 准备batch输入
        prompts = [item.llm_input_tokens for item in turn_items]
        sampling_params_list = [sampling_params] * len(turn_items)

        # Batch推理
        batch_results = self.llm.generate_batch(prompts, sampling_params_list)

        # 提取token_ids
        llm_outputs = [result["token_ids"] for result in batch_results]

        return llm_outputs

    def _batch_flow_generation(
        self,
        turn_items: List[TurnItem],
        llm_results: List[List[int]],
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Batch Flow生成

        Args:
            turn_items: 所有turn项目
            llm_results: LLM输出结果

        Returns:
            所有turn的Flow输出 [(mel, mel_len), ...]
        """
        batch_size = len(turn_items)

        # 准备Flow inputs
        flow_inputs = []
        flow_lens = []
        prompt_mels_batch = []
        prompt_mel_lens_batch = []
        spk_embs_batch = []

        for turn_item, llm_tokens in zip(turn_items, llm_results):
            # 从LLM输出提取speech tokens
            generated_speech_tokens = [
                token - self.config.hf_config.speech_token_offset
                for token in llm_tokens[:-1]  # ignore last eos
            ]

            # 构建Flow input
            flow_input = torch.tensor(turn_item.prompt_speech_tokens + generated_speech_tokens)
            flow_input_len = len(turn_item.prompt_speech_tokens) + len(generated_speech_tokens)

            flow_inputs.append(flow_input)
            flow_lens.append(flow_input_len)
            prompt_mels_batch.append(turn_item.prompt_mel)
            prompt_mel_lens_batch.append(turn_item.prompt_mel_len)
            spk_embs_batch.append(turn_item.spk_emb)

        # Batch处理Flow (需要padding)
        max_len = max(flow_lens)
        flow_inputs_padded = torch.nn.utils.rnn.pad_sequence(
            flow_inputs, batch_first=True, padding_value=0
        ).cuda()
        flow_lens_tensor = torch.tensor(flow_lens).cuda()

        # Prompt mels batching
        prompt_mels_tensor = torch.stack(prompt_mels_batch).cuda()
        prompt_mel_lens_tensor = torch.stack(prompt_mel_lens_batch).cuda()
        spk_embs_tensor = torch.stack([emb.squeeze(0) for emb in spk_embs_batch]).cuda()

        # Flow forward
        with torch.amp.autocast("cuda", dtype=torch.float16 if self.config.hf_config.fp16_flow else torch.float32):
            generated_mels, generated_mels_lens = self.flow(
                flow_inputs_padded,
                flow_lens_tensor,
                prompt_mels_tensor,
                prompt_mel_lens_tensor,
                spk_embs_tensor,
                streaming=False,
                finalize=True
            )

        # 分离batch结果
        flow_results = []
        for i in range(batch_size):
            flow_results.append((
                generated_mels[i],
                generated_mels_lens[i].item()
            ))

        return flow_results

    def _batch_hifigan_generation(
        self,
        turn_items: List[TurnItem],
        flow_results: List[Tuple[torch.Tensor, int]],
    ) -> List[torch.Tensor]:
        """
        Batch HiFiGAN生成

        Args:
            turn_items: 所有turn项目
            flow_results: Flow输出结果

        Returns:
            所有turn的wav结果
        """
        wav_results = []

        # HiFiGAN需要逐个处理（因为mel长度不同）
        for turn_item, (generated_mel, generated_mel_len) in zip(turn_items, flow_results):
            # 提取有效mel区域
            mel = generated_mel[:, :, turn_item.prompt_mel_len[0].item():generated_mel_len]
            mel = mel.unsqueeze(0)  # 添加batch维度

            # HiFiGAN生成
            wav, _ = self.hift(speech_feat=mel)
            wav_results.append(wav)

        return wav_results

    def _reassemble_results(
        self,
        batch_data: List[Dict[str, Any]],
        turn_items: List[TurnItem],
        wav_results: List[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """
        按request_idx重新组装结果

        将所有turns的wav按照request分组并拼接
        """
        num_requests = len(batch_data)

        # 按request_idx分组
        request_wavs = [[] for _ in range(num_requests)]
        request_turn_counts = [0] * num_requests

        for turn_item, wav in zip(turn_items, wav_results):
            request_wavs[turn_item.request_idx].append(wav)
            request_turn_counts[turn_item.request_idx] += 1

        # 拼接每个request的wavs
        final_results = []
        for req_idx in range(num_requests):
            wavs = request_wavs[req_idx]

            if not wavs:
                logger.warning(f"Request {req_idx} has no turns!")
                final_results.append({
                    "audio": np.array([]),
                    "sample_rate": 24000,
                    "num_turns": 0,
                })
                continue

            # 拼接所有turns的音频
            target_audio = None
            for wav in wavs:
                if target_audio is None:
                    target_audio = wav
                else:
                    target_audio = torch.concat([target_audio, wav], axis=1)

            final_results.append({
                "audio": target_audio.cpu().squeeze(0).numpy(),
                "sample_rate": 24000,
                "num_turns": len(wavs),
            })

        logger.info(f"Reassembled {num_requests} requests")
        return final_results
