"""
SoulXPodcast Model Service Layer
"""
import re
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
import random
import concurrent.futures
import gc
import threading
import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import PodcastInferHandler
from soulxpodcast.utils.dataloader import SPK_DICT, TEXT_START, TEXT_END, AUDIO_START
from soulxpodcast.utils.text import normalize_text

from api.config import config as api_config
from api.utils import parse_dialogue_text
import operator
logger = logging.getLogger(__name__)


class SoulXPodcastService:
    """SoulXPodcast模型服务单例"""

    # _instance: Optional['SoulXPodcastService'] = None
    # _lock = threading.Lock()  # 线程锁

    # def __new__(cls):
        # with cls._lock:  # 确保线程安全
        #     if cls._instance is None:
        #         logger.info("Creating new SoulXPodcastService instance")
        #         cls._instance = super(SoulXPodcastService, cls).__new__(cls)
        #         cls._instance._initialized = False  # 实例属性
        #         # cls._instance._generation_lock = threading.Lock()  # 生成锁
        # return cls._instance

    def __init__(self):
        """初始化模型（单例模式，只初始化一次）"""
        # 使用实例属性而不是类属性
        self._load_model()
        logger.info("Initializing SoulXPodcastService (first time)")
        self._preload_data()  # 预加载数据

    def _load_model(self):
        """加载模型"""
        try:
            logger.info(f"Loading SoulXPodcast model from {api_config.model_path}...")
            logger.info(f"Using LLM engine: {api_config.llm_engine}")

            # 加载配置
            hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
                initial_values={"fp16_flow": api_config.fp16_flow},
                json_file=f"{api_config.model_path}/soulxpodcast_config.json"
            )

            # 创建Config对象
            model_config = Config(
                model=api_config.model_path,
                enforce_eager=True,
                llm_engine=api_config.llm_engine,
                hf_config=hf_config
            )

            # 初始化模型
            self.model = SoulXPodcast(model_config)
            self.dataset = PodcastInferHandler(
                self.model.llm.tokenizer,
                None,
                model_config
            )
            self.config = model_config

            logger.info(f"Model loaded successfully with {api_config.llm_engine} engine!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return hasattr(self, 'model') and self.model is not None

    def _preload_data(self):
        """
        预加载不同模式的数据
        模式参数规则 (三位数字):
        - 第一位: 0=单人, 1=双人
        - 第二位: 0=男生, 1=女生 (单人时有效), 2=占位符 (双人时)
        - 第三位: 0=普通话, 1=英语

        示例:
        - "100": 单人男生普通话
        - "110": 单人女生普通话
        - "101": 单人男生英语
        - "120": 双人普通话
        - "121": 双人英语
        """
        try:
            import json
            from pathlib import Path

            logger.info("Preloading data for different modes...")
            self.preloaded_data = {}

            base_path = Path(__file__).parent.parent

            # 定义模式到文件的映射
            mode_configs = {
                "000": {  # 单人男生普通话
                    "script_path": base_path / "example/mel_sigal_script/script_mandarin.json",
                    "description": "单人男生普通话"
                },
                "001": {  # 单人男生英语
                    "script_path": base_path / "example/mel_sigal_script/script_english.json",
                    "description": "单人男生英语"
                },
                "010": {  # 单人女生普通话
                    "script_path": base_path / "example/femel_sigal_script/script_mandarin.json",
                    "description": "单人女生普通话"
                },
                "011": {  # 单人女生英语
                    "script_path": base_path / "example/femel_sigal_script/script_english.json",
                    "description": "单人女生英语"
                },
                "120": {  # 双人普通话
                    "script_path": base_path / "example/podcast_script/script_mandarin.json",
                    "description": "双人普通话"
                },
                "121": {  # 双人英语
                    "script_path": base_path / "example/podcast_script/script_english.json",
                    "description": "双人英语"
                }
            }

            # 预加载每个模式的数据
            for mode, config in mode_configs.items():
                script_path = config["script_path"]

                if not script_path.exists():
                    logger.warning(f"Script file not found for mode {mode}: {script_path}")
                    continue

                # 读取JSON配置
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)

                # 解析speakers数据
                speakers = script_data.get("speakers", {})

                # 构建数据项列表
                prompt_audio_paths = []
                prompt_texts = []

                for spk_id in sorted(speakers.keys()):
                    spk_info = speakers[spk_id]
                    prompt_audio_path = str(base_path / spk_info["prompt_audio"])
                    prompt_text = spk_info["prompt_text"]

                    prompt_audio_paths.append(prompt_audio_path)
                    prompt_texts.append(prompt_text)

                # 处理数据获取特征
                if prompt_audio_paths:
                    # 构建临时数据项用于提取特征
                    temp_dataitem = {
                        "key": f"preload_{mode}",
                        "prompt_text": prompt_texts,
                        "prompt_wav": prompt_audio_paths,
                        "text": [""],  # 占位符
                        "spk": [0],    # 占位符
                    }

                    # 使用dataset处理数据
                    self.dataset.update_datasource([temp_dataitem])
                    data = self.dataset[0]

                    if data is not None:
                        # 保存预处理的特征
                        self.preloaded_data[mode] = {
                            "prompt_text_ids_list": data["prompt_text_tokens"],
                            "spk_emb_list": data["spk_emb"],
                            "mel_list": data["mel"],
                            "mel_len_list": data["mel_len"],
                            "log_mel_list": data["log_mel"],
                            "prompt_audio_paths": prompt_audio_paths,
                            "prompt_texts": prompt_texts,
                            "description": config["description"]
                        }
                        logger.info(f"✓ Preloaded mode {mode}: {config['description']}")
                    else:
                        logger.warning(f"Failed to process data for mode {mode}")

            logger.info(f"Data preloading completed. {len(self.preloaded_data)} modes loaded.")

        except Exception as e:
            logger.error(f"Failed to preload data: {e}", exc_info=True)
            # 不抛出异常,允许服务继续运行
            self.preloaded_data = {}

    def _prepare_batch_requests(self, batch_requests: List[dict]):
        """
        预处理批量请求，解析文本并记录每个请求的片段信息

        Args:
            batch_requests: 批量请求列表

        Returns:
            tuple: (all_text_ids_list, all_spks_lists, request_segment_mapping)
        """
        logger.info(f"Preparing {len(batch_requests)} batch requests...")

        # 准备批量对话文本
        batch_dialogue_texts = []
        for i, request in enumerate(batch_requests):
            dialogue_text = request.get('dialogue_text', '').strip()
            if not dialogue_text:
                raise ValueError(f"批量请求{i}缺少dialogue_text")
            batch_dialogue_texts.append(dialogue_text)

        # 预处理所有文本，并记录每个请求的片段信息
        all_text_ids_list = []
        all_spks_lists = []
        request_segment_mapping = []  # 记录每个请求包含的片段数量和起始索引
        current_segment_idx = 0

        for batch_idx, dialogue_text in enumerate(batch_dialogue_texts):
            # 解析对话文本
            target_text_list = parse_dialogue_text(dialogue_text, 0)
            logger.info(f"Request {batch_idx}: Parsed dialogue into {len(target_text_list)} segments")

            # 提取说话人和文本
            spks, texts = [], []
            for target_text in target_text_list:
                pattern = r'(\[S[1-9]\])(.+)'
                match = re.match(pattern, target_text)
                if match:
                    text, spk = match.group(2), int(match.group(1)[2]) - 1
                    spks.append(spk)
                    texts.append(text)
                else:
                    raise ValueError(f"无效的对话文本格式: {target_text}")

            text_ids_list, spks_list = [], []
            for text, spk in zip(texts, spks):
                text = normalize_text(text)
                text = f"{SPK_DICT[spk]}{TEXT_START}{text}{TEXT_END}{AUDIO_START}"
                text_ids = self.model.llm.tokenizer.encode(text)
                text_ids_list.append(text_ids)
                spks_list.append(spk)

            # 记录当前请求的片段信息
            segment_count = len(text_ids_list)
            request_segment_mapping.append({
                'request_idx': batch_idx,
                'start_segment_idx': current_segment_idx,
                'segment_count': segment_count,
                'end_segment_idx': current_segment_idx + segment_count,
                'dialogue_text': dialogue_text
            })
            current_segment_idx += segment_count

            all_text_ids_list.append(text_ids_list)
            all_spks_lists.append(spks_list)

        logger.info(f"Request segment mapping: {request_segment_mapping}")
        return all_text_ids_list, all_spks_lists, request_segment_mapping

    def _concatenate_batch_results(
        self,
        batch_audio_results: List[np.ndarray],
        request_segment_mapping: List[dict]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        根据请求片段映射信息，将生成的音频结果按请求拼接

        Args:
            batch_audio_results: 所有音频片段的生成结果
            request_segment_mapping: 请求片段映射信息

        Returns:
            List[Tuple[int, np.ndarray]]: 按请求分组并拼接后的音频结果
        """
        logger.info("Concatenating batch audio results...")

        final_results = []
        sample_rate = 24000

        for mapping in request_segment_mapping:
            request_idx = mapping['request_idx']
            start_idx = mapping['start_segment_idx']
            end_idx = mapping['end_segment_idx']
            segment_count = mapping['segment_count']

            logger.info(f"Processing request {request_idx}: segments {start_idx}-{end_idx-1} (total: {segment_count})")

            if segment_count == 1:
                # 单个片段，直接使用
                audio_array = batch_audio_results[start_idx]
                logger.info(f"Request {request_idx}: single segment, audio length: {len(audio_array) / sample_rate:.2f}s")
            else:
                # 多个片段，需要拼接
                segments_to_concat = batch_audio_results[start_idx:end_idx]

                # 使用与generate函数相同的拼接逻辑
                import torch
                target_audio = None
                for segment_audio in segments_to_concat:
                    segment_tensor = torch.from_numpy(segment_audio)
                    if target_audio is None:
                        target_audio = segment_tensor
                    else:
                        target_audio = torch.concat([target_audio, segment_tensor], axis=0)

                audio_array = target_audio.numpy()
                logger.info(f"Request {request_idx}: concatenated {segment_count} segments, total audio length: {len(audio_array) / sample_rate:.2f}s")

            final_results.append((sample_rate, audio_array))

        logger.info(f"Batch concatenation completed: {len(final_results)} concatenated audios")
        return final_results

    def _analyze_dialogue_mode(self, dialogue_text: str) -> dict:
        """
        分析对话文本的模式：单人说话 vs 多人对话

        Args:
            dialogue_text: 对话文本，包含[S1],[S2]等标记

        Returns:
            dict: {
                'is_multi_speaker': bool,  # 是否多人对话
                'speakers': list,          # 涉及的说话人列表
                'segments': list,          # 按说话人分组的文本段落
                'total_segments': int      # 总段落数
            }
        """
        import re

        # 解析所有说话人标记
        pattern = r'\[S([1-9])\]'
        matches = re.findall(pattern, dialogue_text)
        speakers = list(set([int(s)-1 for s in matches]))  # 去重并转为整数

        # 解析对话文本为段落
        target_text_list = parse_dialogue_text(dialogue_text, 0)
        segments = []

        for target_text in target_text_list:
            match = re.match(r'(\[S([1-9])\])(.+)', target_text)
            if match:
                spk_id = int(match.group(2))-1
                text_content = match.group(3)
                segments.append({
                    'speaker': spk_id,
                    'text': text_content,
                    'original': target_text
                })

        is_multi_speaker = len(speakers) > 1

        return {
            'is_multi_speaker': is_multi_speaker,
            'speakers': speakers,
            'segments': segments,
            'total_segments': len(segments)
        }

    def _create_dynamic_batch_scheduler(
        self,
        batch_requests: List[dict],
        target_batch_size: int = None
    ) -> dict:
        """
        创建动态批量调度器，处理单人和多人对话的混合批次

        Args:
            batch_requests: 原始批量请求
            target_batch_size: 目标批次大小，默认使用多人对话的最大词数

        Returns:
            dict: 调度信息，包含每轮需要处理的请求和状态
        """
        logger.info("Creating dynamic batch scheduler...")

        # 分析每个请求的对话模式
        request_analyses = []
        max_segments = 0

        for i, request in enumerate(batch_requests):
            dialogue_text = request.get('dialogue_text', '').strip()
            analysis = self._analyze_dialogue_mode(dialogue_text)
            analysis['request_idx'] = i
            analysis['original_request'] = request
            request_analyses.append(analysis)

            # 记录最大段落数（用于确定循环次数）
            max_segments = max(max_segments, analysis['total_segments'])

            logger.info(f"Request {i}: {'Multi-speaker' if analysis['is_multi_speaker'] else 'Single-speaker'} "
                       f"({analysis['total_segments']} segments, speakers: {analysis['speakers']})")

        # 分离单人和多人请求
        single_requests = [r for r in request_analyses if not r['is_multi_speaker']]
        multi_requests = [r for r in request_analyses if r['is_multi_speaker']]

        # 如果没有多人对话，直接使用原有逻辑
        if not multi_requests:
            return {
                'use_scheduler': False,
                'single_requests_only': True,
                'original_requests': batch_requests
            }

        # 设置目标批次大小（以多人对话的最大词数为准）
        if target_batch_size is None:
            target_batch_size = max_segments

        logger.info(f"Dynamic scheduler: {len(multi_requests)} multi-speaker, "
                   f"{len(single_requests)} single-speaker, "
                   f"max_segments={max_segments}, target_batch_size={target_batch_size}")

        # 构建每轮的调度计划
        round_plans = []

        for round_idx in range(max_segments):
            round_requests = []

            # 1. 添加多人对话的当前轮次段落
            for multi_req in multi_requests:
                if round_idx < multi_req['total_segments']:
                    segment = multi_req['segments'][round_idx]
                    round_requests.append({
                        'request_idx': multi_req['request_idx'],
                        'is_multi_speaker': True,
                        'round_idx': round_idx,
                        'speaker': segment['speaker'],
                        'text': segment['text'],
                        'dialogue_text': f"[S{segment['speaker']}]{segment['text']}",
                        'original_request': multi_req['original_request'],
                        'is_continuation': round_idx > 0
                    })

            # 2. 用单人请求填充批次到目标大小
            current_batch_size = len(round_requests)
            needed_single = target_batch_size - current_batch_size

            if needed_single > 0 and single_requests:
                # 循环使用单人请求填充
                single_idx = 0
                for _ in range(min(needed_single, len(single_requests))):
                    single_req = single_requests[single_idx % len(single_requests)]
                    round_requests.append({
                        'request_idx': single_req['request_idx'],
                        'is_multi_speaker': False,
                        'round_idx': round_idx,
                        'speaker': single_req['segments'][0]['speaker'],
                        'text': single_req['segments'][0]['text'],
                        'dialogue_text': single_req['original_request']['dialogue_text'],
                        'original_request': single_req['original_request'],
                        'is_continuation': False
                    })
                    single_idx += 1

            round_plans.append({
                'round_idx': round_idx,
                'requests': round_requests,
                'batch_size': len(round_requests)
            })

            logger.info(f"Round {round_idx}: {len(round_requests)} requests "
                       f"({sum(1 for r in round_requests if r['is_multi_speaker'])} multi-speaker)")

        return {
            'use_scheduler': True,
            'total_rounds': max_segments,
            'round_plans': round_plans,
            'multi_requests': multi_requests,
            'single_requests': single_requests,
            'target_batch_size': target_batch_size
        }

    def _execute_dynamic_scheduling(
        self,
        scheduler_info: dict,
        preloaded_features: dict,
        temperature: float,
        top_k: int,
        top_p: int,
        repetition_penalty: float
    ) -> List[Tuple[int, np.ndarray]]:
        """
        执行动态调度，按plan逐轮生成音频

        Plan逻辑：
        1. 第一轮：多人对话第1片段 + 单人对话完整内容 → 单人对话完成，返回结果
        2. 第二轮：使用inputs_prompt + 多人对话第2片段 → 生成第2个音频段
        3. 第N轮：使用inputs_prompt + 多人对话第N片段 → 生成第N个音频段
        4. 拼接多人对话的所有音频段成最终结果
        """
        logger.info("Starting dynamic scheduling execution...")

        # 初始化结果存储
        multi_audio_accumulator = {}  # {request_idx: [audio_segments]}
        single_audio_results = {}     # {request_idx: audio_result}
        inputs_prompt = None          # 共享状态

        # 按轮次执行
        target_batch_size = scheduler_info['target_batch_size']

        for round_idx in range(scheduler_info['total_rounds']):
            logger.info(f"=== Processing Round {round_idx} ===")

            # 构建当前轮的处理数据
            current_text_tokens = []
            current_spk_ids = []
            current_request_mapping = []  # 记录每个token对应的原始请求

            # 处理多人对话请求的当前片段
            for multi_req in scheduler_info['multi_requests']:
                request_idx = multi_req['request_idx']
                if round_idx < multi_req['total_segments']:
                    segment = multi_req['segments'][round_idx]

                    # 构造单个片段的dialogue_text处理
                    dialogue_text = f"[S{segment['speaker']}]{segment['text']}"
                    temp_request = [{'dialogue_text': dialogue_text}]
                    text_ids_list, spks_list, _ = self._prepare_batch_requests(temp_request)

                    # 添加到当前轮
                    current_text_tokens.extend(text_ids_list[0])  # 单个请求的tokens
                    current_spk_ids.extend(spks_list[0])          # 单个请求的spk_ids

                    # 记录映射关系
                    for _ in range(len(text_ids_list[0])):
                        current_request_mapping.append({
                            'request_idx': request_idx,
                            'is_multi_speaker': True,
                            'round_idx': round_idx
                        })

            # 填充单人对话请求到目标batch大小
            current_batch_size = len(current_text_tokens)
            needed_fill = target_batch_size - current_batch_size

            if needed_fill > 0 and scheduler_info['single_requests']:
                # 循环使用单人对话请求填充
                single_idx = 0
                for _ in range(needed_fill):
                    single_req = scheduler_info['single_requests'][single_idx % len(scheduler_info['single_requests'])]
                    request_idx = single_req['request_idx']
                    dialogue_text = single_req['original_request']['dialogue_text']
                    temp_request = [{'dialogue_text': dialogue_text}]
                    text_ids_list, spks_list, _ = self._prepare_batch_requests(temp_request)

                    # 添加到当前轮
                    current_text_tokens.extend(text_ids_list[0])
                    current_spk_ids.extend(spks_list[0])

                    # 记录映射关系
                    for _ in range(len(text_ids_list[0])):
                        current_request_mapping.append({
                            'request_idx': request_idx,
                            'is_multi_speaker': False,
                            'round_idx': round_idx,
                            'is_fill': True  # 标记为填充请求
                        })

                    single_idx += 1

                logger.info(f"Round {round_idx}: Filled batch with {needed_fill} single-speaker requests")

            if not current_text_tokens:
                logger.info(f"Round {round_idx}: No tokens to process, skipping")
                continue

            logger.info(f"Round {round_idx}: Processing {len(current_text_tokens)} tokens (target batch size: {target_batch_size}, actual: {len(current_text_tokens)})")

            # 调用forward_batch，使用简化的返回
            batch_results = self.model.forward_batch(
                prompt_mels_for_llm=preloaded_features["log_mel"][0][None] if isinstance(preloaded_features["log_mel"], list) else preloaded_features["log_mel"],
                prompt_mels_lens_for_llm=preloaded_features["mel_len"][0][None] if isinstance(preloaded_features["mel_len"], list) else preloaded_features["mel_len"],
                prompt_text_tokens_for_llm=preloaded_features["prompt_text_tokens"],
                text_tokens_for_llm=current_text_tokens,  # 当前轮的tokens
                prompt_mels_for_flow_ori=preloaded_features["mel"],
                spk_emb_for_flow=torch.tensor(preloaded_features["spk_emb"]),
                sampling_params=self._create_sampling_params(temperature, top_k, top_p, repetition_penalty),
                spk_ids=current_spk_ids,  # 当前轮的spk_ids，一一对应
                inputs_prompt=inputs_prompt,  # 传递状态
                return_state=True  # 需要返回更新的inputs_prompt
            )

            # 更新inputs_prompt状态
            if isinstance(batch_results, dict) and 'inputs_prompt' in batch_results:
                inputs_prompt = batch_results['inputs_prompt']
                audio_results = batch_results['audio_results']
            else:
                # 向后兼容，如果没有返回状态则直接使用
                audio_results = batch_results

            logger.info(f"Round {round_idx} completed: {len(audio_results)} audio segments generated")

            # 分发音频结果
            for i, mapping in enumerate(current_request_mapping):
                if i < len(audio_results):  # 防止索引越界
                    request_idx = mapping['request_idx']
                    is_multi_speaker = mapping['is_multi_speaker']
                    is_fill = mapping.get('is_fill', False)  # 是否为填充请求
                    audio_result = audio_results[i]

                    if is_multi_speaker:
                        # 多人对话：累积音频段
                        if request_idx not in multi_audio_accumulator:
                            multi_audio_accumulator[request_idx] = []
                        multi_audio_accumulator[request_idx].append(audio_result)
                    else:
                        # 单人对话：只在第一次遇到时记录结果（避免填充重复记录）
                        if request_idx not in single_audio_results:
                            single_audio_results[request_idx] = audio_result
                            logger.info(f"Single-speaker request {request_idx}: completed {'(fill)' if is_fill else ''}")
                        elif is_fill:
                            logger.debug(f"Single-speaker request {request_idx}: skipped duplicate fill")

        logger.info("All rounds completed, assembling final results...")
        return self._assemble_final_results(
            scheduler_info, multi_audio_accumulator, single_audio_results
        )

    def _assemble_final_results(self, scheduler_info, multi_audio_accumulator, single_audio_results):
        """组装最终结果"""
        final_results = [None] * len(scheduler_info['multi_requests'] + scheduler_info['single_requests'])
        sample_rate = 24000

        # 处理多人对话请求
        for multi_request in scheduler_info['multi_requests']:
            request_idx = multi_request['request_idx']
            audio_segments = multi_audio_accumulator[request_idx]

            # 拼接音频段
            target_audio = None
            for segment_audio in audio_segments:
                segment_tensor = torch.from_numpy(segment_audio)
                if target_audio is None:
                    target_audio = segment_tensor
                else:
                    target_audio = torch.concat([target_audio, segment_tensor], axis=0)

            final_audio = target_audio.numpy()
            logger.info(f"Multi-speaker request {request_idx}: concatenated {len(audio_segments)} segments, "
                       f"total audio length: {len(final_audio) / sample_rate:.2f}s")

            final_results[request_idx] = (sample_rate, final_audio)

        # 处理单人对话请求
        for single_request in scheduler_info['single_requests']:
            request_idx = single_request['request_idx']
            final_audio = single_audio_results[request_idx]
            logger.info(f"Single-speaker request {request_idx}: audio length: {len(final_audio) / sample_rate:.2f}s")

            final_results[request_idx] = (sample_rate, final_audio)

        logger.info(f"Dynamic scheduling completed: {len(final_results)} final audio results")
        return final_results

    def _create_sampling_params(self, temperature, top_k, top_p, repetition_penalty):
        """创建采样参数"""

        return SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            extra_args={
                "use_ras": True,
                "win_size": 25,
                "tau_r": 0.2,
            },
        )


    def generate_batch(
        self,
        batch_requests: List[dict],
        mode: Optional[str] = None,
        temperature: float = 0.6,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        批量生成语音

        Args:
            batch_requests: 批量请求列表，每个请求包含一个S1对话文本
            mode: 模式参数(三位数字)，使用预加载数据
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
            repetition_penalty: 重复惩罚

        Returns:
            List[Tuple[int, np.ndarray]]: 批量音频结果列表
        """
        logger.info(f"Batch generate called - Instance ID: {id(self)}, Model loaded: {self.is_loaded()}")
        logger.info(f"Processing {len(batch_requests)} batch requests with mode: {mode}")

        if not self.is_loaded():
            raise RuntimeError("模型未加载")

        if not mode or mode not in self.preloaded_data:
            raise ValueError(f"批量生成模式需要提供有效的mode参数，可选: {list(self.preloaded_data.keys())}")

        start_time = time.time()
        results = []

        try:
            # 设置随机种子
            # seed = 0
            # torch.manual_seed(seed)
            # np.random.seed(seed)
            # torch.cuda.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

            # 使用预加载数据
            logger.info(f"Using preloaded data for mode: {mode} ({self.preloaded_data[mode]['description']})")
            preloaded = self.preloaded_data[mode]
            prompt_audio_paths = preloaded["prompt_audio_paths"]
            # prompt_texts = preloaded["prompt_texts"]

            scheduler_info = self._create_dynamic_batch_scheduler(batch_requests)

            if not scheduler_info['use_scheduler']:
                # 没有多人对话，使用原有逻辑
                logger.info("No multi-speaker dialogues detected, using original batch processing")
                all_text_ids_list, all_spks_lists, request_segment_mapping = self._prepare_batch_requests(batch_requests)

                num_speakers = len(prompt_audio_paths)
                logger.info(f"Processing batch for {num_speakers} speaker(s)")

                # 调用模型的批量前向传播
                batch_results = self._forward_batch(
                    preloaded_features={
                        "log_mel": preloaded["log_mel_list"],
                        "spk_emb": preloaded["spk_emb_list"],
                        "mel": preloaded["mel_list"],
                        "mel_len": preloaded["mel_len_list"],
                        "prompt_text_tokens": preloaded["prompt_text_ids_list"],
                    },
                    batch_text_ids_list=all_text_ids_list,
                    batch_spks_lists=all_spks_lists,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

                # 使用新的拼接函数进行音频拼接
                results = self._concatenate_batch_results(
                    batch_audio_results=batch_results,
                    request_segment_mapping=request_segment_mapping
                )
            else:
                # 使用动态调度器进行多轮处理
                logger.info(f"Using dynamic scheduler for multi-speaker processing: {scheduler_info['total_rounds']} rounds")
                results = self._execute_dynamic_scheduling(
                    scheduler_info=scheduler_info,
                    preloaded_features={
                        "log_mel": preloaded["log_mel_list"],
                        "spk_emb": preloaded["spk_emb_list"],
                        "mel": preloaded["mel_list"],
                        "mel_len": preloaded["mel_len_list"],
                        "prompt_text_tokens": preloaded["prompt_text_ids_list"],
                    },
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )           

            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Batch generation completed. {len(results)} results in {execution_time:.3f} seconds")

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}", exc_info=True)
            raise RuntimeError(f"批量语音生成失败: {str(e)}")

    def _forward_batch(
        self,
        preloaded_features: dict,
        batch_text_ids_list: List[List[List[int]]],
        batch_spks_lists: List[List[int]],
        temperature: float,
        top_k: int,
        top_p: int,
        repetition_penalty: float,
    ) -> List[np.ndarray]:
        """批量前向传播辅助方法"""

        # 合并所有请求的文本tokens和说话人IDs为单个列表 - 这是关键
        all_text_tokens = []
        all_spk_ids = []

        for text_ids_list, spks_list in zip(batch_text_ids_list, batch_spks_lists):
            all_text_tokens.extend(text_ids_list)  # 展开每个请求的文本片段
            all_spk_ids.extend(spks_list)         # 展开每个请求的说话人ID

        # 采样参数
        # from soulxpodcast.config import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            extra_args={
                "use_ras": True,
                "win_size": 25,
                "tau_r": 0.2,
            },
        )
        # 准备模型输入 - 按照原有generate方法的方式处理数据
        # import s3tokenizer
        # import torch

        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(preloaded_features["log_mel"])

        spk_emb_for_flow = torch.tensor(preloaded_features["spk_emb"])
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
            preloaded_features["mel"], batch_first=True, padding_value=0
        )
        # prompt_mels_lens_for_flow = torch.tensor(preloaded_features["mel_len"])
        # 调用模型批量前向传播 - 使用与forward_longform相同的接口
        return self.model.forward_batch(
            prompt_mels_for_llm=prompt_mels_for_llm,
            prompt_mels_lens_for_llm=prompt_mels_lens_for_llm,
            prompt_text_tokens_for_llm=preloaded_features["prompt_text_tokens"],
            text_tokens_for_llm=all_text_tokens,  # 批量文本tokens列表
            prompt_mels_for_flow_ori=prompt_mels_for_flow,
            spk_emb_for_flow=spk_emb_for_flow,
            sampling_params=sampling_params,
            spk_ids=all_spk_ids,  # 批量说话人ID列表
            use_dialect_prompt=False
        )

    def generate(
        self,
        prompt_audio_paths: List[str],
        prompt_texts: List[str],
        dialogue_text: str,
        # seed: int = 1988,
        temperature: float = 0.6,
        top_k: int = 100,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
        mode: Optional[str] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        生成语音

        Args:
            prompt_audio_paths: 参考音频路径列表
            prompt_texts: 参考文本列表
            dialogue_text: 对话文本
            seed: 随机种子
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Top-P采样
            repetition_penalty: 重复惩罚
            mode: 模式参数(三位数字),如 "000"=单人男生普通话, "010"=单人女生普通话, "120"=双人普通话
                  如果提供mode,则使用预加载的数据;否则使用传入的prompt_audio_paths和prompt_texts

        Returns:
            Tuple[int, np.ndarray]: (采样率, 音频数组)
        """
        logger.info(f"Generate called - Instance ID: {id(self)}, Model loaded: {self.is_loaded()}")

        if not self.is_loaded():
            raise RuntimeError("模型未加载")

        # 使用锁确保同一时间只有一个生成任务
        # with self._generation_lock:
        # logger.info("Acquired generation lock")
        start_time = time.time()
        try:
            # 设置随机种子
            # torch.manual_seed(seed)
            # np.random.seed(seed)
            # random.seed(seed)

            seed=0
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            dataset_time_end = time.time()
            if mode and mode in self.preloaded_data:
                logger.info(f"Using preloaded data for mode: {mode} ({self.preloaded_data[mode]['description']})")
                preloaded = self.preloaded_data[mode]
                prompt_audio_paths = preloaded["prompt_audio_paths"]
                prompt_texts = preloaded["prompt_texts"]

                # 直接使用预加载的特征
                use_preloaded_features = True
                preloaded_features = {
                    "log_mel": preloaded["log_mel_list"],
                    "spk_emb": preloaded["spk_emb_list"],
                    "mel": preloaded["mel_list"],
                    "mel_len": preloaded["mel_len_list"],
                    "prompt_text_tokens": preloaded["prompt_text_ids_list"],
                }
            elif mode:
                logger.warning(f"Mode {mode} not found in preloaded data, falling back to dynamic processing")
                use_preloaded_features = False
            else:
                use_preloaded_features = False

            num_speakers = len(prompt_audio_paths)
            logger.info(f"Generating audio for {num_speakers} speaker(s)")

            # 解析对话文本
            target_text_list = parse_dialogue_text(dialogue_text, 0)
            logger.info(f"Parsed dialogue into {len(target_text_list)} segments")

            # 提取说话人和文本
            spks, texts = [], []
            for target_text in target_text_list:
                pattern = r'(\[S[1-9]\])(.+)'
                match = re.match(pattern, target_text)
                if match:
                    text, spk = match.group(2), int(match.group(1)[2]) - 1
                    spks.append(spk)
                    texts.append(text)
                else:
                    raise ValueError(f"无效的对话文本格式: {target_text}")
                
            # 如果使用预加载特征,直接构建data
            if use_preloaded_features:
                # 处理文本tokens
                from soulxpodcast.utils.dataloader import SPK_DICT, TEXT_START, TEXT_END, AUDIO_START
                from soulxpodcast.utils.text import normalize_text

                text_ids_list, spks_list = [], []
                for text, spk in zip(texts, spks):
                    text = normalize_text(text)
                    text = f"{SPK_DICT[spk]}{TEXT_START}{text}{TEXT_END}{AUDIO_START}"
                    text_ids = self.model.llm.tokenizer.encode(text)
                    text_ids_list.append(text_ids)
                    spks_list.append(spk)

                data = {
                    "log_mel": preloaded_features["log_mel"],
                    "spk_emb": preloaded_features["spk_emb"],
                    "mel": preloaded_features["mel"],
                    "mel_len": preloaded_features["mel_len"],
                    "prompt_text_tokens": preloaded_features["prompt_text_tokens"],
                    "text_tokens": text_ids_list,
                    "spks_list": spks_list,
                    "info": {
                        "key": f"api_mode_{mode}",
                        "prompt_text": prompt_texts,
                        "prompt_wav": prompt_audio_paths,
                        "text": texts,
                        "spk": spks,
                    }
                }
            else:
                # 构建数据项
                dataitem = {
                    "key": "api_001",
                    "prompt_text": prompt_texts,
                    "prompt_wav": prompt_audio_paths,
                    "text": texts,
                    "spk": spks,
                }
                # 更新数据源
                self.dataset.update_datasource([dataitem])

                # 获取处理后的数据
                data = self.dataset[0]
        # # 构建数据项
            # dataitem = {
            #     "key": "api_001",
            #     "prompt_text": prompt_texts,
            #     "prompt_wav": prompt_audio_paths,
            #     "text": texts,
            #     "spk": spks,
            # }
            # # 更新数据源
            # self.dataset.update_datasource([dataitem])
            # dataset_time = time.time()
            # # 获取处理后的数据
            # data = self.dataset[0]
            # dataset_time_end = time.time()
            # logger.info(f"self.dataset[0] time: {dataset_time_end-dataset_time:.4f}s")            

            # 准备模型输入
            prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
            s3tokenizer_padding = time.time()
            logger.info(f"s3tokenizer_padding time: {s3tokenizer_padding-dataset_time_end:.4f}s")

            spk_emb_for_flow = torch.tensor(data["spk_emb"])
            prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
                data["mel"], batch_first=True, padding_value=0
            )

            prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
            text_tokens_for_llm = data["text_tokens"]
            prompt_text_tokens_for_llm = data["prompt_text_tokens"]
            spk_ids = data["spks_list"]

            # 采样参数
            sampling_params = SamplingParams(
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                extra_args={
                    "use_ras":True,
                    "win_size":25,
                    "tau_r":0.2,
                },
            )

            infos = [data["info"]]
            processed_data = {
                "prompt_mels_for_llm": prompt_mels_for_llm,
                "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
                "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
                "text_tokens_for_llm": text_tokens_for_llm,
                "prompt_mels_for_flow_ori": prompt_mels_for_flow,
                "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
                "spk_emb_for_flow": spk_emb_for_flow,
                "sampling_params": sampling_params,
                "spk_ids": spk_ids,
                "infos": infos,
                "use_dialect_prompt": False,
            }

            generation_time = time.time()
            logger.info(f"DataLoader time: {generation_time-start_time:.4f}s")
            # 模型推理
            logger.info("Running model inference...")

            # 清理之前可能累积的GPU缓存
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            # 设置超时时间（根据音频长度动态调整）
            num_segments = len(texts)
            timeout_seconds = max(1200, num_segments * 12000)  # 每段至少120秒，最少20分钟(1200秒)

            logger.info(f"Starting inference with timeout: {timeout_seconds}s for {num_segments} segments")

            with torch.no_grad():
                results_dict = self.model.forward_longform(**processed_data)

            # 使用线程池执行推理
            # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            #     future = executor.submit(run_inference)
            #     try:
            #         results_dict = future.result(timeout=timeout_seconds)
            #     except concurrent.futures.TimeoutError:
            #         logger.error(f"Model inference timeout after {timeout_seconds} seconds")
            #         # 尝试取消任务
            #         future.cancel()
            #         # 清理GPU内存
            #         if torch.cuda.is_available():
            #             torch.cuda.empty_cache()
            #         raise TimeoutError(f"模型推理超时（{timeout_seconds}秒）。可能是音频过长或GPU内存不足。")
            #     except Exception as e:
            #         logger.error(f"Model inference failed: {e}")
            #         raise RuntimeError(f"模型推理失败: {str(e)}")
            genera = time.time()
            logger.info(f"genera totcl time: {genera-generation_time:.4f}s")
            # 拼接音频
            batch_sigine = list(results_dict['generated_wavs'][0])
            target_audio = None
            for i in range(len(batch_sigine)):
                if target_audio is None:
                    target_audio = batch_sigine[i]
                else:
                    target_audio = torch.concat(
                        [target_audio, batch_sigine[i]], axis=0
                    )

            # 转换为numpy数组
            # audio_array = target_audio.cpu().squeeze(0).numpy()
            audio_array = target_audio.cpu().numpy()
            sample_rate = 24000
            audio_array_time = time.time()
            logger.info(f"audio_array_time time: {audio_array_time-genera:.4f}s")
            # 清理GPU内存
            del target_audio
            del results_dict
            if 'processed_data' in locals():
                del processed_data

            logger.info(f"Audio generation completed. Duration: {len(audio_array) / sample_rate:.2f}s")

            allocated_time = time.time()
            logger.info(f"GPU sys time: {allocated_time-audio_array_time:.4f}s")

            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Generation execution time: {execution_time:.3f} seconds")

            return sample_rate, audio_array

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"语音生成失败: {str(e)}")
        # finally:
        #     logger.info("Released generation lock")


# 全局服务实例
_service: Optional[SoulXPodcastService] = None


def get_service() -> SoulXPodcastService:
    """获取全局服务实例"""
    global _service
    if _service is None:
        _service = SoulXPodcastService()
    return _service
