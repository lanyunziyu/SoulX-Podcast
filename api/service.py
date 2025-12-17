"""
SoulXPodcast Model Service Layer
"""
import re
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional,Dict,Any
import torch
import pickle
import hashlib

import numpy as np
import random
import concurrent.futures
import gc
import threading
import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import PodcastInferHandler

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
        self._init_cache_system()

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

    def _init_cache_system(self):
        """
        初始化缓存系统

        缓存策略：
        - 使用音频文件的SHA256哈希 + 文本的归一化哈希作为缓存key
        - 缓存内容包括：prompt_text_tokens, spk_emb, mel, mel_len, log_mel
        - 缓存持久化到磁盘，支持服务重启后恢复
        """
        try:
            # 创建缓存目录
            self.cache_dir = Path(__file__).parent.parent / "cache" / "prompt_features"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # 内存缓存（快速访问）
            self.memory_cache: Dict[str, Dict[str, Any]] = {}
            self.cache_lock = threading.Lock()

            logger.info(f"Cache system initialized at: {self.cache_dir}")
            logger.info(f"Existing cache files: {len(list(self.cache_dir.glob('*.pkl')))}")

        except Exception as e:
            logger.error(f"Failed to initialize cache system: {e}")
            self.cache_dir = None
            self.memory_cache = {}

    def _compute_audio_hash(self, audio_path: str) -> str:
        """计算音频文件的SHA256哈希"""
        sha256_hash = hashlib.sha256()
        with open(audio_path, "rb") as f:
            # 分块读取以处理大文件
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _compute_text_hash(self, text: str) -> str:
        """计算文本的SHA256哈希（归一化后）"""
        # 文本归一化：去除空格、转小写
        normalized_text = text.strip().lower().replace(" ", "")
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()

    def _get_cache_key(self, audio_path: str, prompt_text: str, dialect_prompt_text: Optional[str] = None) -> str:
        """
        生成缓存key

        Args:
            audio_path: 音频文件路径
            prompt_text: 提示文本
            dialect_prompt_text: 方言提示文本（可选）

        Returns:
            缓存key字符串
        """
        audio_hash = self._compute_audio_hash(audio_path)
        text_hash = self._compute_text_hash(prompt_text)

        # 如果有dialect_prompt_text，也加入哈希
        if dialect_prompt_text:
            dialect_hash = self._compute_text_hash(dialect_prompt_text)
            cache_key = f"{audio_hash}_{text_hash}_{dialect_hash}"
        else:
            cache_key = f"{audio_hash}_{text_hash}"

        return cache_key

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存加载数据

        Args:
            cache_key: 缓存键

        Returns:
            缓存的特征数据，如果不存在返回None
        """
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {cache_key[:16]}...")
            return self.memory_cache[cache_key]

        # 检查磁盘缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)

                    # 加载到内存缓存
                    with self.cache_lock:
                        self.memory_cache[cache_key] = cached_data

                    logger.info(f"Cache hit (disk): {cache_key[:16]}...")
                    return cached_data
                except Exception as e:
                    logger.warning(f"Failed to load cache from {cache_file}: {e}")

        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """
        保存数据到缓存

        Args:
            cache_key: 缓存键
            data: 要缓存的特征数据
        """
        # 保存到内存缓存
        with self.cache_lock:
            self.memory_cache[cache_key] = data

        # 保存到磁盘缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Cached features saved: {cache_key[:16]}...")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_file}: {e}")


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
        dialect_prompt_texts: Optional[List[str]] = None,
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
            dialect_prompt_texts: 方言提示文本列表（可选）

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


            num_speakers = len(prompt_audio_paths)
            logger.info(f"Generating audio for {num_speakers} speaker(s)")

            # 解析对话文本
            target_text_list = parse_dialogue_text(dialogue_text, num_speakers)
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
                
            use_dialect_prompt = dialect_prompt_texts is not None and len(dialect_prompt_texts) > 0
            cached_features_list = []
            need_processing = []  # 需要处理的索引

            for i, (audio_path, prompt_text) in enumerate(zip(prompt_audio_paths, prompt_texts)):
                dialect_text = dialect_prompt_texts[i] if use_dialect_prompt else None
                cache_key = self._get_cache_key(audio_path, prompt_text, dialect_text)

                cached = self._load_from_cache(cache_key)
                if cached:
                    cached_features_list.append(cached)
                    logger.info(f"Using cached features for speaker {i}")
                else:
                    cached_features_list.append(None)
                    need_processing.append(i)
                    logger.info(f"Cache miss for speaker {i}, will process")

            # 如果有需要处理的，批量处理
            if need_processing:
                logger.info(f"Processing {len(need_processing)} speakers without cache")

                # 构建数据项
                dataitem = {
                    "key": "api_001",
                    "prompt_text": prompt_texts,
                    "prompt_wav": prompt_audio_paths,
                    "text": texts,
                    "spk": spks,
                }

                if use_dialect_prompt:
                    dataitem["dialect_prompt_text"] = dialect_prompt_texts

                # 更新数据源并处理
                self.dataset.update_datasource([dataitem])
                data = self.dataset[0]

                # 提取并缓存新处理的特征
                for i in need_processing:
                    dialect_text = dialect_prompt_texts[i] if use_dialect_prompt else None
                    cache_key = self._get_cache_key(prompt_audio_paths[i], prompt_texts[i], dialect_text)

                    # 保存单个speaker的特征
                    speaker_features = {
                        "prompt_text_tokens": data["prompt_text_tokens"][i],
                        "spk_emb": data["spk_emb"][i],
                        "mel": data["mel"][i],
                        "mel_len": data["mel_len"][i],
                        "log_mel": data["log_mel"][i],
                    }
                    if use_dialect_prompt:
                        speaker_features["dialect_prompt_text_tokens"] = data["dialect_prompt_text_tokens"][i]

                    self._save_to_cache(cache_key, speaker_features)
                    cached_features_list[i] = speaker_features

                # 使用处理好的data
            else:
                # 全部来自缓存，重新组装data
                logger.info("All speakers loaded from cache, reconstructing data")

                from soulxpodcast.utils.dataloader import SPK_DICT, TEXT_START, TEXT_END, AUDIO_START
                from soulxpodcast.utils.text import normalize_text

                # 处理目标文本tokens
                text_ids_list, spks_list = [], []
                for text, spk in zip(texts, spks):
                    text = normalize_text(text)
                    text = f"{SPK_DICT[spk]}{TEXT_START}{text}{TEXT_END}{AUDIO_START}"
                    text_ids = self.model.llm.tokenizer.encode(text)
                    text_ids_list.append(text_ids)
                    spks_list.append(spk)

                # 从缓存重建data
                data = {
                    "log_mel": [f["log_mel"] for f in cached_features_list],
                    "spk_emb": [f["spk_emb"] for f in cached_features_list],
                    "mel": [f["mel"] for f in cached_features_list],
                    "mel_len": [f["mel_len"] for f in cached_features_list],
                    "prompt_text_tokens": [f["prompt_text_tokens"] for f in cached_features_list],
                    "text_tokens": text_ids_list,
                    "spks_list": spks_list,
                    "info": {
                        "key": "api_cached",
                        "prompt_text": prompt_texts,
                        "prompt_wav": prompt_audio_paths,
                        "text": texts,
                        "spk": spks,
                    }
                }

                if use_dialect_prompt:
                    data["dialect_prompt_text_tokens"] = [f["dialect_prompt_text_tokens"] for f in cached_features_list]
                    data["use_dialect_prompt"] = True
                
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
            target_audio = None
            for i in range(len(results_dict['generated_wavs'])):
                if target_audio is None:
                    target_audio = results_dict['generated_wavs'][i]
                else:
                    target_audio = torch.concat(
                        [target_audio, results_dict['generated_wavs'][i]], axis=1
                    )

            # 转换为numpy数组
            audio_array = target_audio.cpu().squeeze(0).numpy()
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
