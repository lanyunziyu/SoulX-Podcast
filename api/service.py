"""
SoulXPodcast Model Service Layer
"""
import re
import logging
import time
from typing import List, Tuple, Optional
import torch
import numpy as np
import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import PodcastInferHandler
from soulxpodcast.utils.dataloader import SPK_DICT, TEXT_START, TEXT_END, AUDIO_START


from api.config import config as api_config
from api.utils import parse_dialogue_text

logger = logging.getLogger(__name__)


class SoulXPodcastService:
    """SoulXPodcast模型服务单例"""


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
                    else:
                        logger.warning(f"Failed to process data for mode {mode}")

            logger.info(f"Data preloading completed. {len(self.preloaded_data)} modes loaded.")

        except Exception as e:
            logger.error(f"Failed to preload data: {e}", exc_info=True)
            # 不抛出异常,允许服务继续运行
            self.preloaded_data = {}

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

            # 设置超时时间（根据音频长度动态调整）
            num_segments = len(texts)
            timeout_seconds = max(1200, num_segments * 12000)  # 每段至少120秒，最少20分钟(1200秒)

            logger.info(f"Starting inference with timeout: {timeout_seconds}s for {num_segments} segments")

            with torch.no_grad():
                results_dict = self.model.forward_longform(**processed_data)

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



# 全局服务实例
_service: Optional[SoulXPodcastService] = None


def get_service() -> SoulXPodcastService:
    """获取全局服务实例"""
    global _service
    if _service is None:
        _service = SoulXPodcastService()
    return _service
