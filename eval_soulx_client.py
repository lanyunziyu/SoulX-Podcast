#!/usr/bin/env python3
"""
SoulX-Podcast 使用客户端API进行 Seed-TTS-Eval 数据集评估

使用方法:
    # 确保API服务已启动: python run_api.py

    # 评估中文数据集
    python eval_soulx_client.py --meta_file data/seedtts_testset/zh/meta.lst --output_dir results/zh --lang zh --api_url http://localhost:8000

    # 评估英文数据集
    python eval_soulx_client.py --meta_file data/seedtts_testset/en/meta.lst --output_dir results/en --lang en --api_url http://localhost:8000

    # 限制样本数量
    python eval_soulx_client.py --meta_file data/seedtts_testset/zh/meta.lst --output_dir results/zh --lang zh --num_samples 100 --api_url http://localhost:8000
"""

import os
import sys
import tempfile
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

import requests
import soundfile as sf
import numpy as np
import scipy.signal

# 评估相关 imports
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import zhconv
try:
    from funasr import AutoModel
except ImportError:
    print("Warning: FunASR not installed, Chinese CER evaluation will not work")

# Speaker verification imports
script_dir = Path(__file__).parent
verification_path = script_dir.parent / 'seed-tts-eval' / 'thirdparty' / 'UniSpeech' / 'downstreams' / 'speaker_verification'
sys.path.append(str(verification_path))
try:
    from verification import init_model, verification
except ImportError as e:
    print(f"Warning: Speaker verification module not found, SIM evaluation will not work. Error: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoulXSeedTTSClientEvaluator:
    """SoulX-Podcast 使用客户端API在 Seed-TTS-Eval 数据集上的评估器"""

    def __init__(self, api_url: str):
        """
        初始化评估器

        Args:
            api_url: API服务地址，例如 http://localhost:8000
        """
        self.api_url = api_url.rstrip('/')

        # 检查API健康状态
        logger.info(f"Connecting to API at {self.api_url}")
        self._check_api_health()

        # ASR模型（延迟加载）
        self.whisper_processor = None
        self.whisper_model = None
        self.paraformer_model = None

        # Speaker verification模型（延迟加载）
        self.speaker_model = None
        self.speaker_model_name = "wavlm_large"
        self.speaker_checkpoint = None

    def _check_api_health(self):
        """检查API健康状态"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            response.raise_for_status()
            health = response.json()

            logger.info(f"API Status: {health['status']}")
            logger.info(f"Model Loaded: {health['model_loaded']}")
            logger.info(f"GPU Available: {health['gpu_available']}")

            if not health['model_loaded']:
                raise RuntimeError("API model is not loaded")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to API: {e}")
            raise RuntimeError(f"Cannot connect to API at {self.api_url}. Please ensure the API server is running.")

    def load_asr_models(self, lang: str):
        """加载ASR模型"""
        if lang == "en" and self.whisper_processor is None:
            logger.info("Loading Whisper model for English ASR...")
            model_id = "openai/whisper-large-v3"
            self.whisper_processor = WhisperProcessor.from_pretrained(model_id)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id).cuda()

        elif lang == "zh" and self.paraformer_model is None:
            logger.info("Loading Paraformer model for Chinese ASR...")
            self.paraformer_model = AutoModel(model="paraformer-zh")

    def load_speaker_model(self):
        """加载说话人验证模型"""
        if self.speaker_model is None:
            logger.info("Loading WavLM model for speaker similarity calculation...")
            self.speaker_model = init_model(self.speaker_model_name, self.speaker_checkpoint)

    def parse_meta_file(self, meta_file: str) -> List[Dict]:
        """解析元数据文件"""
        meta_data = []
        base_dir = Path(meta_file).parent

        with open(meta_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')

            if len(parts) != 4:
                logger.warning(f"Invalid format in line {line_idx+1} (expected 4 fields, got {len(parts)}): {line}")
                continue

            try:
                utt_id, prompt_text, prompt_wav, target_text = parts

                # 处理相对路径
                if not os.path.isabs(prompt_wav):
                    prompt_wav = str(base_dir / prompt_wav)

                # 检查音频文件是否存在
                if not os.path.exists(prompt_wav):
                    logger.warning(f"Prompt audio not found: {prompt_wav}")
                    continue

                meta_data.append({
                    'utt_id': utt_id.strip(),
                    'prompt_text': prompt_text.strip(),
                    'prompt_wav': prompt_wav.strip(),
                    'target_text': target_text.strip(),
                    'line_idx': line_idx + 1
                })

            except Exception as e:
                logger.warning(f"Error parsing line {line_idx+1}: {line} - {e}")
                continue

        logger.info(f"Loaded {len(meta_data)} samples from {meta_file}")
        return meta_data

    def synthesize_audio(self, prompt_wav: str, prompt_text: str, target_text: str) -> Optional[np.ndarray]:
        """使用API生成音频"""
        try:
            # 构造对话文本（单说话人）
            dialogue_text = f"[S1]{target_text}"

            # 准备请求数据
            with open(prompt_wav, 'rb') as audio_file:
                files = {
                    'prompt_audio': audio_file
                }
                data = {
                    'prompt_texts': json.dumps([prompt_text]),
                    'dialogue_text': dialogue_text,
                    'seed': 1988
                }

                # 发送请求到API
                response = requests.post(
                    f"{self.api_url}/generate",
                    files=files,
                    data=data,
                    timeout=300  # 5分钟超时
                )
                response.raise_for_status()

            # 保存临时音频文件并读取
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name
                temp_file.write(response.content)

            try:
                # 读取音频数据
                audio_array, sr = sf.read(temp_wav_path)
                return audio_array
            finally:
                # 清理临时文件
                os.unlink(temp_wav_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to synthesize audio: {e}", exc_info=True)
            return None

    def calculate_cer_wer(self, synthesized_audio: np.ndarray, reference_text: str, lang: str) -> Tuple[float, str]:
        """
        计算CER (中文) 或 WER (英文)

        Returns:
            Tuple[float, str]: (错误率, 识别文本)
        """
        try:
            # 将音频保存到临时文件进行ASR
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name
                sf.write(temp_wav_path, synthesized_audio, 24000)

            try:
                if lang == "en":
                    # 英文ASR
                    wav, sr = sf.read(temp_wav_path)
                    if sr != 16000:
                        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))

                    input_features = self.whisper_processor(
                        wav, sampling_rate=16000, return_tensors="pt"
                    ).input_features.cuda()

                    forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                        language="english", task="transcribe"
                    )
                    predicted_ids = self.whisper_model.generate(
                        input_features, forced_decoder_ids=forced_decoder_ids
                    )
                    transcription = self.whisper_processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]

                elif lang == "zh":
                    # 中文ASR
                    res = self.paraformer_model.generate(input=temp_wav_path, batch_size_s=300)
                    transcription = res[0]["text"]
                    # 转换为简体中文
                    transcription = zhconv.convert(transcription, 'zh-cn')

                # 计算CER或WER
                error_rate = self._compute_cer_wer(transcription, reference_text, lang)
                return error_rate, transcription

            finally:
                # 清理临时文件
                os.unlink(temp_wav_path)

        except Exception as e:
            logger.error(f"Failed to calculate CER/WER: {e}", exc_info=True)
            return 1.0, ""  # 返回最大错误率

    def _compute_cer_wer(self, hypothesis: str, reference: str, lang: str) -> float:
        """
        计算CER/WER核心逻辑
        使用与 seed-tts-eval 完全相同的方法
        """
        # 移除标点符号
        punctuation_all = punctuation + string.punctuation

        for x in punctuation_all:
            if x == '\'':
                continue
            reference = reference.replace(x, '')
            hypothesis = hypothesis.replace(x, '')

        reference = reference.replace('  ', ' ').strip()
        hypothesis = hypothesis.replace('  ', ' ').strip()

        if lang == "zh":
            # 中文：将每个字符用空格分隔，然后使用 compute_measures
            # 这是 seed-tts-eval 官方的方法
            reference = " ".join([x for x in reference])
            hypothesis = " ".join([x for x in hypothesis])
        elif lang == "en":
            # 英文：转小写
            reference = reference.lower()
            hypothesis = hypothesis.lower()

        # 使用 jiwer 的 compute_measures 计算
        # 这与 seed-tts-eval 的 run_wer.py 完全一致
        measures = compute_measures(reference, hypothesis)
        error_rate = measures["wer"]  # 对于中文，这实际上是 CER（因为字符被空格分隔了）

        return error_rate

    def calculate_sim(self, synthesized_audio: np.ndarray, prompt_wav: str) -> float:
        """计算说话人相似度 (SIM)"""
        try:
            # 将合成音频保存到临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name
                sf.write(temp_wav_path, synthesized_audio, 24000)

            try:
                # 加载说话人验证模型
                self.load_speaker_model()

                # 使用 verification 函数直接计算相似度
                # 这是 seed-tts-eval 官方使用的方法
                sim, self.speaker_model = verification(
                    self.speaker_model_name,
                    prompt_wav,  # 参考音频
                    temp_wav_path,  # 生成音频
                    use_gpu=True,
                    checkpoint=self.speaker_checkpoint,
                    model=self.speaker_model,
                    device="cuda:0"
                )

                return sim.item()

            finally:
                # 清理临时文件
                os.unlink(temp_wav_path)

        except Exception as e:
            logger.error(f"Failed to calculate speaker similarity: {e}")
            return 0.0  # 返回最小相似度

    def evaluate_dataset(
        self,
        meta_file: str,
        output_dir: str = "results",
        lang: str = "zh",
        num_samples: Optional[int] = None,
        save_audio: bool = False
    ) -> Dict:
        """评估整个数据集"""

        logger.info(f"Starting evaluation with language: {lang}")

        # 加载ASR和说话人验证模型
        self.load_asr_models(lang)

        # 尝试加载说话人验证模型，如果失败则跳过
        try:
            self.load_speaker_model()
            logger.info("Speaker verification model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load speaker verification model: {e}")
            logger.warning("Will skip SIM (Speaker Similarity) calculation")
            self.speaker_model = None

        # 加载数据
        meta_data = self.parse_meta_file(meta_file)

        # 限制样本数量
        if num_samples is not None and num_samples < len(meta_data):
            logger.info(f"Limiting evaluation to {num_samples} samples out of {len(meta_data)} total samples")
            import random
            random.seed(42)
            meta_data = random.sample(meta_data, num_samples)
            logger.info(f"Selected {len(meta_data)} samples for evaluation")
        else:
            logger.info(f"Evaluating all {len(meta_data)} samples")

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_audio:
            audio_output_dir = output_dir / "generated_audio"
            audio_output_dir.mkdir(parents=True, exist_ok=True)

        # 结果存储
        results = []
        successful_syntheses = 0
        cer_wers = []
        sims = []

        for item in tqdm(meta_data, desc="Evaluating samples"):
            utt_id = item['utt_id']

            # 合成音频
            logger.info(f"Synthesizing {utt_id}...")
            synthesized_audio = self.synthesize_audio(
                item['prompt_wav'],
                item['prompt_text'],
                item['target_text']
            )

            if synthesized_audio is None:
                logger.warning(f"Failed to synthesize {utt_id}")
                continue

            successful_syntheses += 1

            # 保存音频（如果需要）
            if save_audio:
                audio_path = audio_output_dir / f"{utt_id}.wav"
                sf.write(audio_path, synthesized_audio, 24000)

            # 计算CER/WER
            cer_wer, transcription = self.calculate_cer_wer(synthesized_audio, item['target_text'], lang)
            cer_wers.append(cer_wer)

            # 计算SIM（说话人相似度）
            sim = None
            if self.speaker_model is not None:
                sim = self.calculate_sim(synthesized_audio, item['prompt_wav'])
                sims.append(sim)

            result = {
                'utt_id': utt_id,
                'prompt_text': item['prompt_text'],
                'target_text': item['target_text'],
                'transcription': transcription,
                'cer' if lang == 'zh' else 'wer': cer_wer,
            }
            if sim is not None:
                result['sim'] = sim
            results.append(result)

            metric_name = "CER" if lang == "zh" else "WER"
            if sim is not None:
                logger.info(f"{utt_id}: {metric_name} = {cer_wer:.4f}, SIM = {sim:.4f}")
            else:
                logger.info(f"{utt_id}: {metric_name} = {cer_wer:.4f}")

        # 计算统计结果
        if results:
            avg_cer_wer = np.mean(cer_wers)
            std_cer_wer = np.std(cer_wers)
            avg_sim = np.mean(sims) if sims else None
            std_sim = np.std(sims) if sims else None

            metric_name = "CER" if lang == "zh" else "WER"
            logger.info(f"\n=== Evaluation Results ===")
            logger.info(f"Language: {lang}")
            logger.info(f"Total samples: {len(meta_data)}")
            logger.info(f"Successful syntheses: {successful_syntheses}")
            logger.info(f"Success rate: {successful_syntheses/len(meta_data)*100:.2f}%")
            logger.info(f"Average {metric_name}: {avg_cer_wer:.4f} ± {std_cer_wer:.4f}")
            if avg_sim is not None:
                logger.info(f"Average SIM: {avg_sim:.4f} ± {std_sim:.4f}")
            logger.info(f"Min {metric_name}: {min(cer_wers):.4f}")
            logger.info(f"Max {metric_name}: {max(cer_wers):.4f}")
            if sims:
                logger.info(f"Min SIM: {min(sims):.4f}")
                logger.info(f"Max SIM: {max(sims):.4f}")

        # 保存详细结果
        results_file = output_dir / "evaluation_results.json"
        result_data = {
            'meta_file': str(meta_file),
            'language': lang,
            'num_samples_requested': num_samples,
            'total_samples_evaluated': len(meta_data),
            'successful_syntheses': successful_syntheses,
            'success_rate': successful_syntheses/len(meta_data)*100 if meta_data else 0,
            f'average_{"cer" if lang == "zh" else "wer"}': avg_cer_wer if results else None,
            f'std_{"cer" if lang == "zh" else "wer"}': std_cer_wer if results else None,
            'results': results
        }
        if avg_sim is not None:
            result_data['average_sim'] = avg_sim
            result_data['std_sim'] = std_sim

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {results_file}")

        return {
            f'avg_{"cer" if lang == "zh" else "wer"}': avg_cer_wer if results else None,
            'avg_sim': avg_sim if results else None,
            'success_rate': successful_syntheses/len(meta_data)*100 if meta_data else 0,
            'total_samples': len(meta_data),
            'successful_syntheses': successful_syntheses
        }


def main():
    parser = argparse.ArgumentParser(description='SoulX-Podcast Seed-TTS-Eval 客户端评估脚本')

    parser.add_argument('--meta_file',default="data/seedtts_testset/zh/meta.lst" ,required=True, help='元数据文件路径 (meta.lst)')
    parser.add_argument('--output_dir',default="results/zh", required=True, help='输出目录')
    parser.add_argument('--api_url', default='http://localhost:8000',
                       help='API服务地址 (默认: http://localhost:8000)')
    parser.add_argument('--lang', default="zh",choices=['zh', 'en'], required=True,
                       help='语言 (zh: 中文, en: 英文)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='测试样本数量（默认: None表示测试全部样本）')
    parser.add_argument('--save_audio', action='store_true',
                       help='是否保存生成的音频文件')

    args = parser.parse_args()

    # 检查meta文件
    if not os.path.exists(args.meta_file):
        logger.error(f"Meta file not found: {args.meta_file}")
        sys.exit(1)

    # 创建评估器
    try:
        evaluator = SoulXSeedTTSClientEvaluator(args.api_url)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # 开始评估
    start_time = time.time()
    results = evaluator.evaluate_dataset(
        meta_file=args.meta_file,
        output_dir=args.output_dir,
        lang=args.lang,
        num_samples=args.num_samples,
        save_audio=args.save_audio
    )
    elapsed_time = time.time() - start_time

    logger.info(f"\n=== Evaluation Completed ===")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")

    # 根据语言显示对应的指标
    metric_name = "CER" if args.lang == "zh" else "WER"
    metric_key = f'avg_{"cer" if args.lang == "zh" else "wer"}'

    if results[metric_key] is not None:
        logger.info(f"Average {metric_name}: {results[metric_key]:.4f}")
    if results['avg_sim'] is not None:
        logger.info(f"Average SIM: {results['avg_sim']:.4f}")
    logger.info(f"Success rate: {results['success_rate']:.2f}%")


if __name__ == "__main__":
    main()
