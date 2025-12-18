"""
简单的批处理测试脚本

直接测试批处理模型，不通过API

使用方法:
    python test_batch_simple.py
"""
import os
import sys
import logging
import time
import numpy as np
import scipy.io.wavfile as wavfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_batch_model():
    """测试批处理模型"""
    logger.info("开始测试批处理模型...")

    # 设置环境变量
    model_path = os.getenv("MODEL_PATH", "pretrained_models/SoulX-Podcast-1.7B")
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        logger.error("请设置环境变量 MODEL_PATH 或确保默认路径存在")
        return False

    try:
        from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
        from soulxpodcast.models.soulxpodcast_batch import SoulXPodcastBatch
        import s3tokenizer
        import torch

        # 创建配置
        logger.info(f"加载模型配置: {model_path}")
        hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
            initial_values={"fp16_flow": True},
            json_file=f"{model_path}/soulxpodcast_config.json"
        )

        config = Config(
            model=model_path,
            enforce_eager=True,
            llm_engine="vllm",
            hf_config=hf_config
        )

        # 初始化批处理模型
        logger.info("初始化批处理模型...")
        batch_model = SoulXPodcastBatch(config)
        logger.info("✓ 批处理模型加载成功")

        # 准备测试数据（单个简单请求）
        logger.info("准备测试数据...")

        audio_file = "example/audios/female_mandarin.wav"
        if not os.path.exists(audio_file):
            logger.error(f"测试音频文件不存在: {audio_file}")
            return False

        # 使用dataloader处理音频
        from soulxpodcast.utils.dataloader import PodcastInferHandler
        dataset = PodcastInferHandler(
            batch_model.llm.tokenizer,
            None,
            config
        )

        dataitem = {
            "key": "test_001",
            "prompt_text": ["喜欢攀岩、徒步、滑雪的语言爱好者。"],
            "prompt_wav": [audio_file],
            "text": ["大家好，欢迎收听今天的节目。"],
            "spk": [0],
        }

        dataset.update_datasource([dataitem])
        data = dataset[0]

        # 准备批处理输入
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
        spk_emb_for_flow = torch.tensor(data["spk_emb"])
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
            data["mel"], batch_first=True, padding_value=0
        )

        batch_data = [{
            "prompt_mels_for_llm": prompt_mels_for_llm,
            "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
            "prompt_text_tokens_for_llm": data["prompt_text_tokens"],
            "text_tokens_for_llm": data["text_tokens"],
            "prompt_mels_for_flow": prompt_mels_for_flow,
            "spk_emb_for_flow": spk_emb_for_flow,
            "spk_ids": data["spks_list"],
        }]

        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.6,
            repetition_penalty=1.25,
            top_k=100,
            top_p=0.9,
            extra_args={
                "use_ras": True,
                "win_size": 25,
                "tau_r": 0.2,
            },
        )

        # 执行批处理
        logger.info("开始批处理推理...")
        start_time = time.time()

        results = batch_model.forward_batch(batch_data, sampling_params)

        elapsed = time.time() - start_time
        logger.info(f"✓ 批处理完成，耗时: {elapsed:.2f}秒")

        # 保存结果
        if results and len(results) > 0:
            result = results[0]
            audio = result["audio"]
            sample_rate = result["sample_rate"]

            output_path = "api/outputs/test_batch_simple.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, sample_rate, audio)

            logger.info(f"✓ 音频已保存到: {output_path}")
            logger.info(f"  采样率: {sample_rate} Hz")
            logger.info(f"  音频长度: {len(audio) / sample_rate:.2f}秒")
            logger.info(f"  生成turns数: {result['num_turns']}")

            return True
        else:
            logger.error("批处理返回空结果")
            return False

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SoulX-Podcast 批处理模型测试")
    logger.info("=" * 60)

    success = test_batch_model()

    logger.info("=" * 60)
    if success:
        logger.info("✓ 测试通过")
        sys.exit(0)
    else:
        logger.error("✗ 测试失败")
        sys.exit(1)
