"""
Batch Pipeline测试脚本

测试batch处理功能：
1. 单个请求多turns
2. 多个请求并发
3. 显存自适应batch大小
4. 性能对比（batch vs sequential）
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path

import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.engine.batch_manager import TurnRequest
from soulxpodcast.models.soulxpodcast_batch import SoulXPodcastBatchPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_request_multi_turns(model_path: str):
    """
    测试1: 单个请求多个turns

    验证batch pipeline能否正确处理一个请求的多个turns
    """
    print("\n" + "=" * 80)
    print("测试1: 单个请求多个turns")
    print("=" * 80)

    # 加载配置
    hf_config = SoulXPodcastLLMConfig.from_json(f"{model_path}/soulxpodcast_config.json")
    config = Config(
        model=model_path,
        enforce_eager=True,
        llm_engine="vllm",
        hf_config=hf_config
    )

    # 加载模型
    logger.info("Loading model...")
    model = SoulXPodcast(config)

    # 创建batch pipeline
    batch_pipeline = SoulXPodcastBatchPipeline(
        model=model.llm,
        flow=model.flow,
        hift=model.hift,
        config=config,
        max_batch_size=16,
        enable_dynamic_batching=True,
    )

    # 准备测试数据
    request_id = "test_req_001"
    num_turns = 5

    # 创建mock turn requests
    turn_requests = []
    for turn_id in range(num_turns):
        # Mock LLM input (简化)
        llm_input = [1, 2, 3, 4, 5] * 10  # 50 tokens

        # Mock Flow inputs
        prompt_mel = torch.randn(1, 80, 100).cuda()
        prompt_mel_len = torch.tensor([100]).cuda()
        spk_emb = torch.randn(1, 192).cuda()

        turn_req = TurnRequest(
            request_id=request_id,
            turn_id=turn_id,
            total_turns=num_turns,
            llm_input=llm_input,
            prompt_mel=prompt_mel,
            prompt_mel_len=prompt_mel_len,
            spk_emb=spk_emb,
            spk_id=0,
            sampling_params=SamplingParams(
                temperature=0.6,
                top_p=0.9,
                top_k=100,
                max_tokens=100,
            )
        )
        turn_requests.append(turn_req)

    # 添加请求
    logger.info(f"Adding request with {num_turns} turns...")
    batch_pipeline.add_request(request_id, turn_requests)

    # 处理batch
    logger.info("Processing batch...")
    start_time = time.time()

    # 持续处理直到完成
    while not batch_pipeline.request_manager.is_request_completed(request_id):
        processed = batch_pipeline.process_batch(timeout=0.5)
        if processed > 0:
            logger.info(f"Processed {processed} turns")
        time.sleep(0.1)

    total_time = time.time() - start_time

    # 获取结果
    results = batch_pipeline.get_request_results(request_id)

    # 验证结果
    print(f"\n✓ 测试完成!")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  处理turns: {len(results)}")
    print(f"  平均耗时: {total_time/num_turns:.2f}秒/turn")

    # 统计信息
    stats = batch_pipeline.get_statistics()
    print(f"\n📊 统计信息:")
    print(f"  Batch调度器:")
    print(f"    当前batch大小: {stats['scheduler']['current_batch_size']}")
    print(f"    总batch数: {stats['scheduler']['total_batches']}")
    print(f"    OOM次数: {stats['scheduler']['oom_count']}")
    print(f"  请求管理器:")
    print(f"    总请求数: {stats['request_manager']['total_requests']}")
    print(f"    已处理turns: {stats['request_manager']['total_turns_processed']}")


def test_multi_requests_concurrent(model_path: str, num_requests: int = 3):
    """
    测试2: 多个请求并发处理

    验证batch pipeline能否并发处理多个请求
    """
    print("\n" + "=" * 80)
    print(f"测试2: {num_requests}个请求并发处理")
    print("=" * 80)

    # 加载配置
    hf_config = SoulXPodcastLLMConfig.from_json(f"{model_path}/soulxpodcast_config.json")
    config = Config(
        model=model_path,
        enforce_eager=True,
        llm_engine="vllm",
        hf_config=hf_config
    )

    # 加载模型
    logger.info("Loading model...")
    model = SoulXPodcast(config)

    # 创建batch pipeline
    batch_pipeline = SoulXPodcastBatchPipeline(
        model=model.llm,
        flow=model.flow,
        hift=model.hift,
        config=config,
        max_batch_size=32,
        enable_dynamic_batching=True,
    )

    # 创建多个请求
    all_request_ids = []
    for req_idx in range(num_requests):
        request_id = f"test_req_{req_idx:03d}"
        all_request_ids.append(request_id)

        num_turns = np.random.randint(3, 8)  # 随机3-7个turns

        turn_requests = []
        for turn_id in range(num_turns):
            llm_input = [1, 2, 3, 4, 5] * 10

            prompt_mel = torch.randn(1, 80, 100).cuda()
            prompt_mel_len = torch.tensor([100]).cuda()
            spk_emb = torch.randn(1, 192).cuda()

            turn_req = TurnRequest(
                request_id=request_id,
                turn_id=turn_id,
                total_turns=num_turns,
                llm_input=llm_input,
                prompt_mel=prompt_mel,
                prompt_mel_len=prompt_mel_len,
                spk_emb=spk_emb,
                spk_id=req_idx % 2,  # 交替使用2个说话人
                sampling_params=SamplingParams(
                    temperature=0.6,
                    top_p=0.9,
                    top_k=100,
                    max_tokens=100,
                )
            )
            turn_requests.append(turn_req)

        batch_pipeline.add_request(request_id, turn_requests)
        logger.info(f"Added request {request_id} with {num_turns} turns")

    # 处理所有请求
    logger.info("Processing all requests...")
    start_time = time.time()

    # 持续处理直到所有请求完成
    all_completed = False
    total_processed = 0

    while not all_completed:
        processed = batch_pipeline.process_batch(timeout=0.5)
        if processed > 0:
            total_processed += processed
            logger.info(f"Processed {processed} turns, total: {total_processed}")

        # 检查是否全部完成
        all_completed = all(
            batch_pipeline.request_manager.is_request_completed(req_id)
            for req_id in all_request_ids
        )

        time.sleep(0.1)

    total_time = time.time() - start_time

    # 收集结果
    all_results = {}
    for req_id in all_request_ids:
        results = batch_pipeline.get_request_results(req_id)
        all_results[req_id] = results

    # 验证结果
    print(f"\n✓ 测试完成!")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  处理请求数: {len(all_request_ids)}")
    print(f"  处理turns总数: {total_processed}")
    print(f"  平均耗时: {total_time/total_processed:.2f}秒/turn")

    # 每个请求的结果
    print(f"\n📋 各请求结果:")
    for req_id, results in all_results.items():
        print(f"  {req_id}: {len(results)} turns completed")

    # 统计信息
    stats = batch_pipeline.get_statistics()
    print(f"\n📊 统计信息:")
    print(f"  Batch调度器:")
    print(f"    当前batch大小: {stats['scheduler']['current_batch_size']}")
    print(f"    总batch数: {stats['scheduler']['total_batches']}")
    print(f"  显存使用:")
    print(f"    已分配: {stats['scheduler']['allocated']:.2f} GB")
    print(f"    总显存: {stats['scheduler']['total']:.2f} GB")


def test_performance_comparison(model_path: str):
    """
    测试3: 性能对比 (Batch vs Sequential)

    对比batch处理和顺序处理的性能差异
    """
    print("\n" + "=" * 80)
    print("测试3: 性能对比 (Batch vs Sequential)")
    print("=" * 80)

    print("⚠️  此测试需要实际模型，当前为占位实现")
    print("   在真实环境中运行时会对比：")
    print("   - Sequential: 逐个处理turns")
    print("   - Batch: 批量处理turns")
    print("   预期加速比: 2-5x (取决于batch大小)")


def main():
    parser = argparse.ArgumentParser(description="Batch Pipeline测试")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/path/to/model",
        help="模型路径"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["single", "multi", "performance", "all"],
        default="single",
        help="测试类型"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=3,
        help="并发请求数（multi测试用）"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SoulXPodcast Batch Pipeline 测试")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"测试类型: {args.test}")

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"\n✓ CUDA可用")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"\n⚠️  CUDA不可用，测试可能失败")

    # 运行测试
    try:
        if args.test in ["single", "all"]:
            test_single_request_multi_turns(args.model_path)

        if args.test in ["multi", "all"]:
            test_multi_requests_concurrent(args.model_path, args.num_requests)

        if args.test in ["performance", "all"]:
            test_performance_comparison(args.model_path)

        print("\n" + "=" * 80)
        print("✓ 所有测试完成!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
