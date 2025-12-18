"""
批量请求客户端

使用示例:
    # 批量生成10个请求
    python client_batch.py --batch-size 10

    # 指定服务器地址
    python client_batch.py --url http://localhost:8000 --batch-size 5
"""
import requests
import time
import json
import argparse
import base64
from pathlib import Path
from typing import List, Dict


def encode_audio_to_base64(audio_path: str) -> str:
    """将音频文件编码为base64字符串"""
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')


def create_batch_requests(batch_size: int, audio_file: str, prompt_text: str, dialogue_text: str) -> List[Dict]:
    """
    创建批量请求数据

    Args:
        batch_size: 批量大小
        audio_file: 参考音频文件路径
        prompt_text: 参考文本
        dialogue_text: 对话文本

    Returns:
        批量请求列表
    """
    # 编码音频文件
    print(f"编码音频文件: {audio_file}")
    audio_base64 = encode_audio_to_base64(audio_file)

    # 创建批量请求
    batch_requests = []
    for i in range(batch_size):
        batch_requests.append({
            "request_id": f"req_{i:03d}",
            "prompt_audio": [audio_base64],  # 单说话人
            "prompt_texts": [prompt_text],
            "dialogue_text": dialogue_text,
        })

    return batch_requests


def test_batch_generation(
    api_url: str,
    batch_size: int = 10,
    audio_file: str = "example/audios/female_mandarin.wav",
    prompt_text: str = "喜欢攀岩、徒步、滑雪的语言爱好者。",
    dialogue_text: str = "[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。",
):
    """
    测试批量生成

    Args:
        api_url: API服务地址
        batch_size: 批量大小
        audio_file: 参考音频文件
        prompt_text: 参考文本
        dialogue_text: 对话文本
    """
    print("=" * 60)
    print(f"批量生成测试")
    print("=" * 60)
    print(f"批量大小: {batch_size}")
    print(f"音频文件: {audio_file}")
    print(f"对话文本: {dialogue_text[:50]}...")
    print("=" * 60)

    # 检查音频文件
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        return

    # 创建批量请求
    print(f"\n准备批量请求数据...")
    start_prepare = time.time()
    batch_requests = create_batch_requests(batch_size, audio_file, prompt_text, dialogue_text)
    prepare_time = time.time() - start_prepare
    print(f"✓ 批量数据准备完成，耗时: {prepare_time:.2f}秒")

    # 发送批量请求
    print(f"\n发送批量请求到: {api_url}/generate-batch")
    start_request = time.time()

    try:
        # 构造表单数据
        form_data = {
            'batch_data': json.dumps(batch_requests),
            'temperature': '0.6',
            'top_k': '100',
            'top_p': '0.9',
            'repetition_penalty': '1.25',
        }

        # 发送POST请求
        response = requests.post(
            f"{api_url}/generate-batch",
            data=form_data,
            timeout=600  # 10分钟超时
        )
        response.raise_for_status()

        request_time = time.time() - start_request
        result = response.json()

        print(f"\n✓ 批量生成成功!")
        print(f"  总耗时: {request_time:.2f}秒")
        print(f"  平均耗时: {request_time/batch_size:.2f}秒/请求")
        print(f"  批次ID: {result['task_id']}")
        print(f"  成功数量: {result['batch_size']}")

        # 显示结果详情
        print(f"\n结果详情:")
        for idx, res in enumerate(result['results']):
            print(f"  请求 {res['request_id']}:")
            print(f"    结果URL: {api_url}{res['result_url']}")
            print(f"    音频长度: {res['audio_length_seconds']:.2f}秒")

        # 下载第一个结果作为示例
        if result['results']:
            first_result = result['results'][0]
            download_url = f"{api_url}{first_result['result_url']}"
            print(f"\n下载第一个结果: {download_url}")

            download_response = requests.get(download_url)
            download_response.raise_for_status()

            output_path = f"api/outputs/batch_test_result_0.wav"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(download_response.content)

            print(f"✓ 已保存到: {output_path}")

        # 性能统计
        print(f"\n性能统计:")
        print(f"  数据准备时间: {prepare_time:.2f}秒")
        print(f"  网络+生成时间: {request_time:.2f}秒")
        print(f"  总时间: {prepare_time + request_time:.2f}秒")
        print(f"  吞吐量: {batch_size / request_time:.2f} 请求/秒")

    except requests.exceptions.Timeout:
        print(f"\n✗ 请求超时 (超过10分钟)")
    except requests.exceptions.RequestException as e:
        print(f"\n✗ 请求失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误详情: {e.response.text}")


def test_batch_multi_speaker(
    api_url: str,
    batch_size: int = 5,
):
    """
    测试批量生成（多说话人）

    Args:
        api_url: API服务地址
        batch_size: 批量大小
    """
    print("=" * 60)
    print(f"批量生成测试 - 多说话人")
    print("=" * 60)

    audio_files = [
        "example/audios/female_mandarin.wav",
        "example/audios/male_mandarin.wav",
    ]

    # 检查音频文件
    for audio_file in audio_files:
        if not Path(audio_file).exists():
            print(f"错误: 找不到音频文件 {audio_file}")
            return

    print(f"批量大小: {batch_size}")
    print(f"说话人数: {len(audio_files)}")
    print("=" * 60)

    # 编码音频
    print(f"\n准备批量请求数据...")
    start_prepare = time.time()

    audio_base64_list = []
    for audio_file in audio_files:
        audio_base64_list.append(encode_audio_to_base64(audio_file))

    # 创建批量请求
    batch_requests = []
    for i in range(batch_size):
        batch_requests.append({
            "request_id": f"multi_req_{i:03d}",
            "prompt_audio": audio_base64_list,
            "prompt_texts": [
                "喜欢攀岩、徒步、滑雪的语言爱好者。",
                "资深科技播客主持人。"
            ],
            "dialogue_text": "[S1]大家好，欢迎收听今天的节目。[S2]是的，今天我们要聊聊人工智能。[S1]这个话题确实很有趣。",
        })

    prepare_time = time.time() - start_prepare
    print(f"✓ 批量数据准备完成，耗时: {prepare_time:.2f}秒")

    # 发送批量请求
    print(f"\n发送批量请求到: {api_url}/generate-batch")
    start_request = time.time()

    try:
        form_data = {
            'batch_data': json.dumps(batch_requests),
            'temperature': '0.6',
        }

        response = requests.post(
            f"{api_url}/generate-batch",
            data=form_data,
            timeout=600
        )
        response.raise_for_status()

        request_time = time.time() - start_request
        result = response.json()

        print(f"\n✓ 批量生成成功!")
        print(f"  总耗时: {request_time:.2f}秒")
        print(f"  平均耗时: {request_time/batch_size:.2f}秒/请求")
        print(f"  吞吐量: {batch_size / request_time:.2f} 请求/秒")

    except Exception as e:
        print(f"\n✗ 请求失败: {e}")


def compare_batch_vs_sequential(api_url: str, test_size: int = 5):
    """
    对比批量处理和顺序处理的性能

    Args:
        api_url: API服务地址
        test_size: 测试请求数量
    """
    print("=" * 60)
    print(f"性能对比测试: 批量 vs 顺序")
    print("=" * 60)
    print(f"测试数量: {test_size} 个请求")
    print("=" * 60)

    audio_file = "example/audios/female_mandarin.wav"
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        return

    prompt_text = "喜欢攀岩、徒步、滑雪的语言爱好者。"
    dialogue_text = "[S1]大家好，欢迎收听今天的节目。"

    # 测试1: 批量处理
    print(f"\n[测试1] 批量处理 ({test_size}个请求)")
    start_batch = time.time()
    test_batch_generation(api_url, test_size, audio_file, prompt_text, dialogue_text)
    batch_time = time.time() - start_batch

    # 测试2: 顺序处理
    print(f"\n[测试2] 顺序处理 ({test_size}个请求)")
    start_seq = time.time()

    audio_base64 = encode_audio_to_base64(audio_file)
    for i in range(test_size):
        try:
            # 发送单个请求到 /generate-batch（batch_size=1）
            batch_requests = [{
                "request_id": f"seq_{i}",
                "prompt_audio": [audio_base64],
                "prompt_texts": [prompt_text],
                "dialogue_text": dialogue_text,
            }]

            response = requests.post(
                f"{api_url}/generate-batch",
                data={'batch_data': json.dumps(batch_requests)},
                timeout=300
            )
            response.raise_for_status()
            print(f"  请求 {i+1}/{test_size} 完成")

        except Exception as e:
            print(f"  请求 {i+1} 失败: {e}")

    seq_time = time.time() - start_seq

    # 对比结果
    print(f"\n" + "=" * 60)
    print(f"性能对比结果")
    print("=" * 60)
    print(f"批量处理总耗时: {batch_time:.2f}秒")
    print(f"顺序处理总耗时: {seq_time:.2f}秒")
    print(f"加速比: {seq_time / batch_time:.2f}x")
    print(f"批量吞吐量: {test_size / batch_time:.2f} 请求/秒")
    print(f"顺序吞吐量: {test_size / seq_time:.2f} 请求/秒")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="批量请求客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API服务地址（默认: http://localhost:8000）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "multi", "compare"],
        default="batch",
        help="测试模式（默认: batch）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="批量大小（默认: 10）"
    )

    args = parser.parse_args()

    print("SoulX-Podcast 批量请求客户端")
    print(f"API地址: {args.url}")
    print()

    # 确保输出目录存在
    Path("api/outputs").mkdir(parents=True, exist_ok=True)

    if args.mode == "batch":
        test_batch_generation(args.url, args.batch_size)
    elif args.mode == "multi":
        test_batch_multi_speaker(args.url, args.batch_size)
    elif args.mode == "compare":
        compare_batch_vs_sequential(args.url, args.batch_size)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
