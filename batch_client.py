"""
API测试客户端示例

功能: 测试同步、多说话人、异步和健康检查等功能的客户端代码。

使用示例:
    # 确保服务端已启动: python run_server.py
    # 运行单次同步测试
    python client_test.py --mode sync
    # 运行并发测试 (100个请求，10个并发线程)
    python client_test.py --mode batch --batch-size 100 --max-workers 10
    python client_test.py --mode async
"""
import requests
import time
import json
import argparse
import os
import concurrent.futures
from pathlib import Path

# ==============================================================================
# 批量并发测试函数 (已修改为适合压测的逻辑)
# ==============================================================================

def test_sync_single_speaker_batch(api_url: str, batch_size: int = 100, max_workers: int = 10):
    """测试同步生成 - 单说话人批量并发请求 (用于压测)"""
    print("\n" + "=" * 60)
    print(f"测试: 同步生成 - 单说话人批量并发 ({batch_size}个请求)")
    print("=" * 60)

    # 准备文件
    audio_file = "example/audios/female_mandarin.wav"
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        print("请确保 'example/audios/female_mandarin.wav' 存在")
        return

    # 使用固定的对话内容，以确保测试的是系统吞吐量
    DIALOGUE_TEXT = '[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。'
    PROMPT_TEXTS = json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。"]) # 使用 json.dumps 保证格式正确

    def single_request(request_id: int):
        """单个请求函数"""
        request_start = time.time()

        # 每个请求都需要独立的文件对象
        # 注意：在并发环境中，必须在每次请求时重新打开文件，以保证线程安全
        with open(audio_file, 'rb') as audio_fp:
            
            # files 必须是字典 {字段名: 文件对象}
            files = {
                'prompt_audio': audio_fp
            }
            # data 必须是字典 {字段名: 值}，或者元组列表 [(k, v), ...]
            # 这里的 prompt_texts 必须是 JSON 字符串，而不是列表
            data = {
                'prompt_texts': PROMPT_TEXTS, # 关键：这里直接使用 JSON 字符串
                'dialogue_text': DIALOGUE_TEXT,
                # 'seed': str(1988 + request_id),  # 使用不同的seed
                'save_output': 'False'
            }

            try:
                # 发送 multipart/form-data 请求
                response = requests.post(f"{api_url}/generate", files=files, data=data, timeout=300)
                response.raise_for_status()
                if response.headers.get('content-type') == 'application/json':
                    response_size = len(response.text.encode('utf-8'))
                else:
                    response_size = len(response.content)
                # 不保存内容，以最小化 I/O 带来的影响
                # output_path = f"api/outputs/batch_test/test_batch_{request_id:03d}.wav"
                # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                # with open(output_path, 'wb') as f:
                #     f.write(response.content)

                elapsed = time.time() - request_start
                return {
                    'request_id': request_id,
                    'success': True,
                    'duration': elapsed,
                    'response_size': len(response.content)
                }

            except Exception as e:
                elapsed = time.time() - request_start
                return {
                    'request_id': request_id,
                    'success': False,
                    'duration': elapsed,
                    'error': str(e)
                }

    print(f"🚀 启动 {batch_size} 个并发请求 (最大并发: {max_workers})...")
    batch_start = time.time()

    # 使用线程池并发执行
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有请求
        future_to_id = {executor.submit(single_request, i+1): i+1 for i in range(batch_size)}

        # 收集结果
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_id):
            result = future.result()
            results.append(result)
            completed_count += 1

            # 实时显示进度
            if completed_count % (batch_size // 10 or 1) == 0 or completed_count == batch_size:
                success_count = sum(1 for r in results if r['success'])
                print(f"📊 进度: {completed_count}/{batch_size} 完成 (成功: {success_count})")

    batch_elapsed = time.time() - batch_start

    # 统计结果
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    print("\n" + "=" * 60)
    print("📈 批量并发测试结果统计")
    print("=" * 60)
    print(f"总请求数: {batch_size}")
    print(f"成功请求: {len(successful_results)}")
    print(f"失败请求: {len(failed_results)}")
    print(f"成功率: {len(successful_results)/batch_size*100:.1f}%")
    print(f"批次总耗时: {batch_elapsed:.2f}秒")

    if successful_results:
        durations = [r['duration'] for r in successful_results]
        response_sizes = [r['response_size'] for r in successful_results]

        # 计算 P95 延迟
        durations.sort()
        p95_index = int(len(durations) * 0.95) - 1
        p95_latency = durations[p95_index]
        
        print(f"\n⏱️  响应时间统计:")
        print(f"   平均响应时间: {sum(durations)/len(durations):.2f}秒")
        print(f"   最快响应: {min(durations):.2f}秒")
        print(f"   最慢响应: {max(durations):.2f}秒")
        print(f"   P95 响应时间: {p95_latency:.2f}秒")
        print(f"   响应时间中位数 (P50): {durations[len(durations)//2]:.2f}秒")

        print(f"\n🚀 吞吐量统计:")
        print(f"   实际并发度: {max_workers}")
        print(f"   平均吞吐量: {len(successful_results)/batch_elapsed:.2f} 请求/秒")
        print(f"   总数据量: {sum(response_sizes)/1024/1024:.1f} MB")

    # 显示失败的请求
    if failed_results:
        print(f"\n❌ 失败请求详情:")
        for result in failed_results[:10]:  # 只显示前10个失败请求
            print(f"   请求 {result['request_id']}: {result['error']}")
        if len(failed_results) > 10:
            print(f"   ... 还有 {len(failed_results)-10} 个失败请求")

    return {
        'total_requests': batch_size,
        'successful': len(successful_results),
        'failed': len(failed_results),
        'batch_duration': batch_elapsed,
        'success_rate': len(successful_results)/batch_size*100,
        'avg_response_time': sum(r['duration'] for r in successful_results)/len(successful_results) if successful_results else 0,
        'throughput': len(successful_results)/batch_elapsed if successful_results else 0
    }

# ==============================================================================
# 其他不变的函数 (test_sync_single_speaker, test_sync_multi_speaker, test_async, test_health)
# ... (为节省篇幅省略，请使用您提供的原文件中的代码) ...
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="API测试客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API服务地址（默认: http://localhost:8000）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        # 增加 'batch' 模式选项
        choices=["health", "sync", "async", "batch", "all"], 
        default="all",
        help="测试模式（默认: all）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100, # 默认设置为100个请求
        help="批量测试请求数量（默认: 100）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10, # 默认设置为10个并发
        help="最大并发线程数（默认: 10）"
    )

    args = parser.parse_args()

    print("SoulX-Podcast API 测试客户端")
    print(f"API地址: {args.url}")

    # 确保输出目录存在
    Path("api/outputs").mkdir(parents=True, exist_ok=True)
    Path("api/outputs/batch_test").mkdir(parents=True, exist_ok=True) # 确保批量测试目录存在


    # 新增批量并发测试模式
    if args.mode in ["batch", "all"]:
        # 注意：这里直接调用了我们修改后的并发函数
        test_sync_single_speaker_batch(args.url, args.batch_size, args.max_workers) 


    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()