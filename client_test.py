"""
API测试客户端示例

功能: 测试同步、多说话人、异步和健康检查等功能的客户端代码。

使用示例:
    # 确保服务端已启动: python run_server.py
    python client_test.py --mode sync
    python client_test.py --mode async
"""
import requests
import time
import json
import argparse
import os
import concurrent.futures
from pathlib import Path


def test_sync_single_speaker(api_url: str):
    """测试同步生成 - 单说话人 (使用传统方式上传文件)"""
    print("\n" + "=" * 60)
    print("测试: 同步生成 - 单说话人 (传统方式)")
    print("=" * 60)

    # 准备文件 (假设文件路径存在)
    audio_file = "example/audios/female_mandarin.wav"
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        print("请确保 'example/audios/female_mandarin.wav' 存在")
        return

    # files 用于发送二进制文件 (prompt_audio)
    files = {
        'prompt_audio': open(audio_file, 'rb')
    }
    # data 用于发送 JSON 或表单文本字段
    data = {
        # prompt_texts 必须是 JSON 字符串，因为它是列表 [str]
        'prompt_texts': json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。"]),
        # dialogue_text 是一个长的文本字符串
        'dialogue_text': '[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。',
        'seed': 1988
    }

    print(f"发送请求到: {api_url}/generate")
    start_time = time.time()

    try:
        # 发送 multipart/form-data 请求
        response = requests.post(f"{api_url}/generate", files=files, data=data)
        response.raise_for_status()

        # 保存结果
        output_path = "api/outputs/test_single_sync.wav"
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)

        elapsed = time.time() - start_time
        print(f"✓ 生成成功!")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  保存到: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
    finally:
        # 必须关闭文件对象
        files['prompt_audio'].close()


def test_cache_performance(api_url: str):
    """测试缓存性能 - 相同音频和文本的多次请求"""
    print("\n" + "=" * 60)
    print("测试: 缓存性能测试")
    print("=" * 60)

    # 准备文件
    audio_file = "example/audios/female_mandarin.wav"
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        return

    dialogue_text = '[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。'

    print("说明: 我们将使用相同的音频和文本发送3次请求")
    print("第1次: 处理并缓存特征（慢）")
    print("第2次和第3次: 使用缓存（快）")
    print()

    times = []

    for i in range(1, 2):
        print(f"\n--- 第{i}次请求 ---")

        files = {'prompt_audio': open(audio_file, 'rb')}
        data = {
            'prompt_texts': json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。"]),
            'dialogue_text': dialogue_text,
            'seed': 1988
        }

        start_time = time.time()

        try:
            response = requests.post(f"{api_url}/generate", files=files, data=data)
            response.raise_for_status()

            elapsed = time.time() - start_time
            times.append(elapsed)

            # 保存结果
            output_path = f"api/outputs/test_cache_{i}.wav"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)

            cache_status = "❄️  (处理+缓存)" if i == 1 else "⚡ (使用缓存)"
            print(f"✓ 生成成功! {cache_status}")
            print(f"  耗时: {elapsed:.2f}秒")
            print(f"  保存到: {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"✗ 请求失败: {e}")
        finally:
            files['prompt_audio'].close()

    # # 统计
    # if len(times) == 3:
    #     print("\n" + "=" * 60)
    #     print("📊 缓存性能统计")
    #     print("=" * 60)
    #     print(f"第1次请求 (无缓存): {times[0]:.2f}秒")
    #     print(f"第2次请求 (有缓存): {times[1]:.2f}秒")
    #     print(f"第3次请求 (有缓存): {times[2]:.2f}秒")

    #     if times[0] > times[1]:
    #         speedup = times[0] / times[1]
    #         print(f"\n⚡ 缓存加速比: {speedup:.2f}x")
    #         print(f"💾 节省时间: {times[0] - times[1]:.2f}秒 ({(1 - times[1]/times[0])*100:.1f}%)")
    #     else:
    #         print("\n⚠️  注意: 第2次请求未出现预期的加速，可能缓存未生效")

def test_sync_single_speaker_batch(api_url: str, batch_size: int = 100, max_workers: int = 10):
    """测试同步生成 - 单说话人批量并发请求"""
    print("\n" + "=" * 60)
    print(f"测试: 同步生成 - 单说话人批量并发 ({batch_size}个请求)")
    print("=" * 60)

    # 准备文件
    audio_file = "example/audios/female_mandarin.wav"
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        print("请确保 'example/audios/female_mandarin.wav' 存在")
        return

    def single_request(request_id: int):
        """单个请求函数"""
        request_start = time.time()

        # 每个请求都需要独立的文件对象
        with open(audio_file, 'rb') as audio_fp:
            files = {
                'prompt_audio': audio_fp
            }
            data = [
                'prompt_texts', json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。"]),
                'dialogue_text', f'今天我们要聊一聊人工智能的最新进展。',
                # ('seed', str(1988 + request_id))  # 使用不同的seed
                'save_output', 'False'
            ]

            try:
                response = requests.post(f"{api_url}/generate-async", files=files, data=data, timeout=300)
                response.raise_for_status()
                if response.headers.get('content-type') == 'application/json':
                    response_size = len(response.text.encode('utf-8'))
                else:
                    response_size = len(response.content)

                # 保存结果
                # output_path = f"api/outputs/batch_test/test_batch_{request_id:03d}.wav"
                # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                # with open(output_path, 'wb') as f:
                #     f.write(response.content)

                elapsed = time.time() - request_start
                return {
                    'request_id': request_id,
                    'success': True,
                    'duration': elapsed,
                    # 'output_path': output_path,
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
            if completed_count % 10 == 0 or completed_count == batch_size:
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

        print(f"\n⏱️  响应时间统计:")
        print(f"   平均响应时间: {sum(durations)/len(durations):.2f}秒")
        print(f"   最快响应: {min(durations):.2f}秒")
        print(f"   最慢响应: {max(durations):.2f}秒")
        print(f"   响应时间中位数: {sorted(durations)[len(durations)//2]:.2f}秒")

        print(f"\n🚀 吞吐量统计:")
        print(f"   实际并发度: {max_workers}")
        print(f"   平均吞吐量: {len(successful_results)/batch_elapsed:.2f} 请求/秒")
        print(f"   理论最大吞吐: {batch_size/batch_elapsed:.2f} 请求/秒")

        print(f"\n💾 响应大小:")
        print(f"   平均响应大小: {sum(response_sizes)/len(response_sizes)/1024:.1f} KB")
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


def test_sync_multi_speaker(api_url: str):
    """测试同步生成 - 多说话人"""
    print("\n" + "=" * 60)
    print("测试: 同步生成 - 多说话人")
    print("=" * 60)

    # 准备文件
    audio_files = [
        "example/audios/female_mandarin.wav",
        "example/audios/male_mandarin.wav"
    ]

    for f in audio_files:
        if not Path(f).exists():
            print(f"错误: 找不到音频文件 {f}")
            print("请确保 'example/audios/female_mandarin.wav' 和 'example/audios/male_mandarin.wav' 存在")
            return

    # files 必须是包含元组 (字段名, 文件对象) 的列表
    files = [
        ('prompt_audio', open(audio_files[0], 'rb')),
        ('prompt_audio', open(audio_files[1], 'rb'))
    ]
    # FastAPI List[str] Form 需要多个同名字段
    data = [
        ('prompt_texts', "喜欢攀岩、徒步、滑雪的语言爱好者。"),
        ('prompt_texts', "资深科技播客主持人。"),
        ('dialogue_text', '[S1]大家好，欢迎收听今天的节目。[S2]是的，今天我们要聊聊人工智能。[S1]这个话题确实很有趣。'),
        ('seed', '1988')  # 表单字段建议用字符串
    ]

    print(f"发送请求到: {api_url}/generate")
    start_time = time.time()

    try:
        response = requests.post(f"{api_url}/generate", files=files, data=data)
        response.raise_for_status()

        # 保存结果
        output_path = "api/outputs/test_multi_sync.wav"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)

        elapsed = time.time() - start_time
        print(f"✓ 生成成功!")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  保存到: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
    finally:
        # 必须关闭所有文件对象
        for _, file_obj in files:
            file_obj.close()


def test_async(api_url: str):
    """测试异步生成"""
    print("\n" + "=" * 60)
    print("测试: 异步生成")
    print("=" * 60)

    # 准备文件
    audio_files = [
        "example/audios/female_mandarin.wav",
        "example/audios/male_mandarin.wav"
    ]

    for f in audio_files:
        if not Path(f).exists():
            print(f"错误: 找不到音频文件 {f}")
            print("请确保 'example/audios/female_mandarin.wav' 和 'example/audios/male_mandarin.wav' 存在")
            return

    files = [
        ('prompt_audio', open(audio_files[0], 'rb')),
        ('prompt_audio', open(audio_files[1], 'rb'))
    ]
    data = {
        'prompt_texts': json.dumps([
            "喜欢攀岩、徒步、滑雪的语言爱好者。",
            "资深科技播客主持人。"
        ]),
        'dialogue_text': '[S1]欢迎收听本期节目。[S2]今天的话题是AI语音合成。[S1]这确实是个很有意思的方向。[S2]没错，让我们深入探讨一下。',
        'seed': 1988
    }

    print(f"提交异步任务到: {api_url}/generate-async")

    try:
        # 提交任务
        response = requests.post(f"{api_url}/generate-async", files=files, data=data)
        response.raise_for_status()
        result = response.json()

        task_id = result['task_id']
        print(f"✓ 任务已创建: {task_id}")

        # 轮询任务状态
        print("\n等待任务完成...")
        max_attempts = 120  
        attempt = 0

        while attempt < max_attempts:
            time.sleep(2)
            attempt += 1

            status_response = requests.get(f"{api_url}/task/{task_id}")
            status_response.raise_for_status()
            status = status_response.json()

            print(f"  [{attempt}] 状态: {status['status']}, 进度: {status.get('progress', 0)}%")

            if status['status'] == 'completed':
                print(f"\n✓ 任务完成!")

                # 下载结果
                download_url = f"{api_url}{status['result_url']}"
                print(f"  下载URL: {download_url}")

                audio_response = requests.get(download_url)
                audio_response.raise_for_status()

                output_path = "api/outputs/test_async.wav"
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(audio_response.content)

                print(f"  保存到: {output_path}")
                break

            elif status['status'] == 'failed':
                print(f"\n✗ 任务失败: {status.get('error', '未知错误')}")
                break

        else:
            print(f"\n✗ 超时: 任务未在{max_attempts * 2}秒内完成")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
    finally:
        for _, file_obj in files:
            file_obj.close()


def test_health(api_url: str):
    """测试健康检查"""
    print("\n" + "=" * 60)
    print("测试: 健康检查")
    print("=" * 60)

    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        health = response.json()

        print(f"✓ API运行正常")
        print(f"  状态: {health['status']}")
        print(f"  模型已加载: {health['model_loaded']}")
        print(f"  GPU可用: {health['gpu_available']}")
        print(f"  活跃任务: {health['active_tasks']}")
        print(f"  版本: {health['version']}")

    except requests.exceptions.RequestException as e:
        print(f"✗ 健康检查失败: {e}")


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
        choices=["health", "sync", "async", "cache", "all"],
        default="sync",
        help="测试模式（默认: sync）。cache: 测试缓存性能"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="批量测试请求数量（默认: 10）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="最大并发线程数（默认: 10）"
    )

    args = parser.parse_args()

    print("SoulX-Podcast API 测试客户端")
    print(f"API地址: {args.url}")

    # 确保输出目录存在
    Path("api/outputs").mkdir(parents=True, exist_ok=True)

    if args.mode in ["health", "all"]:
        test_health(args.url)

    if args.mode in ["sync", "all"]:
        test_sync_single_speaker(args.url)
        # test_sync_multi_speaker(args.url)
        # test_sync_single_speaker_batch(args.url, args.batch_size, args.max_workers)

    if args.mode in ["cache", "all"]:
        test_cache_performance(args.url)

    if args.mode in ["async", "all"]:
        test_async(args.url)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()