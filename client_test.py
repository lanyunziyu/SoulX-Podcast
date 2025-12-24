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
import asyncio
import httpx
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
                response = requests.post(f"{api_url}/generate", files=files, data=data, timeout=300)
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





def get_random_dialogue(request_id: int):
    """
    根据请求 ID 生成不同的对话内容
    """
    subjects = ["人工智能", "量子计算", "深度学习", "自动驾驶", "生物科技", "星际探索", "数字艺术", "气候变化"]
    actions = ["最新进展", "未来挑战", "核心原理", "行业应用", "伦理问题", "技术突破"]
    
    sub = subjects[request_id % len(subjects)]
    act = actions[request_id % len(actions)]
    
    # 模拟真实的多段标注格式 [S1]
    templates = [
        f"[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。",
        f"[S1]欢迎收听科技频道。[S1]欢迎收听今天的节目，今天我们要聊一聊[S1]人工智能的最新进展。",
        f"[S1]深度探讨时刻。[S1]今天的主题是{sub}，重点关注其{act}。"
    ]
    return templates[request_id % len(templates)]


async def monitor_task_wakeup(client: httpx.AsyncClient, api_url: str, task_id: str, req_idx: int):
    """
    【唤醒监听协程】
    利用后端 /task/{id} 的长轮询机制，实现结果一出来就立刻“唤醒”客户端。
    """
    start_time = time.time()
    # timeout=60 是传递给后端的，告诉后端：没结果请让我的连接挂起 60 秒
    poll_url = f"{api_url}/task/{task_id}?timeout=60"
    
    try:
        # httpx 的 timeout 必须大于后端的长轮询 timeout
        response = await client.get(poll_url, timeout=65)
        response.raise_for_status()
        status_data = response.json()

        # 如果后端因为超时返回了 processing 状态，我们需要继续发起请求（虽然通常一次长轮询就够了）
        while status_data['status'] not in ['completed', 'failed']:
            response = await client.get(poll_url, timeout=65)
            status_data = response.json()

        end_time = time.time()
        duration = end_time - start_time
        
        if status_data['status'] == 'completed':
            print(f"✨ [唤醒通知] 请求 {req_idx} ({task_id[:8]}) 成功! 耗时: {duration:.2f}s")
            return {"task_id": task_id, "success": True, "duration": duration}
        else:
            print(f"❌ [失败通知] 请求 {req_idx} ({task_id[:8]}) 失败: {status_data.get('error')}")
            return {"task_id": task_id, "success": False, "duration": duration}

    except Exception as e:
        print(f"⚠️ [网络异常] 请求 {req_idx}: {e}")
        return {"task_id": task_id, "success": False, "duration": 0}

async def test_async_batch_generation_wakeup(api_url: str, batch_size: int = 5, mode: str = "120"):
    """
    重写后的异步批量测试：
    1. 批量分发任务
    2. 并发唤醒监听
    """
    print("\n" + "=" * 60)
    print(f"🚀 启动测试: 异步批量生成 (唤醒模式) | 规模: {batch_size} | 模式: {mode}")
    print("=" * 60)

    # 1. 准备批量请求数据
    batch_requests = []
    # 模拟混合负载：1个多段对话，其余单段
    # batch_requests.append({"dialogue_text": "[S1]大家好，欢迎收听今天的节目。[S2]是的，今天我们要聊聊人工智能。[S1]这个话题确实很有趣。"})
    # batch_requests.append({"dialogue_text": "[S1]哈喽，AI时代的冲浪先锋们！欢迎收听《AI生活进行时》[S2]哎，大家好呀！我是能唠，爱唠，天天都想唠的唠嗑！[S1]最近活得特别赛博朋克哈！以前老是觉得AI是科幻片儿里的"})
    for i in range(batch_size):
        batch_requests.append({"dialogue_text": f"[S1]大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。"})

    async with httpx.AsyncClient() as client:
        # Step A: 批量提交任务 (Dispatch)
        print(f"正在分发 {batch_size} 个任务到后端队列...")
        dispatch_start = time.time()
        
        try:
            submit_resp = await client.post(
                f"{api_url}/generate-batch-async",
                data={
                    'batch_requests': json.dumps(batch_requests),
                    'mode': mode,
                    'speak':1,
                },
                timeout=15
            )
            submit_resp.raise_for_status()
            tasks_info = submit_resp.json()
            task_ids = [t['task_id'] for t in tasks_info]
            print(f"✓ 分发成功! 耗时: {time.time()-dispatch_start:.2f}s, 已获得 {len(task_ids)} 个任务ID")
        except Exception as e:
            print(f"✗ 任务提交失败: {e}")
            return

        # Step B: 并发监听唤醒 (Listen)
        print(f"\n⏳ 正在挂起等待后端唤醒结果 (不占用 CPU)...\n")
        
        # 为每个任务 ID 创建一个协程任务
        monitor_coroutines = [
            monitor_task_wakeup(client, api_url, tid, i+1) 
            for i, tid in enumerate(task_ids)
        ]
        
        # 使用 gather 并发执行所有监听
        all_results = await asyncio.gather(*monitor_coroutines)

        # Step C: 结果统计
        total_duration = time.time() - dispatch_start
        success_results = [r for r in all_results if r['success']]
        
        print("\n" + "=" * 60)
        print("📈 异步唤醒模式统计结果")
        print("=" * 60)
        print(f"总请求数: {len(all_results)}")
        print(f"成功完成: {len(success_results)}")
        print(f"总运行时间: {total_duration:.2f}秒")
        if success_results:
            avg_task_time = sum(r['duration'] for r in success_results) / len(success_results)
            print(f"任务平均周期: {avg_task_time:.2f}秒")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="API测试客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8001",
        help="API服务地址（默认: http://localhost:8000）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sync", "all", "preset", "async-batch"],
        default="async-batch",
        help="测试模式（默认: preset）。preset: 测试预设模式, batch: 测试批量生成, async-batch: 测试异步批量生成"
    )
    parser.add_argument(
        "--preset-mode",
        type=str,
        default="120",
        choices=["000", "001", "010", "011", "120", "121"],
        help="预设模式参数: 000=单人男生普通话, 001=单人男生英语, 010=单人女生普通话, 011=单人女生英语, 120=双人普通话, 121=双人英语（默认: 010）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批量测试请求数量（默认: 10）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="最大并发线程数（默认: 10）"
    )

    args = parser.parse_args()

    print("SoulX-Podcast API 测试客户端")
    print(f"API地址: {args.url}")

    # 确保输出目录存在
    Path("api/outputs").mkdir(parents=True, exist_ok=True)

    if args.mode in ["sync", "all"]:
        test_sync_single_speaker(args.url)
        # test_sync_multi_speaker(args.url)
        # test_sync_single_speaker_batch(args.url, args.batch_size, args.max_workers)

    
    if args.mode == "async-batch":
        # 测试异步批量生成功能
        asyncio.run(test_async_batch_generation_wakeup(args.url, args.batch_size, args.preset_mode))


    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()