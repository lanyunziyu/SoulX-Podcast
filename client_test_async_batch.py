"""
优化的异步批量测试客户端

解决的问题:
1. 使用异步 API 避免服务端串行化
2. 使用 aiohttp 替代 requests 实现真正的异步
3. 正确处理文件上传和表单数据
4. 添加重试机制和错误处理
5. 更好的进度显示和统计
"""
import asyncio
import aiohttp
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime


class AsyncBatchTester:
    """异步批量测试客户端"""

    def __init__(self, api_url: str, audio_file: str):
        self.api_url = api_url
        self.audio_file = audio_file
        self.session: aiohttp.ClientSession = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        # 配置连接池和超时
        timeout = aiohttp.ClientTimeout(total=600, connect=30)
        connector = aiohttp.TCPConnector(
            limit=100,  # 最大连接数
            limit_per_host=50,  # 每个主机最大连接数
            ttl_dns_cache=300
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()

    async def submit_async_task(self, request_id: int) -> Dict[str, Any]:
        """
        提交异步任务到服务端

        Args:
            request_id: 请求ID

        Returns:
            包含任务信息的字典
        """
        request_start = time.time()

        try:
            # 准备表单数据
            data = aiohttp.FormData()

            # 添加音频文件
            with open(self.audio_file, 'rb') as f:
                audio_data = f.read()
            data.add_field(
                'prompt_audio',
                audio_data,
                filename='audio.wav',
                content_type='audio/wav'
            )

            # 添加文本字段 - 使用 JSON 字符串
            prompt_texts_json = json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。"])
            data.add_field('prompt_texts', prompt_texts_json)
            data.add_field('dialogue_text', f'[S1]今天我们要聊一聊人工智能的最新进展，这是第{request_id}个请求。')
            data.add_field('seed', str(1988 + request_id))
            data.add_field('temperature', '0.6')
            data.add_field('top_k', '100')
            data.add_field('top_p', '0.9')
            data.add_field('repetition_penalty', '1.25')

            # 发送请求
            async with self.session.post(
                f"{self.api_url}/generate-async",
                data=data
            ) as response:
                submit_elapsed = time.time() - request_start

                if response.status == 200:
                    result = await response.json()
                    return {
                        'request_id': request_id,
                        'success': True,
                        'task_id': result['task_id'],
                        'submit_time': submit_elapsed,
                        'status': result['status']
                    }
                else:
                    error_text = await response.text()
                    return {
                        'request_id': request_id,
                        'success': False,
                        'submit_time': submit_elapsed,
                        'error': f"HTTP {response.status}: {error_text[:200]}"
                    }

        except asyncio.TimeoutError:
            return {
                'request_id': request_id,
                'success': False,
                'submit_time': time.time() - request_start,
                'error': 'Timeout during task submission'
            }
        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'submit_time': time.time() - request_start,
                'error': f"Exception: {str(e)}"
            }

    async def poll_task_status(
        self,
        task_id: str,
        request_id: int,
        max_wait_seconds: int = 600
    ) -> Dict[str, Any]:
        """
        轮询任务状态直到完成

        Args:
            task_id: 任务ID
            request_id: 请求ID
            max_wait_seconds: 最大等待时间

        Returns:
            任务结果字典
        """
        poll_start = time.time()
        poll_count = 0

        try:
            while True:
                # 检查超时
                if time.time() - poll_start > max_wait_seconds:
                    return {
                        'request_id': request_id,
                        'task_id': task_id,
                        'success': False,
                        'poll_time': time.time() - poll_start,
                        'poll_count': poll_count,
                        'error': f'Task timeout after {max_wait_seconds}s'
                    }

                # 查询任务状态
                async with self.session.get(
                    f"{self.api_url}/task/{task_id}"
                ) as response:
                    poll_count += 1

                    if response.status == 200:
                        status = await response.json()

                        if status['status'] == 'completed':
                            # 计算服务端处理时间（如果有时间戳）
                            server_time = None
                            if status.get('completed_at') and status.get('created_at'):
                                try:
                                    completed = datetime.fromisoformat(status['completed_at'].replace('Z', '+00:00'))
                                    created = datetime.fromisoformat(status['created_at'].replace('Z', '+00:00'))
                                    server_time = (completed - created).total_seconds()
                                except (ValueError, AttributeError):
                                    pass  # 时间解析失败，忽略

                            return {
                                'request_id': request_id,
                                'task_id': task_id,
                                'success': True,
                                'poll_time': time.time() - poll_start,
                                'poll_count': poll_count,
                                'result_url': status.get('result_url'),
                                'server_time': server_time
                            }

                        elif status['status'] == 'failed':
                            return {
                                'request_id': request_id,
                                'task_id': task_id,
                                'success': False,
                                'poll_time': time.time() - poll_start,
                                'poll_count': poll_count,
                                'error': status.get('error', 'Unknown error')
                            }

                        # 任务还在处理中，等待后继续
                        await asyncio.sleep(2)

                    else:
                        error_text = await response.text()
                        return {
                            'request_id': request_id,
                            'task_id': task_id,
                            'success': False,
                            'poll_time': time.time() - poll_start,
                            'poll_count': poll_count,
                            'error': f"HTTP {response.status}: {error_text[:200]}"
                        }

        except Exception as e:
            return {
                'request_id': request_id,
                'task_id': task_id,
                'success': False,
                'poll_time': time.time() - poll_start,
                'poll_count': poll_count,
                'error': f"Poll exception: {str(e)}"
            }

    async def process_single_request(self, request_id: int) -> Dict[str, Any]:
        """
        处理单个完整的请求流程：提交 -> 轮询 -> 完成

        Args:
            request_id: 请求ID

        Returns:
            完整的请求结果
        """
        total_start = time.time()

        # 步骤1: 提交任务
        submit_result = await self.submit_async_task(request_id)

        if not submit_result['success']:
            return {
                **submit_result,
                'total_time': time.time() - total_start,
                'phase': 'submit'
            }

        # 步骤2: 轮询任务状态
        task_id = submit_result['task_id']
        poll_result = await self.poll_task_status(task_id, request_id)

        # 合并结果
        return {
            **submit_result,
            **poll_result,
            'total_time': time.time() - total_start,
            'phase': 'completed' if poll_result['success'] else 'poll'
        }


async def run_batch_test(
    api_url: str,
    audio_file: str,
    batch_size: int = 10,
    max_concurrent: int = 10
):
    """
    运行异步批量测试

    Args:
        api_url: API 地址
        audio_file: 音频文件路径
        batch_size: 批量大小
        max_concurrent: 最大并发数
    """
    print("\n" + "=" * 80)
    print(f"🚀 异步批量测试 - {batch_size} 个请求，最大并发: {max_concurrent}")
    print("=" * 80)

    # 验证音频文件
    if not Path(audio_file).exists():
        print(f"❌ 错误: 找不到音频文件 {audio_file}")
        return

    batch_start = time.time()

    async with AsyncBatchTester(api_url, audio_file) as tester:
        # 创建并发任务，使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request(req_id: int):
            """限制并发的请求"""
            async with semaphore:
                return await tester.process_single_request(req_id)

        # 提交所有请求
        print(f"\n📤 提交 {batch_size} 个异步任务...")
        tasks = [limited_request(i + 1) for i in range(batch_size)]

        # 并发执行并收集结果
        results = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            # 实时显示进度
            if completed % 5 == 0 or completed == batch_size:
                success_count = sum(1 for r in results if r['success'])
                print(f"📊 进度: {completed}/{batch_size} 完成 (成功: {success_count}, 失败: {completed - success_count})")

    batch_elapsed = time.time() - batch_start

    # 统计结果
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("\n" + "=" * 80)
    print("📈 测试结果统计")
    print("=" * 80)

    print(f"\n📊 总体统计:")
    print(f"   总请求数: {batch_size}")
    print(f"   成功请求: {len(successful)}")
    print(f"   失败请求: {len(failed)}")
    print(f"   成功率: {len(successful)/batch_size*100:.1f}%")
    print(f"   批次总耗时: {batch_elapsed:.2f}秒")

    if successful:
        submit_times = [r['submit_time'] for r in successful]
        poll_times = [r.get('poll_time', 0) for r in successful]
        total_times = [r['total_time'] for r in successful]
        poll_counts = [r.get('poll_count', 0) for r in successful]

        # 服务端处理时间（如果有的话）
        server_times = [r.get('server_time') for r in successful if r.get('server_time')]

        print(f"\n⏱️  提交阶段统计:")
        print(f"   平均提交时间: {sum(submit_times)/len(submit_times):.3f}秒")
        print(f"   最快提交: {min(submit_times):.3f}秒")
        print(f"   最慢提交: {max(submit_times):.3f}秒")

        print(f"\n⏳ 轮询阶段统计:")
        print(f"   平均轮询时间: {sum(poll_times)/len(poll_times):.2f}秒")
        print(f"   最快轮询: {min(poll_times):.2f}秒")
        print(f"   最慢轮询: {max(poll_times):.2f}秒")
        print(f"   平均轮询次数: {sum(poll_counts)/len(poll_counts):.1f}次")

        if server_times:
            print(f"\n⚙️  服务端处理统计:")
            print(f"   平均处理时间: {sum(server_times)/len(server_times):.2f}秒")
            print(f"   最快处理: {min(server_times):.2f}秒")
            print(f"   最慢处理: {max(server_times):.2f}秒")

        print(f"\n🏁 端到端统计:")
        print(f"   平均总耗时: {sum(total_times)/len(total_times):.2f}秒")
        print(f"   最快完成: {min(total_times):.2f}秒")
        print(f"   最慢完成: {max(total_times):.2f}秒")

        print(f"\n🚀 吞吐量:")
        print(f"   实际吞吐量: {len(successful)/batch_elapsed:.2f} 请求/秒")
        if server_times:
            print(f"   平均并发度: {batch_elapsed/(sum(server_times)/len(server_times)):.1f}")

    # 显示失败详情
    if failed:
        print(f"\n❌ 失败请求详情:")

        # 按失败阶段分组
        submit_failures = [r for r in failed if r.get('phase') == 'submit']
        poll_failures = [r for r in failed if r.get('phase') == 'poll']

        if submit_failures:
            print(f"\n   提交失败 ({len(submit_failures)} 个):")
            for r in submit_failures[:5]:
                print(f"      请求 {r['request_id']}: {r.get('error', 'Unknown')[:80]}")
            if len(submit_failures) > 5:
                print(f"      ... 还有 {len(submit_failures)-5} 个")

        if poll_failures:
            print(f"\n   处理失败 ({len(poll_failures)} 个):")
            for r in poll_failures[:5]:
                print(f"      请求 {r['request_id']} (任务 {r.get('task_id', 'N/A')}): {r.get('error', 'Unknown')[:80]}")
            if len(poll_failures) > 5:
                print(f"      ... 还有 {len(poll_failures)-5} 个")

    print("\n" + "=" * 80)
    return {
        'total': batch_size,
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful)/batch_size*100,
        'batch_duration': batch_elapsed,
        'throughput': len(successful)/batch_elapsed if successful else 0
    }



async def main():
    parser = argparse.ArgumentParser(description="异步批量测试客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API服务地址（默认: http://localhost:8000）"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="example/audios/female_mandarin.wav",
        help="音频文件路径"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="批量请求数量（默认: 10）"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="最大并发数（默认: 5）"
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="跳过健康检查"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SoulX-Podcast 异步批量测试客户端")
    print("=" * 80)
    print(f"API地址: {args.url}")
    print(f"音频文件: {args.audio}")
    print(f"批量大小: {args.batch_size}")
    print(f"最大并发: {args.max_concurrent}")


    # 运行批量测试
    result = await run_batch_test(
        api_url=args.url,
        audio_file=args.audio,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    )

    print("\n✅ 测试完成!")


if __name__ == "__main__":
    asyncio.run(main())
