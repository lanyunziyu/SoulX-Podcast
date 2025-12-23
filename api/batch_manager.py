"""
异步批处理管理器

管理批处理请求队列、状态跟踪和结果存储
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

from api.config import config

logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    """请求状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchRequest:
    """批处理请求数据模型"""
    request_id: str
    dialogue_text: str
    mode: str  # "010", "120" etc.

    # 对话元数据
    is_multi_speaker: bool
    total_segments: int
    segments: List[Dict]

    # 生成参数
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float

    # 状态管理
    status: RequestStatus = RequestStatus.PENDING
    current_segment: int = 0
    accumulated_audios: List[np.ndarray] = field(default_factory=list)
    inputs_prompt: Optional[List[int]] = None  # 多轮LLM上下文

    # 时间戳
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None


class RequestResultStore:
    """请求结果存储"""

    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()

    async def store_result(self, request_id: str, audio: np.ndarray, sample_rate: int = 24000):
        """存储请求结果"""
        async with self.lock:
            try:
                # 保存音频文件
                output_path = config.output_dir / f"{request_id}.wav"
                wavfile.write(str(output_path), sample_rate, audio)

                # 保留原有的created_at（如果存在）
                existing_record = self.results.get(request_id, {})
                created_at = existing_record.get('created_at', time.time())

                self.results[request_id] = {
                    'status': RequestStatus.COMPLETED,
                    'output_path': output_path,
                    'sample_rate': sample_rate,
                    'audio_length': len(audio),
                    'created_at': created_at,
                    'completed_at': time.time()
                }

                logger.info(f"Stored result for request {request_id}: {len(audio)/sample_rate:.2f}s audio")
            except Exception as e:
                logger.error(f"Failed to store result for request {request_id}: {e}")
                existing_record = self.results.get(request_id, {})
                created_at = existing_record.get('created_at', time.time())

                self.results[request_id] = {
                    'status': RequestStatus.FAILED,
                    'error': str(e),
                    'created_at': created_at,
                    'completed_at': time.time()
                }

    async def store_pending(self, request_id: str, created_at: float):
        """存储pending状态"""
        async with self.lock:
            self.results[request_id] = {
                'status': RequestStatus.PENDING,
                'created_at': created_at
            }
            logger.debug(f"Stored pending status for request {request_id}")

    async def store_failure(self, request_id: str, error: str):
        """存储失败结果"""
        async with self.lock:
            self.results[request_id] = {
                'status': RequestStatus.FAILED,
                'error': error,
                'completed_at': time.time()
            }
            logger.error(f"Stored failure for request {request_id}: {error}")

    async def get_result(self, request_id: str) -> Optional[Dict]:
        """查询请求结果"""
        async with self.lock:
            result = self.results.get(request_id)
            logger.debug(f"get_result({request_id}): found={result is not None}, total_results={len(self.results)}")
            return result

    async def cleanup_old_results(self, max_age_seconds: int = 86400):
        """清理旧结果（默认24小时）"""
        async with self.lock:
            cutoff_time = time.time() - max_age_seconds
            expired_ids = [
                req_id for req_id, result in self.results.items()
                if result.get('completed_at', 0) < cutoff_time
            ]

            for req_id in expired_ids:
                result = self.results[req_id]
                # 删除音频文件
                if 'output_path' in result:
                    try:
                        Path(result['output_path']).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete file for {req_id}: {e}")

                del self.results[req_id]

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired results")


class AsyncBatchManager:
    """异步批处理管理器"""

    def __init__(self):
        self.pending_queue: asyncio.Queue[BatchRequest] = asyncio.Queue(
            maxsize=config.max_batch_queue_size
        )
        self.processing_multi_speaker: Dict[str, BatchRequest] = {}
        self.result_store = RequestResultStore()

        # 统计信息
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0

        logger.info("AsyncBatchManager initialized")

    async def add_request(self, request: BatchRequest):
        """添加请求到队列"""
        try:
            await self.pending_queue.put(request)
            self.total_requests += 1

            # 立即在result_store中创建pending状态记录
            await self.result_store.store_pending(request.request_id, request.created_at)

            logger.info(f"Added request {request.request_id} to queue (queue_size={self.pending_queue.qsize()})")
        except asyncio.QueueFull:
            error_msg = "Batch queue is full, please try again later"
            await self.result_store.store_failure(request.request_id, error_msg)
            logger.error(f"Failed to add request {request.request_id}: {error_msg}")
            raise RuntimeError(error_msg)

    async def get_batch(self, batch_size: int, timeout: float) -> List[BatchRequest]:
        """
        收集batch，支持超时机制

        Args:
            batch_size: 目标batch大小
            timeout: 超时时间（秒）

        Returns:
            BatchRequest列表
        """
        batch = []
        deadline = asyncio.get_event_loop().time() + timeout

        # 1. 优先添加多人对话的continuation segments
        continuation_requests = []
        for req_id, req in list(self.processing_multi_speaker.items()):
            if req.current_segment < req.total_segments:
                continuation_requests.append(req)
                if len(continuation_requests) >= batch_size:
                    break

        batch.extend(continuation_requests)

        # 2. 从pending_queue填充新请求
        while len(batch) < batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break  # 超时，立即返回当前batch

            try:
                request = await asyncio.wait_for(
                    self.pending_queue.get(),
                    timeout=min(remaining, 0.1)
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        if batch:
            logger.info(f"Collected batch: size={len(batch)}, "
                       f"continuation={len(continuation_requests)}, "
                       f"new={len(batch)-len(continuation_requests)}")

        return batch

    async def store_result(self, request_id: str, audio: np.ndarray):
        """存储请求结果"""
        await self.result_store.store_result(request_id, audio)
        self.completed_requests += 1

    async def store_failure(self, request_id: str, error: str):
        """存储失败结果"""
        await self.result_store.store_failure(request_id, error)
        self.failed_requests += 1

    async def get_result(self, request_id: str) -> Optional[Dict]:
        """查询请求结果"""
        return await self.result_store.get_result(request_id)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'failed_requests': self.failed_requests,
            'pending_queue_size': self.pending_queue.qsize(),
            'processing_multi_speaker': len(self.processing_multi_speaker),
        }


# 全局单例
_batch_manager: Optional[AsyncBatchManager] = None


def get_batch_manager() -> AsyncBatchManager:
    """获取全局批处理管理器实例"""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = AsyncBatchManager()
    return _batch_manager
