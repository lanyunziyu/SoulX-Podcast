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
class SegmentState:
    """单段状态追踪"""
    segment_idx: int
    vllm_request_id: Optional[str] = None  # vLLM请求ID
    llm_completed: bool = False
    llm_completed_at: Optional[float] = None
    audio_task_id: Optional[str] = None  # flow+hifi任务ID
    audio_completed: bool = False
    audio_completed_at: Optional[float] = None
    audio_result: Optional[np.ndarray] = None


@dataclass
class BatchRequest:
    """批处理请求数据模型"""
    request_id: str
    dialogue_text: str
    mode: str  # "010", "120" etc.

    # 对话元数据
    is_multi_speaker: bool
    total_segments: int
    speak : int
    segments: List[Dict]

    # 生成参数
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float

    # 状态管理
    status: RequestStatus = RequestStatus.PENDING
    current_segment: int = 0

    # 新增：多段状态追踪
    segment_states: List[SegmentState] = field(default_factory=list)
    accumulated_audios: Dict[int, np.ndarray] = field(default_factory=dict)  # {seg_idx: audio}
    inputs_prompt: Optional[List[int]] = None  # LLM上下文状态

    # 时间戳
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):
        """初始化segment状态"""
        if not self.segment_states:
            self.segment_states = [
                SegmentState(segment_idx=i)
                for i in range(self.total_segments)
            ]


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
                # output_path = config.output_dir / f"{request_id}.wav"
                # wavfile.write(str(output_path), sample_rate, audio)

                # 保留原有的created_at（如果存在）
                existing_record = self.results.get(request_id, {})
                created_at = existing_record.get('created_at', time.time())

                self.results[request_id] = {
                    'status': RequestStatus.COMPLETED,
                    # 'output_path': output_path,
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
        self.result_store = RequestResultStore()

        # 统计信息
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.global_start_time: Optional[float] = None
        self.total_tokens_produced: int = 0
        self.lock = asyncio.Lock()
        self.total_audio_duration_produced: float = 0.0

        logger.info("AsyncBatchManager initialized")

    async def add_request(self, request: BatchRequest):
        """添加请求到队列"""
        try:
            await self.result_store.store_pending(request.request_id, request.created_at)
            await self.pending_queue.put(request)
            self.total_requests += 1

            logger.info(f"Added request {request.request_id} to queue (queue_size={self.pending_queue.qsize()})")
        except asyncio.QueueFull:
            error_msg = "Batch queue is full, please try again later"
            await self.result_store.store_failure(request.request_id, error_msg)
            logger.error(f"Failed to add request {request.request_id}: {error_msg}")
            raise RuntimeError(error_msg)


    async def store_result(self, request_id: str, audio: np.ndarray, token_count: int):
        """存储请求结果"""
        sample_rate = 24000 
        current_audio_duration = len(audio) / sample_rate        
        async with self.lock:
            # 1. 第一次结果完成时，标记全局开始时间
            if self.global_start_time is None:
                self.global_start_time = time.time()

            # 2. 累加数据
            self.total_tokens_produced += token_count
            self.total_audio_duration_produced += current_audio_duration
            self.completed_requests += 1

            # 3. 计算系统级瞬时指标 (System-level metrics)
            elapsed = time.time() - self.global_start_time
            system_tps = self.total_tokens_produced / elapsed
            system_rtf = elapsed / self.total_audio_duration_produced
            audio_speed = self.total_audio_duration_produced / elapsed
            # 打印全局看板
            logger.info(
                f"\n==== Global Performance Monitor ====\n"
                f"Completed: {self.completed_requests}/{self.total_requests}\n"
                f"Elapsed Time: {elapsed:.2f}s\n"
                f"Total Audio: {self.total_audio_duration_produced:.2f}s\n"
                f"--------------------------------------\n"
                f"System TPS: {system_tps:.2f} tokens/s\n"
                f"System RTF: {system_rtf:.4f} (Speed: {audio_speed:.2f}x)\n"
                f"========================================"
            )
        await self.result_store.store_result(request_id=request_id, audio=audio)
        # self.completed_requests += 1

    async def store_failure(self, request_id: str, error: str):
        """存储失败结果"""
        await self.result_store.store_failure(request_id, error)
        self.failed_requests += 1

    async def get_result(self, request_id: str) -> Optional[Dict]:
        """查询请求结果"""
        return await self.result_store.get_result(request_id)
    
    async def get_next_request(self) -> BatchRequest:
            """Worker 调用，阻塞直到拿到新任务"""
            return await self.pending_queue.get()    




# 全局单例
_batch_manager: Optional[AsyncBatchManager] = None


def get_batch_manager() -> AsyncBatchManager:
    """获取全局批处理管理器实例"""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = AsyncBatchManager()
    return _batch_manager
