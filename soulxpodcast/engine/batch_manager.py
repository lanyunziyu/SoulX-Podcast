"""
Batch Request Manager for SoulXPodcast

管理多个请求的batch处理，区分request_id和turn_id
"""
import logging
import threading
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from queue import Queue, Empty
from enum import Enum

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """请求状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TurnRequest:
    """单个turn的请求"""
    request_id: str  # 请求ID
    turn_id: int     # Turn ID (0-based)
    total_turns: int # 该请求的总turn数

    # LLM输入
    llm_input: List[int]  # token ids

    # Flow输入参数
    prompt_mel: Any  # torch.Tensor
    prompt_mel_len: Any  # torch.Tensor
    spk_emb: Any  # torch.Tensor
    spk_id: int

    # 采样参数
    sampling_params: Any

    # 元数据
    created_at: float = field(default_factory=time.time)

    def get_key(self) -> str:
        """获取唯一标识"""
        return f"{self.request_id}_turn_{self.turn_id}"


@dataclass
class BatchResult:
    """Batch处理结果"""
    request_id: str
    turn_id: int

    # LLM输出
    llm_output_tokens: List[int]

    # Flow输出
    generated_mel: Any  # torch.Tensor
    generated_mel_len: int

    # HiFiGAN输出
    generated_wav: Any  # torch.Tensor

    # 时间统计
    llm_time: float = 0.0
    flow_time: float = 0.0
    hifigan_time: float = 0.0


class BatchRequestManager:
    """
    Batch请求管理器

    功能：
    1. 收集多个请求的turns
    2. 按照显存容量自动调度batch
    3. 维护request_id和turn_id的映射关系
    """

    def __init__(self, max_batch_size: int = 32, max_queue_size: int = 1000):
        """
        Args:
            max_batch_size: 最大batch大小
            max_queue_size: 请求队列最大长度
        """
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size

        # 请求队列
        self.request_queue: Queue[TurnRequest] = Queue(maxsize=max_queue_size)

        # 结果存储: {request_id_turn_id: BatchResult}
        self.results: Dict[str, BatchResult] = {}

        # 请求状态: {request_id: {turn_id: status}}
        self.request_status: Dict[str, Dict[int, RequestStatus]] = {}

        # 统计信息
        self.total_requests = 0
        self.total_turns_processed = 0

        # 线程锁
        self.lock = threading.Lock()

        logger.info(f"BatchRequestManager initialized: max_batch_size={max_batch_size}")

    def add_turn_request(self, turn_request: TurnRequest):
        """
        添加turn请求到队列

        Args:
            turn_request: TurnRequest对象
        """
        with self.lock:
            # 初始化请求状态
            if turn_request.request_id not in self.request_status:
                self.request_status[turn_request.request_id] = {}
                self.total_requests += 1

            self.request_status[turn_request.request_id][turn_request.turn_id] = RequestStatus.PENDING

        # 添加到队列
        self.request_queue.put(turn_request, block=True)
        logger.debug(f"Added turn request: {turn_request.get_key()}, queue_size={self.request_queue.qsize()}")

    def get_batch(self, batch_size: Optional[int] = None, timeout: float = 0.1) -> List[TurnRequest]:
        """
        从队列获取一个batch的请求

        Args:
            batch_size: batch大小，如果None则使用max_batch_size
            timeout: 等待超时时间（秒）

        Returns:
            List[TurnRequest]: batch请求列表
        """
        if batch_size is None:
            batch_size = self.max_batch_size

        batch = []
        deadline = time.time() + timeout

        while len(batch) < batch_size and time.time() < deadline:
            try:
                remaining_time = max(0.001, deadline - time.time())
                turn_request = self.request_queue.get(timeout=remaining_time)
                batch.append(turn_request)

                # 更新状态
                with self.lock:
                    self.request_status[turn_request.request_id][turn_request.turn_id] = RequestStatus.PROCESSING

            except Empty:
                break

        if batch:
            logger.info(f"Got batch of {len(batch)} turn requests")

        return batch

    def store_result(self, result: BatchResult):
        """
        存储处理结果

        Args:
            result: BatchResult对象
        """
        key = f"{result.request_id}_turn_{result.turn_id}"

        with self.lock:
            self.results[key] = result
            self.request_status[result.request_id][result.turn_id] = RequestStatus.COMPLETED
            self.total_turns_processed += 1

        logger.debug(f"Stored result: {key}")

    def get_result(self, request_id: str, turn_id: int) -> Optional[BatchResult]:
        """
        获取指定turn的结果

        Args:
            request_id: 请求ID
            turn_id: Turn ID

        Returns:
            BatchResult or None
        """
        key = f"{request_id}_turn_{turn_id}"
        return self.results.get(key)

    def get_all_results(self, request_id: str) -> List[BatchResult]:
        """
        获取指定请求的所有turn结果

        Args:
            request_id: 请求ID

        Returns:
            List[BatchResult]: 按turn_id排序的结果列表
        """
        results = []

        if request_id not in self.request_status:
            return results

        for turn_id in sorted(self.request_status[request_id].keys()):
            result = self.get_result(request_id, turn_id)
            if result:
                results.append(result)

        return results

    def is_request_completed(self, request_id: str) -> bool:
        """
        检查请求是否全部完成

        Args:
            request_id: 请求ID

        Returns:
            bool: 是否全部完成
        """
        if request_id not in self.request_status:
            return False

        turn_statuses = self.request_status[request_id]
        return all(status == RequestStatus.COMPLETED for status in turn_statuses.values())

    def get_queue_size(self) -> int:
        """获取队列当前大小"""
        return self.request_queue.qsize()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            pending_turns = sum(
                1 for req_turns in self.request_status.values()
                for status in req_turns.values()
                if status == RequestStatus.PENDING
            )
            processing_turns = sum(
                1 for req_turns in self.request_status.values()
                for status in req_turns.values()
                if status == RequestStatus.PROCESSING
            )

            return {
                "total_requests": self.total_requests,
                "total_turns_processed": self.total_turns_processed,
                "queue_size": self.request_queue.qsize(),
                "pending_turns": pending_turns,
                "processing_turns": processing_turns,
                "completed_requests": sum(1 for req_id in self.request_status.keys() if self.is_request_completed(req_id))
            }

    def cleanup_old_results(self, max_age_seconds: float = 3600):
        """
        清理旧的结果

        Args:
            max_age_seconds: 结果保留时间（秒）
        """
        current_time = time.time()
        keys_to_remove = []

        with self.lock:
            for key, result in self.results.items():
                # 计算结果年龄（简化处理，使用第一个result的时间）
                if hasattr(result, 'created_at'):
                    age = current_time - getattr(result, 'created_at', current_time)
                    if age > max_age_seconds:
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.results[key]
                logger.debug(f"Cleaned up old result: {key}")

        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} old results")
