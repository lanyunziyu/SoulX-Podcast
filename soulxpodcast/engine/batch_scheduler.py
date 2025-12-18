"""
Batch Scheduler with GPU Memory Awareness

根据GPU显存自动调整batch大小
"""
import logging
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """显存使用配置"""
    total_memory: float  # GB
    reserved_memory: float  # GB，系统预留
    available_memory: float  # GB，可用于batch

    # 每个组件的显存占用估算（单位：GB per sample）
    llm_memory_per_sample: float = 0.5
    flow_memory_per_sample: float = 0.3
    hifigan_memory_per_sample: float = 0.1

    def estimate_max_batch_size(self) -> int:
        """估算最大batch大小"""
        # 使用最保守的估算（LLM通常占用最多）
        max_batch = int(self.available_memory / self.llm_memory_per_sample)
        return max(1, max_batch)


class BatchScheduler:
    """
    Batch调度器

    功能：
    1. 启动时检测GPU显存
    2. 动态调整batch大小
    3. 监控显存使用情况
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        memory_margin: float = 0.2,  # 20%的显存余量
        enable_dynamic: bool = True,
    ):
        """
        Args:
            min_batch_size: 最小batch大小
            max_batch_size: 最大batch大小
            memory_margin: 显存安全余量（0-1）
            enable_dynamic: 是否启用动态调整
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_margin = memory_margin
        self.enable_dynamic = enable_dynamic

        # 显存配置
        self.memory_profile: Optional[MemoryProfile] = None
        self.current_batch_size = min_batch_size

        # 统计信息
        self.total_batches = 0
        self.oom_count = 0  # Out of Memory错误计数

        # 初始化
        self._detect_gpu_memory()

        logger.info(f"BatchScheduler initialized: current_batch_size={self.current_batch_size}")

    def _detect_gpu_memory(self):
        """检测GPU显存"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU mode")
            self.current_batch_size = self.min_batch_size
            return

        try:
            # 获取第一个GPU的显存信息
            device = torch.device("cuda:0")
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB

            # 获取当前已使用显存
            torch.cuda.empty_cache()
            reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB

            # 计算可用显存
            available_memory = total_memory * (1 - self.memory_margin) - allocated_memory

            self.memory_profile = MemoryProfile(
                total_memory=total_memory,
                reserved_memory=reserved_memory,
                available_memory=max(0, available_memory)
            )

            # 估算初始batch大小
            estimated_batch = self.memory_profile.estimate_max_batch_size()
            self.current_batch_size = min(estimated_batch, self.max_batch_size)
            self.current_batch_size = max(self.current_batch_size, self.min_batch_size)

            logger.info(f"GPU Memory detected:")
            logger.info(f"  Total: {total_memory:.2f} GB")
            logger.info(f"  Reserved: {reserved_memory:.2f} GB")
            logger.info(f"  Allocated: {allocated_memory:.2f} GB")
            logger.info(f"  Available: {available_memory:.2f} GB")
            logger.info(f"  Estimated batch size: {self.current_batch_size}")

        except Exception as e:
            logger.error(f"Failed to detect GPU memory: {e}")
            self.current_batch_size = self.min_batch_size

    def get_batch_size(self) -> int:
        """
        获取当前推荐的batch大小

        Returns:
            int: batch大小
        """
        return self.current_batch_size

    def report_batch_success(self, batch_size: int, memory_used: Optional[float] = None):
        """
        报告batch成功执行

        Args:
            batch_size: 执行的batch大小
            memory_used: 使用的显存（GB），可选
        """
        self.total_batches += 1

        if self.enable_dynamic and memory_used is not None:
            # 如果显存使用率较低，可以尝试增加batch
            if self.memory_profile:
                usage_ratio = memory_used / self.memory_profile.available_memory
                if usage_ratio < 0.6 and self.current_batch_size < self.max_batch_size:
                    old_size = self.current_batch_size
                    self.current_batch_size = min(self.current_batch_size + 1, self.max_batch_size)
                    logger.info(f"Increased batch size: {old_size} -> {self.current_batch_size}")

    def report_batch_failure(self, batch_size: int, error: Exception):
        """
        报告batch执行失败

        Args:
            batch_size: 执行的batch大小
            error: 错误信息
        """
        # 检查是否是OOM错误
        is_oom = "out of memory" in str(error).lower() or "cuda" in str(error).lower()

        if is_oom:
            self.oom_count += 1
            logger.warning(f"OOM error detected, total OOM count: {self.oom_count}")

            # 减小batch大小
            if self.enable_dynamic and self.current_batch_size > self.min_batch_size:
                old_size = self.current_batch_size
                self.current_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
                logger.info(f"Reduced batch size due to OOM: {old_size} -> {self.current_batch_size}")

                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        获取当前显存使用情况

        Returns:
            Dict: 显存使用信息（GB）
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "total": 0}

        device = torch.device("cuda:0")
        return {
            "allocated": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved": torch.cuda.memory_reserved(device) / (1024**3),
            "total": torch.cuda.get_device_properties(device).total_memory / (1024**3),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "current_batch_size": self.current_batch_size,
            "total_batches": self.total_batches,
            "oom_count": self.oom_count,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
        }

        if self.memory_profile:
            stats.update({
                "total_memory_gb": self.memory_profile.total_memory,
                "available_memory_gb": self.memory_profile.available_memory,
            })

        # 添加当前显存使用
        stats.update(self.get_current_memory_usage())

        return stats

    def reset_batch_size(self):
        """重置batch大小到初始估算值"""
        self._detect_gpu_memory()
        logger.info(f"Batch size reset to {self.current_batch_size}")
