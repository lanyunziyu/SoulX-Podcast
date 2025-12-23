"""
批处理后台Worker

负责从队列收集请求、组装batch、执行推理并分发结果
"""
import asyncio
import logging
import time
from typing import List, Dict
import torch
import numpy as np

from api.batch_manager import AsyncBatchManager, RequestStatus, get_batch_manager
from api.config import config
from api.service import get_service
from soulxpodcast.config import SamplingParams

logger = logging.getLogger(__name__)


def concatenate_audios(audio_segments: List[np.ndarray]) -> np.ndarray:
    """拼接多个音频段"""
    if not audio_segments:
        return np.array([])

    if len(audio_segments) == 1:
        return audio_segments[0]

    # 使用torch.concat保持一致性
    tensors = [torch.from_numpy(seg) for seg in audio_segments]
    concatenated = torch.concat(tensors, axis=0)
    return concatenated.numpy()


async def handle_request_pipeline(req, service, batch_manager, audio_semaphore):
    """
    单个请求的全流程流水线（协程）
    实现：Segment(N) LLM 完 -> 立即触发 Segment(N) Audio & Segment(N+1) LLM
    """
    try:
        current_inputs_prompt = req.inputs_prompt
        
        for seg_idx in range(req.total_segments):
            # --- 阶段 1: LLM 推理 ---
            seg = req.segments[seg_idx]
            llm_inputs, context = service.model.prepare_llm_inputs(
                segment=seg, mode=req.mode, inputs_prompt=current_inputs_prompt,
                preloaded_data=service.preloaded_data.get(req.mode)
            )

            # 提交 vLLM 并等待（vLLM 内部会自动处理并发和 Batching）
            llm_output = await service.model.llm.generate_async(
                llm_inputs,
                sampling_param=SamplingParams(
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    extra_args={"use_ras": True, "win_size": 25, "tau_r": 0.2}
                )
            )
            
            # 更新状态和上下文供下一段使用
            req.segment_states[seg_idx].llm_completed = True
            current_inputs_prompt = service.model.update_inputs_prompt(
                current_inputs_prompt, llm_output['token_ids']
            )

            # --- 阶段 2: 启动 Audio 并发处理 (不阻塞下一段 LLM) ---
            async def process_audio_task(s_idx, l_out, ctx):
                async with audio_semaphore:
                    audio = await service.model.process_llm_result_async(l_out, ctx)
                    req.accumulated_audios[s_idx] = audio
                    req.segment_states[s_idx].audio_completed = True
                    
                    # 检查是否全部段落都完成了音频合成
                    if all(state.audio_completed for state in req.segment_states):
                        final_audio = concatenate_audios([req.accumulated_audios[i] for i in range(req.total_segments)])
                        await batch_manager.store_result(req.request_id, final_audio)
                        logger.info(f"Request {req.request_id} fully completed.")

            # 关键：这里用 create_task 启动音频合成，循环会立即进入下一个 seg_idx 的 LLM 推理
            asyncio.create_task(process_audio_task(seg_idx, llm_output, context))

    except Exception as e:
        logger.error(f"Pipeline error for {req.request_id}: {e}", exc_info=True)
        await batch_manager.store_failure(req.request_id, str(e))

async def batch_worker_loop(batch_manager):
    service = get_service()
    # 限制 Flow+HiFi 并发，GPU 算力分配
    audio_semaphore = asyncio.Semaphore(4) 
    
    while True:
        # 被动等待队列
        request = await batch_manager.get_next_request()
        # 为每个新请求开启独立协程
        asyncio.create_task(handle_request_pipeline(request, service, batch_manager, audio_semaphore))


async def cleanup_worker():
    """定期清理旧结果的worker"""
    logger.info("Cleanup worker started")
    batch_manager = get_batch_manager()

    while True:
        try:
            await asyncio.sleep(3600)  # 每小时清理一次
            await batch_manager.result_store.cleanup_old_results(max_age_seconds=86400)  # 清理24小时前的结果
        except asyncio.CancelledError:
            logger.info("Cleanup worker cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup worker error: {e}", exc_info=True)
