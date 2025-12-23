"""
批处理后台Worker

负责从队列收集请求、组装batch、执行推理并分发结果
"""
import asyncio
import logging
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


async def run_batch_inference(
    segments: List[Dict],
    mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> Dict:
    """
    执行批量推理

    Args:
        segments: 当前轮次的segments列表
        mode: 模式参数
        temperature, top_k, top_p, repetition_penalty: 生成参数

    Returns:
        Dict包含audio_results和inputs_prompt
    """
    service = get_service()

    # 准备批量请求数据
    batch_requests = [{'dialogue_text': seg['text']} for seg in segments]
    text_ids_list, spks_lists, _ = service._prepare_batch_requests(batch_requests)

    # 提取inputs_prompt（如果有continuation）
    inputs_prompt = segments[0].get('inputs_prompt') if segments else None

    # 获取预加载数据
    preloaded = service.preloaded_data[mode]

    # === 关键修复：数据预处理 ===
    import s3tokenizer

    # 1. 处理prompt_mels（使用s3tokenizer.padding）
    prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(preloaded["log_mel_list"])

    # 2. 处理spk_emb
    spk_emb_for_flow = torch.tensor(preloaded["spk_emb_list"])

    # 3. 处理prompt_mels_for_flow（使用pad_sequence）
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
        preloaded["mel_list"], batch_first=True, padding_value=0
    )

    # 4. 创建采样参数对象
    sampling_params = SamplingParams(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        extra_args={
            "use_ras": True,
            "win_size": 25,
            "tau_r": 0.2,
        },
    )

    # 合并所有文本tokens和说话人ID
    all_text_tokens = []
    all_spk_ids = []
    for text_ids, spks in zip(text_ids_list, spks_lists):
        all_text_tokens.extend(text_ids)
        all_spk_ids.extend(spks)

    # 调用forward_batch（现在所有参数都是正确的tensor）
    batch_results = service.model.forward_batch(
        prompt_mels_for_llm=prompt_mels_for_llm,  # ✓ tensor
        prompt_mels_lens_for_llm=prompt_mels_lens_for_llm,  # ✓ tensor
        prompt_text_tokens_for_llm=preloaded["prompt_text_ids_list"],  # ✓ list[list[int]]
        text_tokens_for_llm=all_text_tokens,
        prompt_mels_for_flow_ori=prompt_mels_for_flow,  # ✓ tensor
        spk_emb_for_flow=spk_emb_for_flow,  # ✓ tensor
        sampling_params=sampling_params,  # ✓ SamplingParams对象
        spk_ids=all_spk_ids,
        inputs_prompt=inputs_prompt,
        return_state=True,  # 返回更新的inputs_prompt
    )

    return batch_results


async def batch_worker_loop(batch_manager: AsyncBatchManager):
    """
    后台worker主循环

    Args:
        batch_manager: 批处理管理器实例
    """
    logger.info("Batch worker started")

    MAX_RETRIES = 3

    while True:
        try:
            # 1. 收集batch
            batch = await batch_manager.get_batch(
                batch_size=config.default_batch_size,
                timeout=config.batch_timeout
            )

            if not batch:
                await asyncio.sleep(0.1)
                continue

            logger.info(f"Processing batch of {len(batch)} requests")

            # 2. 准备当前轮次的数据
            current_segments = []
            segment_to_request_mapping = []  # 记录每个segment对应的请求

            for req in batch:
                if req.is_multi_speaker:
                    # 多人对话：取当前segment
                    seg = req.segments[req.current_segment]
                    current_segments.append({
                        'request_id': req.request_id,
                        'text': f"[S{seg['speaker'] + 1}]{seg['text']}",  # 0-based转回1-based
                        'spk_id': seg['speaker'],
                        'inputs_prompt': req.inputs_prompt
                    })
                else:
                    # 单人对话：完整处理
                    current_segments.append({
                        'request_id': req.request_id,
                        'text': req.dialogue_text,
                        'spk_id': 0,
                        'inputs_prompt': None
                    })

                segment_to_request_mapping.append(req)

            # 3. 执行批量推理（带重试）
            results = None
            for attempt in range(MAX_RETRIES):
                try:
                    results = await run_batch_inference(
                        segments=current_segments,
                        mode=batch[0].mode,
                        temperature=batch[0].temperature,
                        top_k=batch[0].top_k,
                        top_p=batch[0].top_p,
                        repetition_penalty=batch[0].repetition_penalty
                    )
                    break  # 成功，跳出重试循环
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"OOM error on attempt {attempt+1}/{MAX_RETRIES}")
                    if attempt < MAX_RETRIES - 1:
                        torch.cuda.empty_cache()
                        await asyncio.sleep(1)
                        # 尝试拆分batch
                        if len(batch) > 1:
                            logger.warning("Splitting batch due to OOM")
                            mid = len(batch) // 2
                            # 将后半部分放回队列
                            for req in batch[mid:]:
                                await batch_manager.pending_queue.put(req)
                            batch = batch[:mid]
                            segment_to_request_mapping = segment_to_request_mapping[:mid]
                            current_segments = current_segments[:mid]
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Inference error on attempt {attempt+1}/{MAX_RETRIES}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(1)
                    else:
                        raise

            if results is None:
                logger.error("Batch inference failed after all retries")
                for req in batch:
                    await batch_manager.store_failure(req.request_id, "Inference failed after retries")
                continue

            # 4. 分发结果
            audio_results = results['audio_results']
            updated_inputs_prompt = results.get('inputs_prompt')

            for i, req in enumerate(segment_to_request_mapping):
                if i >= len(audio_results):
                    logger.error(f"Missing audio result for request {req.request_id}")
                    await batch_manager.store_failure(req.request_id, "Missing audio result")
                    continue

                audio = audio_results[i]

                try:
                    if req.is_multi_speaker:
                        # 累积音频段
                        req.accumulated_audios.append(audio)
                        req.inputs_prompt = updated_inputs_prompt
                        req.current_segment += 1
                        req.status = RequestStatus.PROCESSING

                        logger.info(f"Request {req.request_id}: segment {req.current_segment}/{req.total_segments} completed")

                        if req.current_segment >= req.total_segments:
                            # 所有段落完成，拼接音频
                            final_audio = concatenate_audios(req.accumulated_audios)
                            await batch_manager.store_result(req.request_id, final_audio)
                            req.status = RequestStatus.COMPLETED
                            req.completed_at = asyncio.get_event_loop().time()

                            # 从processing_multi_speaker中移除
                            if req.request_id in batch_manager.processing_multi_speaker:
                                del batch_manager.processing_multi_speaker[req.request_id]

                            logger.info(f"Request {req.request_id}: all segments completed, audio length: {len(final_audio)/24000:.2f}s")
                        else:
                            # 继续处理下一轮
                            batch_manager.processing_multi_speaker[req.request_id] = req
                    else:
                        # 单人对话直接完成
                        await batch_manager.store_result(req.request_id, audio)
                        req.status = RequestStatus.COMPLETED
                        req.completed_at = asyncio.get_event_loop().time()

                        logger.info(f"Request {req.request_id}: completed, audio length: {len(audio)/24000:.2f}s")

                except Exception as e:
                    logger.error(f"Failed to process result for request {req.request_id}: {e}")
                    await batch_manager.store_failure(req.request_id, str(e))

        except asyncio.CancelledError:
            logger.info("Batch worker cancelled")
            break
        except Exception as e:
            logger.error(f"Batch worker error: {e}", exc_info=True)
            await asyncio.sleep(1)  # 避免错误循环过快


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
