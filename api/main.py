"""
FastAPI Main Application for SoulX-Podcast Voice Cloning API
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime
import json
import threading

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import scipy.io.wavfile as wavfile

from api.config import config
from api.models import (
    TaskCreateResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse,
    TaskStatus,
)
from api.service import get_service
from api.tasks import get_task_manager
from api.batch_manager import get_batch_manager, BatchRequest, RequestStatus
from api.batch_worker import batch_worker_loop, cleanup_worker
from api.utils import (
    generate_task_id,
    save_upload_file,
    validate_audio_files,
    validate_dialogue_format,
    cleanup_old_files,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建一个全局锁来控制同步推理的并发
inference_lock = threading.Lock()
active_inferences = 0
MAX_CONCURRENT_SYNC_INFERENCES = 1  # 限制同步推理的并发数


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting SoulX-Podcast API...")

    # 初始化模型（在主线程）
    logger.info("Loading model...")
    service = get_service()
    if not service.is_loaded():
        raise RuntimeError("Failed to load model")

    # 启动任务管理器
    task_manager = get_task_manager()
    task_manager.start_workers(config.max_concurrent_tasks)

    # 启动批处理worker
    batch_manager = get_batch_manager()
    batch_workers = []
    for i in range(config.num_batch_workers):
        worker = asyncio.create_task(batch_worker_loop(batch_manager))
        batch_workers.append(worker)
    logger.info(f"Started {len(batch_workers)} batch worker(s)")

    # 启动批处理结果清理worker
    cleanup_worker_task = asyncio.create_task(cleanup_worker())

    # 启动文件清理任务
    async def cleanup_task():
        while True:
            await asyncio.sleep(600)  # 每10分钟清理一次
            count = cleanup_old_files(config.temp_dir, config.file_cleanup_minutes)
            count += cleanup_old_files(config.output_dir, config.file_cleanup_minutes)
            if count > 0:
                logger.info(f"Cleaned up {count} old files")

    cleanup_task_handle = asyncio.create_task(cleanup_task())

    logger.info("API started successfully!")

    yield

    # 关闭时
    logger.info("Shutting down API...")

    # 关闭批处理workers
    for worker in batch_workers:
        worker.cancel()
    cleanup_worker_task.cancel()
    await asyncio.gather(*batch_workers, cleanup_worker_task, return_exceptions=True)

    cleanup_task_handle.cancel()

    # 快速关闭任务管理器
    try:
        await asyncio.wait_for(task_manager.shutdown(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Task manager shutdown timeout, forcing exit")

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("API shutdown completed")


# 创建FastAPI应用
app = FastAPI(
    title="SoulX-Podcast Voice Cloning API",
    description="基于SoulX-Podcast的语音克隆API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """根路径"""
    return {
        "name": "SoulX-Podcast Voice Cloning API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    service = get_service()
    task_manager = get_task_manager()

    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded(),
        gpu_available=torch.cuda.is_available(),
        llm_engine=config.llm_engine,
        active_tasks=task_manager.get_active_task_count(),
        version="1.0.0"
    )


@app.post("/generate", tags=["Generation"])
async def generate_sync(
    prompt_audio: List[UploadFile] = File(default=None, description="参考音频文件（1-4个），如果提供mode参数则可选"),
    prompt_texts: List[str] = Form(default=None, description="参考文本JSON数组，如: [\"文本1\", \"文本2\"]，如果提供mode参数则可选"),
    dialogue_text: str = Form(..., description="要生成的对话文本"),
    mode: str = Form(default=None, description="模式参数(三位数字): 000=单人男生普通话, 010=单人女生普通话, 001=单人男生英语, 011=单人女生英语, 120=双人普通话, 121=双人英语"),
    seed: int = Form(default=1988, description="随机种子"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="采样温度"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K采样"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P采样"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="重复惩罚"),
    save_output: bool = Form(default=True, description="是否将生成的音频文件保存到磁盘"),
):
    """
    同步生成语音（直接返回音频文件）

    适用于短音频生成（预计<30秒）
    """
    task_id = generate_task_id()

    try:
        # 如果提供了mode参数，使用预加载数据
        if mode:
            logger.info(f"Using preset mode: {mode}")
            # 验证mode格式
            if not (len(mode) == 3 and mode.isdigit()):
                raise HTTPException(status_code=400, detail=f"无效的mode格式: {mode}，应为三位数字")

            # mode模式下不需要audio和texts
            audio_paths = []
            prompt_text_list = []

            # 根据mode推断说话人数量
            num_speakers = 2 if mode[0] == '1' else 1
        else:
            # 验证必须提供音频和文本
            if not prompt_audio or not prompt_texts:
                raise HTTPException(
                    status_code=400,
                    detail="必须提供mode参数或同时提供prompt_audio和prompt_texts"
                )
            # 验证音频文件
            validate_audio_files(prompt_audio)

            # 解析prompt_texts
            try:
                prompt_text_list = prompt_texts
                if not isinstance(prompt_text_list, list):
                    raise ValueError("prompt_texts必须是JSON数组")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"prompt_texts JSON格式错误: {str(e)}")

            # 验证数量匹配
            if len(prompt_audio) != len(prompt_text_list):
                raise HTTPException(
                    status_code=400,
                    detail=f"参考音频数量({len(prompt_audio)})与参考文本数量({len(prompt_text_list)})不匹配"
                )
            num_speakers = len(prompt_audio)
             # 保存上传的文件
            audio_paths = []
            for i, file in enumerate(prompt_audio):
                path = save_upload_file(file, task_id, i)
                audio_paths.append(str(path))

        # 验证对话格式
        is_valid, error_msg = validate_dialogue_format(dialogue_text, num_speakers)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)


        logger.info(f"Sync generation started: task_id={task_id}, speakers={len(audio_paths)}")

        # 调用服务生成
        service = get_service()
        sample_rate, audio_array = service.generate(
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            # seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            mode=mode,
        )

        # 保存结果
        # output_filename = f"{task_id}.wav"
        # output_path = config.output_dir / output_filename
        # wavfile.write(str(output_path), sample_rate, audio_array)

        # logger.info(f"Sync generation completed: task_id={task_id}")

        # # 返回文件
        # return FileResponse(
        #     path=str(output_path),
        #     media_type="audio/wav",
        #     filename=output_filename
        # )
        if save_output:
            # 保存结果
            output_filename = f"{task_id}.wav"
            output_path = config.output_dir / output_filename
            wavfile.write(str(output_path), sample_rate, audio_array)
            logger.info(f"Output saved to: {output_path}")
            # 返回文件
            return FileResponse(
                path=str(output_path),
                media_type="audio/wav",
                filename=output_filename
            )
        else:
            # 不保存文件，返回一个简单的 JSON 响应告知成功
            logger.info(f"Output not saved (save_output=False). Returning success JSON.")
            return JSONResponse(
                content={"message": "Audio generated successfully, file not saved to disk as requested.",
                         "sample_rate": sample_rate,
                         "audio_length_samples": len(audio_array)},
                status_code=200
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-batch", tags=["Generation"])
async def generate_batch(
    batch_requests: str = Form(..., description="批量请求JSON数组，每个请求包含dialogue_text字段"),
    mode: str = Form(..., description="模式参数(三位数字): 000=单人男生普通话, 010=单人女生普通话, 001=单人男生英语, 011=单人女生英语, 120=双人普通话, 121=双人英语"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="采样温度"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K采样"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P采样"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="重复惩罚"),
    return_format: str = Form(default="files", description="返回格式: 'files'=返回音频文件列表, 'json'=返回成功消息"),
):
    """
    批量生成语音（同步）

    适用于多个短音频的批量生成，每个请求包含一个S1对话文本
    """
    try:
        # 验证mode格式
        if not (len(mode) == 3 and mode.isdigit()):
            raise HTTPException(status_code=400, detail=f"无效的mode格式: {mode}，应为三位数字")

        # 解析批量请求
        try:
            batch_request_list = json.loads(batch_requests)
            if not isinstance(batch_request_list, list):
                raise ValueError("batch_requests必须是JSON数组")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"batch_requests JSON格式错误: {str(e)}")

        if not batch_request_list:
            raise HTTPException(status_code=400, detail="批量请求列表不能为空")

        if len(batch_request_list) > 100:  # 限制批量大小
            raise HTTPException(status_code=400, detail="批量请求数量不能超过100个")

        # 验证每个请求的格式
        for i, request in enumerate(batch_request_list):
            if not isinstance(request, dict):
                raise HTTPException(status_code=400, detail=f"批量请求{i}必须是JSON对象")

            dialogue_text = request.get('dialogue_text', '').strip()
            if not dialogue_text:
                raise HTTPException(status_code=400, detail=f"批量请求{i}缺少dialogue_text字段")

            # 根据mode推断说话人数量进行验证
            num_speakers = 2 if mode[0] == '1' else 1
            is_valid, error_msg = validate_dialogue_format(dialogue_text, num_speakers)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"批量请求{i}对话格式错误: {error_msg}")

        logger.info(f"Batch generation started: {len(batch_request_list)} requests with mode={mode}")

        # 调用服务生成
        service = get_service()
        batch_results = service.generate_batch(
            batch_requests=batch_request_list,
            mode=mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if return_format == "json":
            # 返回简单的成功消息
            return JSONResponse(
                content={
                    "message": f"成功生成{len(batch_results)}个音频",
                    "batch_size": len(batch_results),
                    "mode": mode,
                    "sample_rate": 24000 if batch_results else None,
                    "audio_lengths": [len(audio_array) for _, audio_array in batch_results]
                },
                status_code=200
            )
        else:
            # 保存结果文件并返回文件列表
            batch_task_id = generate_task_id()
            output_files = []

            for i, (sample_rate, audio_array) in enumerate(batch_results):
                output_filename = f"{batch_task_id}_batch_{i:03d}.wav"
                output_path = config.output_dir / output_filename
                wavfile.write(str(output_path), sample_rate, audio_array)
                output_files.append({
                    "index": i,
                    "filename": output_filename,
                    "download_url": f"/download/{output_filename}",
                    "duration_seconds": len(audio_array) / sample_rate
                })

            logger.info(f"Batch generation completed: {len(batch_results)} files saved")

            return JSONResponse(
                content={
                    "message": f"批量生成成功，共{len(batch_results)}个音频文件",
                    "batch_id": batch_task_id,
                    "batch_size": len(batch_results),
                    "mode": mode,
                    "files": output_files
                },
                status_code=200
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-batch-async", response_model=List[TaskCreateResponse], tags=["Generation"])
async def generate_batch_async(
    batch_requests: str = Form(..., description="批量请求JSON数组，每个请求包含dialogue_text字段"),
    mode: str = Form(..., description="模式参数(三位数字): 000=单人男生普通话, 010=单人女生普通话, 001=单人男生英语, 011=单人女生英语, 120=双人普通话, 121=双人英语"),
    batch_size: Optional[int] = Form(None, description="批次大小（可选，默认使用配置的default_batch_size）"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="采样温度"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K采样"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P采样"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="重复惩罚"),
):
    """
    异步批量生成语音（返回任务ID列表）

    适用于批量请求，每个请求独立返回，无需等待整个批次完成
    客户端通过 /task/{request_id} 轮询各个请求的状态
    """
    try:
        # 验证mode格式
        if not (len(mode) == 3 and mode.isdigit()):
            raise HTTPException(status_code=400, detail=f"无效的mode格式: {mode}，应为三位数字")

        # 解析批量请求
        try:
            batch_request_list = json.loads(batch_requests)
            if not isinstance(batch_request_list, list):
                raise ValueError("batch_requests必须是JSON数组")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"batch_requests JSON格式错误: {str(e)}")

        if not batch_request_list:
            raise HTTPException(status_code=400, detail="批量请求列表不能为空")

        if len(batch_request_list) > 100:
            raise HTTPException(status_code=400, detail="批量请求数量不能超过100个")

        # 验证batch_size
        effective_batch_size = batch_size or config.default_batch_size
        if effective_batch_size > config.max_batch_size:
            raise HTTPException(status_code=400, detail=f"batch_size不能超过{config.max_batch_size}")

        # 验证每个请求的格式
        for i, req_data in enumerate(batch_request_list):
            if not isinstance(req_data, dict):
                raise HTTPException(status_code=400, detail=f"批量请求{i}必须是JSON对象")

            dialogue_text = req_data.get('dialogue_text', '').strip()
            if not dialogue_text:
                raise HTTPException(status_code=400, detail=f"批量请求{i}缺少dialogue_text字段")

            # 根据mode推断说话人数量进行验证
            num_speakers = 2 if mode[0] == '1' else 1
            is_valid, error_msg = validate_dialogue_format(dialogue_text, num_speakers)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"批量请求{i}对话格式错误: {error_msg}")

        logger.info(f"Async batch generation started: {len(batch_request_list)} requests with mode={mode}, batch_size={effective_batch_size}")

        # 为每个请求创建BatchRequest对象
        batch_manager = get_batch_manager()
        service = get_service()
        responses = []

        logger.info("Creating BatchRequest objects...")
        for i, req_data in enumerate(batch_request_list):
            request_id = generate_task_id()
            logger.info(f"Processing request {i+1}/{len(batch_request_list)}: {request_id}")

            # 解析对话模式
            logger.info(f"Analyzing dialogue mode for request {i+1}...")
            analysis = service._analyze_dialogue_mode(req_data['dialogue_text'])
            logger.info(f"Analysis done: is_multi_speaker={analysis['is_multi_speaker']}, total_segments={analysis['total_segments']}")

            batch_req = BatchRequest(
                request_id=request_id,
                dialogue_text=req_data['dialogue_text'],
                mode=mode,
                is_multi_speaker=analysis['is_multi_speaker'],
                total_segments=analysis['total_segments'],
                segments=analysis['segments'],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                status=RequestStatus.PENDING,
            )

            # 加入队列
            await batch_manager.add_request(batch_req)

            responses.append(TaskCreateResponse(
                task_id=request_id,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                message="请求已加入批处理队列"
            ))

        logger.info(f"Added {len(responses)} requests to async batch queue")
        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async batch generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-async", response_model=TaskCreateResponse, tags=["Generation"])
async def generate_async(
    prompt_audio: List[UploadFile] = File(..., description="参考音频文件（1-4个）"),
    prompt_texts: str = Form(..., description="参考文本JSON数组"),
    dialogue_text: str = Form(..., description="要生成的对话文本"),
    # seed: int = Form(default=1988, description="随机种子"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="采样温度"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K采样"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P采样"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="重复惩罚"),
):
    """
    异步生成语音（返回任务ID）

    适用于长音频生成或批量任务
    """
    task_id = generate_task_id()

    try:
        # 验证音频文件
        validate_audio_files(prompt_audio)

        # 解析prompt_texts
        try:
            prompt_text_list = json.loads(prompt_texts)
            if not isinstance(prompt_text_list, list):
                raise ValueError("prompt_texts必须是JSON数组")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"prompt_texts JSON格式错误: {str(e)}")

        # 验证数量匹配
        if len(prompt_audio) != len(prompt_text_list):
            raise HTTPException(
                status_code=400,
                detail=f"参考音频数量({len(prompt_audio)})与参考文本数量({len(prompt_text_list)})不匹配"
            )

        # 验证对话格式
        is_valid, error_msg = validate_dialogue_format(dialogue_text, len(prompt_audio))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # 保存上传的文件
        audio_paths = []
        for i, file in enumerate(prompt_audio):
            path = save_upload_file(file, task_id, i)
            audio_paths.append(str(path))

        # 创建异步任务
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            task_id=task_id,
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            # seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        logger.info(f"Async task created: task_id={task_id}")

        return TaskCreateResponse(
            task_id=task_id,
            status=task.status,
            created_at=task.created_at,
            message=f"任务已创建，当前队列中有 {task_manager.queue.qsize()} 个任务"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str, timeout: int = 30):
    """
    查询任务状态（支持长轮询）

    Args:
        task_id: 任务ID
        timeout: 等待超时时间（秒），默认30秒。如果任务未完成，最多等待这么久

    长轮询机制：
    - 如果任务已完成/失败：立即返回结果
    - 如果任务进行中：等待最多timeout秒，期间每0.5秒检查一次状态
    - 如果超时仍未完成：返回当前状态（pending/processing）
    """
    import asyncio
    import time

    batch_manager = get_batch_manager()
    task_manager = get_task_manager()

    # 验证timeout范围
    timeout = max(1, min(timeout, 120))  # 限制在1-120秒
    start_time = time.time()
    check_interval = 0.5  # 每0.5秒检查一次

    logger.info(f"Query task {task_id} with long polling (timeout={timeout}s)")

    while True:
        # 优先查询批处理管理器
        batch_result = await batch_manager.get_result(task_id)

        if batch_result:
            # 批处理请求
            status_str = batch_result.get('status', RequestStatus.PENDING)
            if isinstance(status_str, RequestStatus):
                status_str = status_str.value

            # 将RequestStatus映射到TaskStatus
            status_mapping = {
                'pending': TaskStatus.PENDING,
                'processing': TaskStatus.RUNNING,
                'completed': TaskStatus.COMPLETED,
                'failed': TaskStatus.FAILED,
            }
            task_status = status_mapping.get(status_str, TaskStatus.PENDING)

            # 如果任务已完成或失败，立即返回
            if task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                result_url = None
                if batch_result['status'] == RequestStatus.COMPLETED and 'output_path' in batch_result:
                    result_url = f"/download/{batch_result['output_path'].name}"

                logger.info(f"Task {task_id} finished with status {task_status}")
                return TaskStatusResponse(
                    task_id=task_id,
                    status=task_status,
                    progress=100 if task_status == TaskStatus.COMPLETED else 0,
                    result_url=result_url,
                    error=batch_result.get('error'),
                    created_at=datetime.fromtimestamp(batch_result.get('created_at', 0)) if 'created_at' in batch_result else None,
                    started_at=None,
                    completed_at=datetime.fromtimestamp(batch_result['completed_at']) if 'completed_at' in batch_result else None,
                )

            # 任务进行中，检查是否超时
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.info(f"Task {task_id} still running, long poll timeout after {elapsed:.1f}s")
                return TaskStatusResponse(
                    task_id=task_id,
                    status=task_status,
                    progress=50 if task_status == TaskStatus.RUNNING else 0,
                    result_url=None,
                    error=None,
                    created_at=datetime.fromtimestamp(batch_result.get('created_at', 0)) if 'created_at' in batch_result else None,
                    started_at=None,
                    completed_at=None,
                )

            # 等待一段时间再检查
            await asyncio.sleep(check_interval)
            continue

        # 回退到单次任务管理器
        task = task_manager.get_task(task_id)

        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        # 如果任务已完成或失败，立即返回
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            result_url = None
            if task.status == TaskStatus.COMPLETED and task.result_path:
                result_url = f"/download/{task.result_path.name}"

            logger.info(f"Task {task_id} finished with status {task.status}")
            return TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result_url=result_url,
                error=task.error,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
            )

        # 任务进行中，检查是否超时
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            logger.info(f"Task {task_id} still running, long poll timeout after {elapsed:.1f}s")
            return TaskStatusResponse(
                task_id=task.task_id,
                status=task.status,
                progress=task.progress,
                result_url=None,
                error=None,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=None,
            )

        # 等待一段时间再检查
        await asyncio.sleep(check_interval)
        continue


@app.get("/download/{filename}", tags=["Download"])
async def download_file(filename: str):
    """下载生成的音频文件"""
    file_path = config.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="info"
    )
