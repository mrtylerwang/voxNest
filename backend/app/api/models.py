"""
模型管理 API 路由
包含模型列表、下载、取消、删除、存储管理
"""
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import asyncio
from ..models.manager import model_manager
from ..models.storage import storage
from ..models.catalog import get_model_by_key
from ..downloader.queue import download_queue
from ..downloader.base import DownloadProgress
from ..config import config
from ..utils.logger import logger

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/list")
async def list_models(task_type: Optional[str] = None):
    """
    获取模型列表
    task_type: asr / tts / None（全部）
    """
    from ..engines.base import TaskType
    tt = None
    if task_type == "asr":
        tt = TaskType.ASR
    elif task_type == "tts":
        tt = TaskType.TTS

    models = model_manager.get_model_list(tt)
    return {"models": models}


@router.get("/auxiliary")
async def list_auxiliary_models():
    """获取辅助模型列表"""
    models = model_manager.get_auxiliary_model_list()
    return {"models": models}


@router.post("/download")
async def download_model(
    model_key: str = Form(...),
    download_source: Optional[str] = Form(None),
    use_hf_mirror: Optional[bool] = Form(None),
):
    """开始下载模型。download_source / use_hf_mirror 按本次请求指定，缺省回退全局配置。"""
    model_config = get_model_by_key(model_key)
    if model_config is None:
        raise HTTPException(status_code=400, detail=f"未知模型: {model_key}")

    # 如果模型已下载，先删除旧文件（重新下载场景）
    if storage.is_model_downloaded(model_key):
        logger.info(f"模型已下载，重新下载前删除旧文件: {model_key}")
        storage.delete_model(model_key)

    result = download_queue.start_download(
        model_key,
        download_source=download_source,
        use_hf_mirror=use_hf_mirror,
    )
    return result


@router.post("/cancel")
async def cancel_download(model_key: str = Form(...)):
    """取消下载"""
    result = download_queue.cancel_download(model_key)
    return result


@router.get("/progress/{model_key}")
async def get_download_progress(model_key: str):
    """获取指定模型的下载进度"""
    progress = download_queue.get_download_progress(model_key)
    if progress is None:
        return {"model_key": model_key, "status": "not_found"}
    return progress


@router.get("/downloads")
async def get_all_downloads():
    """获取所有下载任务状态"""
    downloads = download_queue.get_all_downloads()
    return {"downloads": downloads}


@router.get("/events")
async def download_events():
    """
    SSE 下载进度事件流
    前端通过 EventSource 订阅，实时接收所有模型的下载进度
    """
    async def event_generator():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=200)

        def enqueue_progress(progress: DownloadProgress) -> None:
            try:
                queue.put_nowait(progress)
            except asyncio.QueueFull:
                # 队列满时丢弃最旧事件，优先保留最新进度
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(progress)
                except asyncio.QueueFull:
                    pass

        def callback(progress: DownloadProgress):
            try:
                loop.call_soon_threadsafe(enqueue_progress, progress)
            except RuntimeError:
                # 连接关闭后事件循环可能已结束
                pass

        download_queue.add_global_progress_callback(callback)

        try:
            while True:
                try:
                    progress = await asyncio.wait_for(queue.get(), timeout=30.0)
                    data = {
                        "model_key": progress.model_key,
                        "status": progress.status.value,
                        "percent": progress.percent,
                        "downloaded_bytes": progress.downloaded_bytes,
                        "total_bytes": progress.total_bytes,
                        "speed_bytes_per_sec": progress.speed_bytes_per_sec,
                        "estimated_remaining_seconds": progress.estimated_remaining_seconds,
                        "error": progress.error,
                        "current_file": progress.current_file,
                        "files_downloaded": progress.files_downloaded,
                        "files_total": progress.files_total,
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # 心跳保活
                    yield ": heartbeat\n\n"
        finally:
            download_queue.remove_global_progress_callback(callback)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/delete")
async def delete_model(model_key: str = Form(...)):
    """删除已下载的模型"""
    # 检查是否正在下载
    running = download_queue.get_running_downloads()
    if model_key in running:
        raise HTTPException(status_code=400, detail="模型正在下载中，请先取消下载")

    # 卸载模型（如果已加载）
    model_config = get_model_by_key(model_key)
    if model_config:
        try:
            model_manager.unload_model(model_key)
        except Exception:
            pass

    success = storage.delete_model(model_key)
    if success:
        return {"success": True, "model_key": model_key, "message": "模型已删除"}
    else:
        raise HTTPException(status_code=400, detail="模型删除失败或不存在")


@router.get("/storage")
async def get_storage_info():
    """获取模型存储信息"""
    info = storage.get_storage_info()
    return info


@router.post("/storage/migrate")
async def migrate_storage(new_dir: str = Form(...)):
    """
    迁移模型存储目录
    将已下载的模型文件迁移到新目录
    """
    from pathlib import Path
    new_path = Path(new_dir)

    if not new_path.parent.exists():
        raise HTTPException(status_code=400, detail="存储目录的父目录不存在")

    try:
        result = storage.change_models_dir(new_path, migrate=True)
        return result
    except Exception as e:
        logger.error(f"存储目录迁移失败: {e}")
        raise HTTPException(status_code=500, detail="存储目录迁移失败")


@router.post("/load")
async def load_model(model_key: str = Form(...)):
    """加载模型（通用接口）"""
    try:
        result = model_manager.load_model(model_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/unload")
async def unload_model(model_key: str = Form(...)):
    """卸载模型（通用接口）"""
    try:
        result = model_manager.unload_model(model_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
