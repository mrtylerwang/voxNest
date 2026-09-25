"""
下载队列
管理多个下载任务的排队、并发控制、状态查询
"""
import threading
from typing import Dict, List, Optional, Callable
from .task import DownloadTask
from .base import DownloadProgress, DownloadStatus
from ..config import config
from ..models.catalog import get_model_by_key
from ..utils.logger import logger


class DownloadQueue:
    """下载队列管理器（单例）"""

    _instance: Optional["DownloadQueue"] = None

    def __new__(cls) -> "DownloadQueue":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """初始化"""
        self._tasks: Dict[str, DownloadTask] = {}  # model_key -> task
        self._lock = threading.Lock()
        self._global_progress_callbacks: List[Callable[[DownloadProgress], None]] = []
        # 每个下载源最多同时跑的任务数（超出排队）
        self._max_per_source = 2
        # 等待队列：[{model_key, on_progress, download_source, use_hf_mirror, source}]
        self._pending: List[dict] = []

    @staticmethod
    def _resolve_source(req_source, model_config):
        """根据请求指定与模型可用源，决定本次走哪个下载源。"""
        from ..config import DownloadSource
        rs = (req_source or config.download_source.value).lower()
        if rs == DownloadSource.MODELSCOPE.value and getattr(model_config, "modelscope_id", None):
            return DownloadSource.MODELSCOPE
        return DownloadSource.HUGGINGFACE

    def _source_running(self, source) -> int:
        """统计指定源当前正在运行的任务数。调用方需持锁。"""
        return sum(
            1 for t in self._tasks.values()
            if t.is_running and getattr(t, "source", None) == source
        )

    def _spawn(self, model_key, on_progress, req_source, req_mirror) -> DownloadTask:
        """创建并启动一个下载任务。调用方需持锁。"""
        task = DownloadTask(
            model_key,
            on_progress=lambda p: self._handle_progress(model_key, p, on_progress),
            on_complete=lambda key, success: self._handle_complete(key, success),
            download_source=req_source,
            use_hf_mirror=req_mirror,
        )
        task.start()
        self._tasks[model_key] = task
        return task

    def _pump_pending(self, source) -> None:
        """某源有槽位空出时，从等待队列取同源任务启动。调用方需持锁。"""
        while True:
            idx = next(
                (i for i, pend in enumerate(self._pending) if pend["source"] == source),
                None,
            )
            if idx is None:
                return
            if self._source_running(source) >= self._max_per_source:
                return
            item = self._pending.pop(idx)
            self._tasks.pop(item["model_key"], None)
            self._spawn(
                item["model_key"], item["on_progress"],
                item["download_source"], item["use_hf_mirror"],
            )
            logger.info(f"排队任务已启动: {item['model_key']}, source={source.value}")

    def add_global_progress_callback(
        self, callback: Callable[[DownloadProgress], None]
    ) -> None:
        """添加全局进度回调（用于 SSE 广播）"""
        self._global_progress_callbacks.append(callback)

    def remove_global_progress_callback(
        self, callback: Callable[[DownloadProgress], None]
    ) -> None:
        """移除全局进度回调"""
        if callback in self._global_progress_callbacks:
            self._global_progress_callbacks.remove(callback)

    def start_download(
        self,
        model_key: str,
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
        bypass_concurrency_limit: bool = False,
        download_source: Optional[str] = None,
        use_hf_mirror: Optional[bool] = None,
    ) -> dict:
        """
        开始下载模型
        返回任务信息
        """
        with self._lock:
            # 检查是否已在下载
            if model_key in self._tasks:
                task = self._tasks[model_key]
                if task.is_running and not task.is_cancelled:
                    return {
                        "success": True,
                        "model_key": model_key,
                        "job_id": task.job_id,
                        "message": "模型已在下载中",
                        "progress": self._progress_to_dict(task.progress),
                    }
                # 任务已取消/失败/完成，移除旧任务，允许重新下载
                if not task.is_running or task.is_cancelled:
                    logger.info(f"移除旧任务以重新下载: {model_key}, status={task.progress.status}, cancelled={task.is_cancelled}")
                    del self._tasks[model_key]

            # 确定本次走哪个下载源（按源分别限流，每个源最多 2 个并发）
            model_config = get_model_by_key(model_key)
            resolved_source = self._resolve_source(download_source, model_config)

            if bypass_concurrency_limit or self._source_running(resolved_source) < self._max_per_source:
                # 立即启动
                task = self._spawn(model_key, on_progress, download_source, use_hf_mirror)
                started = task.progress.status != DownloadStatus.FAILED
                return {
                    "success": True,
                    "model_key": model_key,
                    "job_id": task.job_id,
                    "message": "下载已开始" if started else "下载启动失败",
                    "progress": self._progress_to_dict(task.progress),
                }

            # 该源并发已满，入等待队列
            pending_task = DownloadTask(
                model_key,
                on_progress=lambda p: self._handle_progress(model_key, p, on_progress),
                on_complete=lambda key, success: self._handle_complete(key, success),
                download_source=download_source,
                use_hf_mirror=use_hf_mirror,
            )
            self._tasks[model_key] = pending_task
            self._pending.append({
                "model_key": model_key,
                "on_progress": on_progress,
                "download_source": download_source,
                "use_hf_mirror": use_hf_mirror,
                "source": resolved_source,
            })
            logger.info(
                f"下载任务排队: {model_key}, source={resolved_source.value}, "
                f"running={self._source_running(resolved_source)}/{self._max_per_source}"
            )
            return {
                "success": True,
                "model_key": model_key,
                "job_id": pending_task.job_id,
                "message": "排队中，等待下载槽位",
                "progress": self._progress_to_dict(pending_task.progress),
            }

    def wait_for_download(
        self,
        model_key: str,
        poll_interval: float = 0.5,
    ) -> Optional[dict]:
        """等待指定下载任务结束并返回最终进度"""
        import time

        while True:
            with self._lock:
                task = self._tasks.get(model_key)
                if task is None:
                    return None
                progress = self._progress_to_dict(task.progress)
                if not task.is_running:
                    return progress
            time.sleep(poll_interval)

    def cancel_download(self, model_key: str) -> dict:
        """取消下载"""
        with self._lock:
            task = self._tasks.get(model_key)
            if task is None:
                return {
                    "success": False,
                    "model_key": model_key,
                    "message": "未找到下载任务",
                }

            if not task.is_running:
                # 任务未运行（可能是排队中），从任务表与等待队列移除
                del self._tasks[model_key]
                self._pending = [q for q in self._pending if q["model_key"] != model_key]
                return {
                    "success": True,
                    "model_key": model_key,
                    "message": "下载任务已移除",
                }

            success = task.cancel()
            # 取消后立即从任务列表移除，允许重新下载
            # 下载线程会在后台自行退出（daemon=True）
            if success:
                del self._tasks[model_key]
                # 取消后可能有槽位空出，尝试启动排队任务
                if task.source:
                    self._pump_pending(task.source)
            return {
                "success": success,
                "model_key": model_key,
                "message": "下载已取消" if success else "取消失败",
            }

    def get_download_progress(self, model_key: str) -> Optional[dict]:
        """获取指定模型的下载进度"""
        with self._lock:
            task = self._tasks.get(model_key)
            if task:
                return self._progress_to_dict(task.progress)
        return None

    def get_all_downloads(self) -> List[dict]:
        """获取所有下载任务状态"""
        result = []
        with self._lock:
            for model_key, task in self._tasks.items():
                result.append({
                    "model_key": model_key,
                    "job_id": task.job_id,
                    "is_running": task.is_running,
                    "source": task.source.value if task.source else None,
                    "use_hf_mirror": getattr(task, "_req_mirror", None),
                    "progress": self._progress_to_dict(task.progress),
                })
        return result

    def get_running_downloads(self) -> List[str]:
        """获取正在下载的模型 key 列表"""
        return [key for key, task in self._tasks.items() if task.is_running]

    def cleanup_completed(self) -> None:
        """清理已完成/失败/取消的任务"""
        with self._lock:
            keys_to_remove = []
            for key, task in self._tasks.items():
                if not task.is_running:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._tasks[key]

    def _handle_progress(
        self,
        model_key: str,
        progress: DownloadProgress,
        user_callback: Optional[Callable[[DownloadProgress], None]],
    ) -> None:
        """处理进度更新（转发到全局回调和用户回调）"""
        # 全局回调（SSE 广播）
        for callback in self._global_progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"全局进度回调异常: {e}")

        # 用户回调
        if user_callback:
            try:
                user_callback(progress)
            except Exception as e:
                logger.error(f"用户进度回调异常: {e}")

    def _handle_complete(self, model_key: str, success: bool) -> None:
        """处理下载完成"""
        logger.info(
            f"下载任务完成: {model_key}, success={success}"
        )
        # 任务结束，槽位空出，尝试启动该源的排队任务
        task = self._tasks.get(model_key)
        src = task.source if task else None
        if src:
            with self._lock:
                self._pump_pending(src)
        # 清理任务（延迟清理，让前端能获取最终状态）
        # 不立即清理，由 cleanup_completed 或新下载时清理

    def _progress_to_dict(self, progress: DownloadProgress) -> dict:
        """将进度对象转为字典"""
        return {
            "model_key": progress.model_key,
            "status": progress.status.value,
            "total_bytes": progress.total_bytes,
            "downloaded_bytes": progress.downloaded_bytes,
            "speed_bytes_per_sec": progress.speed_bytes_per_sec,
            "percent": progress.percent,
            "elapsed_seconds": progress.elapsed_seconds,
            "estimated_remaining_seconds": progress.estimated_remaining_seconds,
            "error": progress.error,
            "current_file": progress.current_file,
            "files_total": progress.files_total,
            "files_downloaded": progress.files_downloaded,
        }


# 全局下载队列实例
download_queue = DownloadQueue()
