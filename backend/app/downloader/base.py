"""
下载器基类
定义统一的下载接口，所有下载器必须继承此基类
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import time


class DownloadStatus(str, Enum):
    """下载状态"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    """下载进度"""
    model_key: str
    status: DownloadStatus = DownloadStatus.PENDING
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bytes_per_sec: float = 0.0
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    error: str = ""
    current_file: str = ""
    files_total: int = 0
    files_downloaded: int = 0


class BaseDownloader(ABC):
    """下载器基类"""

    def __init__(self, model_key: str, model_id: str, target_dir: str):
        self.model_key = model_key
        self.model_id = model_id
        self.target_dir = target_dir
        self._cancelled = False
        self._progress = DownloadProgress(model_key=model_key)
        self._start_time: Optional[float] = None
        self._progress_callback: Optional[Callable[[DownloadProgress], None]] = None

    def set_progress_callback(self, callback: Callable[[DownloadProgress], None]) -> None:
        """设置进度回调"""
        self._progress_callback = callback

    def cancel(self) -> None:
        """取消下载"""
        self._cancelled = True
        self._progress.status = DownloadStatus.CANCELLED
        self._notify_progress()
        self._cleanup()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def get_progress(self) -> DownloadProgress:
        return self._progress

    def _notify_progress(self) -> None:
        """通知进度更新"""
        if self._progress_callback:
            try:
                self._progress_callback(self._progress)
            except Exception:
                pass

    def _update_progress(
        self,
        downloaded_bytes: int = None,
        total_bytes: int = None,
        current_file: str = None,
        files_downloaded: int = None,
        files_total: int = None,
    ) -> None:
        """更新进度"""
        if self._cancelled:
            return

        if downloaded_bytes is not None:
            self._progress.downloaded_bytes = downloaded_bytes
        if total_bytes is not None:
            self._progress.total_bytes = total_bytes
        if current_file is not None:
            self._progress.current_file = current_file
        if files_downloaded is not None:
            self._progress.files_downloaded = files_downloaded
        if files_total is not None:
            self._progress.files_total = files_total

        # 计算百分比
        if self._progress.total_bytes > 0:
            self._progress.percent = (
                self._progress.downloaded_bytes / self._progress.total_bytes * 100
            )

        # 计算速度和剩余时间
        if self._start_time:
            elapsed = time.time() - self._start_time
            self._progress.elapsed_seconds = elapsed
            if elapsed > 0 and self._progress.downloaded_bytes > 0:
                self._progress.speed_bytes_per_sec = (
                    self._progress.downloaded_bytes / elapsed
                )
                if self._progress.speed_bytes_per_sec > 0:
                    remaining_bytes = (
                        self._progress.total_bytes - self._progress.downloaded_bytes
                    )
                    self._progress.estimated_remaining_seconds = (
                        remaining_bytes / self._progress.speed_bytes_per_sec
                    )

        self._notify_progress()

    def _start_download(self) -> None:
        """开始下载（内部调用）"""
        self._cancelled = False
        self._start_time = time.time()
        self._progress.status = DownloadStatus.DOWNLOADING
        self._notify_progress()

    def _complete_download(self) -> None:
        """完成下载（内部调用）"""
        self._progress.status = DownloadStatus.COMPLETED
        self._progress.percent = 100.0
        if self._start_time:
            self._progress.elapsed_seconds = time.time() - self._start_time
        self._notify_progress()

    def _fail_download(self, error: str) -> None:
        """下载失败（内部调用）"""
        self._progress.status = DownloadStatus.FAILED
        self._progress.error = error
        self._notify_progress()
        self._cleanup()

    def _cleanup(self) -> None:
        """清理临时文件（子类可重写）"""
        pass

    @abstractmethod
    def download(self) -> bool:
        """
        执行下载
        返回 True 表示成功，False 表示失败
        """
        pass
