"""
下载任务管理
管理单个下载任务的生命周期，支持进度回调、取消、重试
"""
import threading
import time
from typing import Optional, Callable, List
from .base import BaseDownloader, DownloadProgress, DownloadStatus
from .modelscope import ModelScopeDownloader
from .huggingface import HuggingFaceDownloader
from ..config import config, DownloadSource
from ..models.catalog import get_model_by_key
from ..models.storage import storage
from ..utils.logger import logger


class DownloadTask:
    """下载任务"""

    def __init__(
        self,
        model_key: str,
        on_progress: Optional[Callable[[DownloadProgress], None]] = None,
        on_complete: Optional[Callable[[str, bool], None]] = None,
        download_source: Optional[str] = None,
        use_hf_mirror: Optional[bool] = None,
    ):
        self.model_key = model_key
        self.job_id = f"download_{model_key}_{int(time.time() * 1000)}"
        self._downloader: Optional[BaseDownloader] = None
        self._thread: Optional[threading.Thread] = None
        self._on_progress = on_progress
        self._on_complete = on_complete
        # 本次下载显式指定的源；None 表示回退全局 config
        self._req_source = download_source
        self._req_mirror = use_hf_mirror
        self._cancelled = False
        self._progress = DownloadProgress(model_key=model_key)
        # 本次任务实际使用的下载源（start() 时确定），供 queue 按源限流
        self.source: Optional[DownloadSource] = None

    @property
    def progress(self) -> DownloadProgress:
        return self._progress

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def start(self) -> bool:
        """开始下载任务"""
        if self.is_running:
            logger.warning(f"下载任务已在运行: {self.model_key}")
            return False

        model_config = get_model_by_key(self.model_key)
        if model_config is None:
            logger.error(f"未知模型: {self.model_key}")
            return False

        # 检查是否已下载
        if storage.is_model_downloaded(self.model_key):
            logger.info(f"模型已下载，跳过: {self.model_key}")
            self._progress.status = DownloadStatus.COMPLETED
            self._progress.percent = 100.0
            if self._on_complete:
                self._on_complete(self.model_key, True)
            return True

        # 创建下载器
        target_dir = str(storage.get_model_dir(self.model_key))
        # 优先用本次下载请求显式指定的源，缺省回退全局 config
        req_source = (self._req_source or config.download_source.value).lower()
        req_mirror = config.use_hf_mirror if self._req_mirror is None else bool(self._req_mirror)

        if req_source == DownloadSource.MODELSCOPE.value and model_config.modelscope_id:
            download_source = DownloadSource.MODELSCOPE
            model_id = model_config.modelscope_id
            self._downloader = ModelScopeDownloader(
                self.model_key, model_id, target_dir
            )
        else:
            download_source = DownloadSource.HUGGINGFACE
            # ModelScope 不支持或用户选择 HuggingFace 时，使用 HuggingFace
            model_id = model_config.huggingface_id
            if not model_id:
                self._fail("该模型不支持任何下载源")
                return False
            self._downloader = HuggingFaceDownloader(
                self.model_key, model_id, target_dir,
                use_mirror=req_mirror,
            )

        # 设置进度回调
        self._downloader.set_progress_callback(self._handle_progress)
        self.source = download_source

        # 启动下载线程
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        logger.info(
            f"下载任务已启动: {self.model_key}, "
            f"source={download_source.value}, "
            f"hf_mirror={req_mirror if download_source == DownloadSource.HUGGINGFACE else 'N/A'}"
        )
        return True

    def _run(self) -> None:
        """下载线程主函数"""
        try:
            success = self._downloader.download()
            if self._cancelled:
                self._progress.status = DownloadStatus.CANCELLED
                if self._on_complete:
                    self._on_complete(self.model_key, False)
                return

            if success:
                self._progress = self._downloader.get_progress()
                # 下载依赖模型
                dependencies_ok = self._download_dependencies()
                if dependencies_ok:
                    if self._on_complete:
                        self._on_complete(self.model_key, True)
                else:
                    self._fail("依赖模型下载失败")
            else:
                self._progress = self._downloader.get_progress()
                if self._on_complete:
                    self._on_complete(self.model_key, False)
        except Exception as e:
            logger.error(f"下载任务异常: {self.model_key}, error: {e}")
            self._fail(str(e))

    def _download_dependencies(self) -> bool:
        """下载依赖的辅助模型"""
        model_config = get_model_by_key(self.model_key)
        if not model_config:
            return True

        for dep in model_config.dependencies:
            if dep.required and not storage.is_model_downloaded(dep.key):
                logger.info(f"自动下载依赖模型: {dep.key} ({dep.name})")
                from .queue import download_queue

                result = download_queue.start_download(
                    dep.key,
                    bypass_concurrency_limit=True,
                    download_source=self._req_source,
                    use_hf_mirror=self._req_mirror,
                )
                if not result.get("success"):
                    logger.error(
                        f"依赖模型下载启动失败: {dep.key}, message={result.get('message', '')}"
                    )
                    return False

                final_progress = download_queue.wait_for_download(dep.key)
                if not final_progress:
                    logger.error(f"依赖模型下载状态丢失: {dep.key}")
                    return False

                if final_progress["status"] != DownloadStatus.COMPLETED.value:
                    logger.error(
                        f"依赖模型下载失败: {dep.key}, status={final_progress['status']}, "
                        f"error={final_progress.get('error', '')}"
                    )
                    return False

        return True

    def cancel(self) -> bool:
        """取消下载任务"""
        if not self.is_running:
            return False

        self._cancelled = True
        if self._downloader:
            self._downloader.cancel()

        logger.info(f"下载任务已取消: {self.model_key}")
        return True

    def _handle_progress(self, progress: DownloadProgress) -> None:
        """处理进度更新"""
        self._progress = progress
        if self._on_progress:
            try:
                self._on_progress(progress)
            except Exception as e:
                logger.error(f"进度回调异常: {e}")

    def _fail(self, error: str) -> None:
        """标记任务失败"""
        self._progress.status = DownloadStatus.FAILED
        self._progress.error = error
        if self._on_complete:
            self._on_complete(self.model_key, False)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待下载完成"""
        if self._thread:
            self._thread.join(timeout=timeout)
        return self._progress.status == DownloadStatus.COMPLETED
