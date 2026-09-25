"""
HuggingFace 下载器
使用 huggingface_hub 官方 SDK 下载模型，支持 HF-Mirror 代理加速
HF-Mirror 不是独立下载源，而是 HuggingFace 的镜像代理
"""
import os
import threading
import time
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadProgress, DownloadStatus
from ..utils.logger import logger


class HuggingFaceDownloader(BaseDownloader):
    """HuggingFace 模型下载器（支持 HF-Mirror 代理）"""

    def __init__(
        self,
        model_key: str,
        model_id: str,
        target_dir: str,
        use_mirror: bool = False,
    ):
        super().__init__(model_key, model_id, target_dir)
        self.use_mirror = use_mirror
        self._mirror_url = "https://hf-mirror.com"
        self._poller_stop = threading.Event()

    def download(self) -> bool:
        """使用 huggingface_hub.snapshot_download 下载模型"""
        # 注意：endpoint 必须显式传参。huggingface_hub 的 constants.ENDPOINT 在
        # import 时固化，运行时改 HF_ENDPOINT 环境变量对已加载模块无效；
        # 而 Tauri 启动的后端不继承 shell 环境（~/.zshrc 的 HF_ENDPOINT），
        # 依赖环境变量会导致 snapshot_download 始终直连 huggingface.co（国内超时）。
        endpoint = self._mirror_url if self.use_mirror else None

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self._fail_download("huggingface_hub 未安装，请先安装依赖")
            return False

        self._start_download()

        if self.use_mirror:
            logger.info(
                f"开始从 HuggingFace (HF-Mirror) 下载: {self.model_id} -> {self.target_dir}"
            )
        else:
            logger.info(
                f"开始从 HuggingFace 下载: {self.model_id} -> {self.target_dir}"
            )

        try:
            # 创建目标目录
            target_path = Path(self.target_dir)
            target_path.mkdir(parents=True, exist_ok=True)

            # 启动前先从 HF 仓库元数据获取真实总大小，用于计算百分比
            total_size = self._fetch_repo_size(endpoint)
            if total_size and total_size > 0:
                self._update_progress(total_bytes=total_size, current_file="fetching...")
                logger.info(f"HF 仓库总大小: {total_size / 1024 / 1024:.1f}MB ({self.model_id})")

            # 启动进度轮询线程
            poller = threading.Thread(target=self._poll_progress, daemon=True)
            poller.start()

            # 使用 snapshot_download 下载到本地目录（显式指定 endpoint，避免依赖 import 时机）
            downloaded_path = snapshot_download(
                repo_id=self.model_id,
                local_dir=str(target_path),
                endpoint=endpoint,
            )

            # 停止进度轮询
            self._poller_stop.set()
            poller.join(timeout=2)

            if self._cancelled:
                return False

            # 更新进度为完成
            total_size = self._calculate_dir_size(target_path)
            self._update_progress(
                downloaded_bytes=total_size,
                total_bytes=total_size,
            )
            self._complete_download()
            logger.info(f"HuggingFace 下载完成: {self.model_id} -> {self.target_dir} ({total_size / 1024 / 1024:.1f}MB)")
            return True

        except Exception as e:
            self._poller_stop.set()
            error_msg = str(e)
            logger.error(f"HuggingFace 下载失败: {self.model_id}, error: {error_msg}")

            # 常见错误处理
            if "404" in error_msg or "Repository Not Found" in error_msg:
                error_msg = f"模型不存在: {self.model_id}"
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                if self.use_mirror:
                    error_msg = "HF-Mirror 下载超时，请检查网络或关闭镜像使用直连"
                else:
                    error_msg = "HuggingFace 下载超时，国内用户建议启用 HF-Mirror 加速"
            elif "401" in error_msg or "gated" in error_msg.lower():
                error_msg = "模型需要授权访问，请检查模型是否为公开模型"
            elif "EntryNotFound" in error_msg:
                error_msg = f"模型文件不存在: {self.model_id}"

            self._fail_download(error_msg)
            return False

        finally:
            # 下载失败时清理 .incomplete 临时文件，避免误判为已下载
            if not self._cancelled and self._progress.status != DownloadStatus.COMPLETED:
                self._cleanup()

    def _fetch_repo_size(self, endpoint: Optional[str] = None) -> int:
        """通过 HfApi.model_info 获取仓库文件总大小（字节）。失败返回 0。"""
        try:
            from huggingface_hub import HfApi
            api = HfApi(endpoint=endpoint)
            info = api.model_info(self.model_id, files_metadata=True)
            total = 0
            for s in (info.siblings or []):
                sz = getattr(s, "size", None)
                if sz:
                    total += int(sz)
            return total
        except Exception as e:
            logger.warning(f"获取 HF 仓库元数据失败，进度将按未知总量显示: {e}")
            return 0

    def _poll_progress(self) -> None:
        """轮询目标目录大小，更新下载进度。total_bytes 保留启动时获取的真实总大小。"""
        while not self._poller_stop.is_set() and not self._cancelled:
            try:
                target_path = Path(self.target_dir)
                if target_path.exists():
                    # 统计所有文件（包括 .cache/ 中的临时文件 .incomplete/.lock/.tmp）
                    current_size = self._calculate_dir_size(target_path)
                    # 不传 total_bytes，保留启动时 _fetch_repo_size 设置的真实总量
                    self._update_progress(
                        downloaded_bytes=current_size,
                        current_file="downloading...",
                    )
            except Exception:
                pass
            time.sleep(1.0)

    def _calculate_dir_size(self, dir_path: Path) -> int:
        """计算目录总大小"""
        total = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                try:
                    total += (Path(root) / file).stat().st_size
                except OSError:
                    pass
        return total

    def _cleanup(self) -> None:
        """清理临时文件"""
        self._poller_stop.set()
        try:
            target_path = Path(self.target_dir)
            if target_path.exists():
                for f in target_path.glob("*.incomplete"):
                    f.unlink(missing_ok=True)
        except Exception:
            pass
