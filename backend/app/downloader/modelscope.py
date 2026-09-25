"""
ModelScope 下载器
使用 modelscope 命令行工具（子进程）下载模型
经测试：Python API snapshot_download 大文件下载不稳定，命令行工具稳定
"""
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
from .base import BaseDownloader, DownloadStatus
from ..models.catalog import get_model_by_key
from ..utils.logger import logger


class ModelScopeDownloader(BaseDownloader):
    """ModelScope 模型下载器（命令行工具）"""

    def __init__(self, model_key: str, model_id: str, target_dir: str):
        super().__init__(model_key, model_id, target_dir)
        self._poller_stop = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._last_output_time = time.time()
        model_config = get_model_by_key(model_key)
        estimated_size_mb = getattr(model_config, "estimated_size_mb", 0) if model_config else 0
        self._expected_total_bytes = int(estimated_size_mb * 1024 * 1024) if estimated_size_mb else 0

    def _find_modelscope_bin(self) -> Optional[str]:
        """查找 modelscope 命令行工具路径"""
        import sys
        # 从当前 Python 解释器路径推导（打包后用 venv 中的 Python 启动）
        python_dir = Path(sys.executable).parent
        modelscope_bin = python_dir / "modelscope"
        if modelscope_bin.exists():
            return str(modelscope_bin)

        # 尝试从 PATH 查找
        from shutil import which
        result = which("modelscope")
        if result:
            return result

        return None

    def download(self) -> bool:
        """使用 modelscope 命令行工具下载模型"""
        modelscope_bin = self._find_modelscope_bin()

        if modelscope_bin is None:
            # 回退到 Python API
            logger.warning("未找到 modelscope 命令行工具，回退到 Python API")
            return self._download_python_api()

        self._start_download()
        logger.info(f"开始从 ModelScope(CLI) 下载: {self.model_id} -> {self.target_dir}")
        if self._expected_total_bytes > 0:
            self._update_progress(total_bytes=self._expected_total_bytes)

        try:
            target_path = Path(self.target_dir)
            target_path.mkdir(parents=True, exist_ok=True)

            # 设置大文件并行下载环境变量
            env = os.environ.copy()
            env.setdefault('MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS', '4')
            env.setdefault('MODELSCOPE_DOWNLOAD_PARALLEL_THRESHOLD_MB', '100')

            # 启动进度轮询线程
            poller = threading.Thread(target=self._poll_progress, daemon=True)
            poller.start()

            # 构建命令
            cmd = [
                modelscope_bin,
                "download",
                "--model", self.model_id,
                "--local_dir", str(target_path),
            ]

            logger.info(f"执行命令: {' '.join(cmd)}")

            # 启动子进程（start_new_session 创建进程组，方便终止所有子进程）
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            # 读取输出（用于日志和调试）
            def _read_output():
                try:
                    for line in self._process.stdout:
                        line = line.strip()
                        if line and not line.startswith('\r'):
                            self._last_output_time = time.time()
                            # 只记录非进度条的行
                            if '%' not in line and 'file/s' not in line:
                                logger.debug(f"[modelscope] {line[:200]}")
                except Exception:
                    pass

            reader = threading.Thread(target=_read_output, daemon=True)
            reader.start()

            # 等待子进程完成
            return_code = self._process.wait()
            self._poller_stop.set()
            poller.join(timeout=2)

            if self._cancelled:
                return False

            if return_code != 0:
                error_msg = f"modelscope 命令行工具退出码: {return_code}"
                logger.error(f"ModelScope(CLI) 下载失败: {self.model_id}, {error_msg}")
                self._fail_download(error_msg)
                self._cleanup()
                return False

            # 下载完成后校验完整性
            if not self._verify_download(target_path):
                error_msg = "下载不完整，大文件未完成。请检查网络后重试"
                logger.error(f"ModelScope(CLI) 下载校验失败: {self.model_id}")
                self._fail_download(error_msg)
                self._cleanup()
                return False

            total_size = self._calculate_dir_size(target_path)
            self._update_progress(
                downloaded_bytes=total_size,
                total_bytes=total_size,
            )
            self._complete_download()
            logger.info(f"ModelScope(CLI) 下载完成: {self.model_id} -> {self.target_dir} ({total_size / 1024 / 1024:.1f}MB)")
            return True

        except Exception as e:
            self._poller_stop.set()
            error_msg = str(e)
            logger.error(f"ModelScope(CLI) 下载异常: {self.model_id}, error: {error_msg}")
            self._fail_download(error_msg)
            self._cleanup()
            return False

    def _download_python_api(self) -> bool:
        """回退：使用 Python API snapshot_download"""
        try:
            from modelscope import snapshot_download
        except ImportError:
            self._fail_download("modelscope 未安装，请先安装依赖")
            return False

        self._start_download()
        logger.info(f"开始从 ModelScope(Python API) 下载: {self.model_id} -> {self.target_dir}")

        try:
            target_path = Path(self.target_dir)
            target_path.mkdir(parents=True, exist_ok=True)

            poller = threading.Thread(target=self._poll_progress, daemon=True)
            poller.start()
            if self._expected_total_bytes > 0:
                self._update_progress(total_bytes=self._expected_total_bytes)

            downloaded_path = snapshot_download(
                model_id=self.model_id,
                local_dir=str(target_path),
                local_files_only=False,
                max_workers=4,
            )

            self._poller_stop.set()
            poller.join(timeout=2)

            if self._cancelled:
                return False

            if not self._verify_download(target_path):
                error_msg = "下载不完整，大文件未完成"
                self._fail_download(error_msg)
                self._cleanup()
                return False

            total_size = self._calculate_dir_size(target_path)
            self._update_progress(
                downloaded_bytes=total_size,
                total_bytes=total_size,
            )
            self._complete_download()
            logger.info(f"ModelScope(Python API) 下载完成: {self.model_id} ({total_size / 1024 / 1024:.1f}MB)")
            return True

        except Exception as e:
            self._poller_stop.set()
            error_msg = str(e)
            logger.error(f"ModelScope(Python API) 下载失败: {self.model_id}, error: {error_msg}")
            self._fail_download(error_msg)
            self._cleanup()
            return False

    def cancel(self) -> bool:
        """取消下载：终止整个进程组（包括并行下载的 worker 子进程）"""
        self._cancelled = True
        if self._process and self._process.poll() is None:
            try:
                # 终止整个进程组
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                # 进程可能已经退出
                self._process.kill()
        self._poller_stop.set()
        self._cleanup()
        return True

    def _verify_download(self, target_path: Path) -> bool:
        """校验下载完整性：无 .incomplete 文件"""
        for f in target_path.rglob("*.incomplete"):
            logger.warning(f"发现未完成的下载文件: {f}")
            return False
        for f in target_path.rglob("*.partial"):
            logger.warning(f"发现未完成的下载文件: {f}")
            return False
        return True

    def _poll_progress(self) -> None:
        """轮询目标目录大小，更新下载进度并记录长时间无变化状态"""
        last_size = 0
        last_growth_time = time.time()
        warned_stall = False
        stall_timeout = 900  # 仅做告警，不自动强杀

        while not self._poller_stop.is_set() and not self._cancelled:
            try:
                target_path = Path(self.target_dir)
                if target_path.exists():
                    current_size = self._calculate_dir_size(target_path)
                    if current_size > last_size:
                        last_size = current_size
                        last_growth_time = time.time()
                        warned_stall = False
                    else:
                        elapsed = time.time() - last_growth_time
                        output_idle = time.time() - self._last_output_time
                        if elapsed > stall_timeout and current_size > 0 and not warned_stall:
                            logger.warning(
                                f"ModelScope(CLI) 下载长时间无目录增长: {self.model_key}, "
                                f"当前大小 {current_size/1024/1024:.1f}MB, "
                                f"{elapsed:.0f}秒无增长，最近输出静默 {output_idle:.0f}秒"
                            )
                            warned_stall = True

                    self._update_progress(
                        downloaded_bytes=current_size,
                        total_bytes=max(self._expected_total_bytes, current_size),
                        current_file="downloading...",
                    )
            except Exception:
                pass
            time.sleep(1.0)

    def _calculate_dir_size(self, dir_path: Path) -> int:
        """计算目录总大小（包括 .incomplete 临时文件）"""
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
                for f in target_path.rglob("*.incomplete"):
                    f.unlink(missing_ok=True)
                for f in target_path.rglob("*.partial"):
                    f.unlink(missing_ok=True)
                cache_dir = target_path / ".cache"
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass
