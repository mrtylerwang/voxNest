"""
音频格式转换
使用 ffmpeg 将各种音频/视频格式转换为 WAV（16kHz 单声道）
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from ..utils.logger import logger


class AudioConverter:
    """音频格式转换器"""

    # 支持的输入格式
    SUPPORTED_INPUT_FORMATS = {
        ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac",
        ".wma", ".opus", ".webm", ".mkv", ".avi", ".mov",
    }

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self._ffmpeg_path = ffmpeg_path or self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """查找 ffmpeg 可执行文件（优先级：内置 > 环境变量 > PATH > Homebrew）"""
        # 1. 应用内置 ffmpeg（打包后随应用分发）
        #    开发环境：项目根 bin/ 目录
        #    打包环境：应用资源目录 bin/
        candidate_paths = []
        # 环境变量指定的项目根目录（开发模式）
        if os.environ.get("VOXNEST_PROJECT_ROOT"):
            candidate_paths.append(Path(os.environ["VOXNEST_PROJECT_ROOT"]) / "bin" / "ffmpeg")
        # 从当前文件位置推导 tauri-app 根目录
        try:
            # backend/app/audio/converter.py -> 上溯3层到 tauri-app/
            project_root = Path(__file__).resolve().parents[3]
            candidate_paths.append(project_root / "bin" / "ffmpeg")
        except Exception:
            pass
        # 环境变量指定的资源目录
        if os.environ.get("VOXNEST_RESOURCE_DIR"):
            candidate_paths.append(Path(os.environ["VOXNEST_RESOURCE_DIR"]) / "bin" / "ffmpeg")
        # 当前工作目录下的 bin
        candidate_paths.append(Path.cwd() / "bin" / "ffmpeg")

        for path in candidate_paths:
            if path.exists() and os.access(path, os.X_OK):
                logger.info(f"使用内置 ffmpeg: {path}")
                return str(path)

        # 2. 环境变量
        if os.environ.get("FFMPEG_PATH"):
            return os.environ["FFMPEG_PATH"]

        # 3. PATH 中查找
        try:
            result = subprocess.run(
                ["which", "ffmpeg"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        # 4. Homebrew 常见路径（macOS，开发环境备用）
        homebrew_paths = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
        for path in homebrew_paths:
            if os.path.exists(path):
                return path

        logger.warning("未找到 ffmpeg，音频格式转换可能失败")
        return "ffmpeg"  # 回退到系统命令

    def is_supported(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_INPUT_FORMATS

    def convert_to_wav(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> str:
        """
        将音频/视频文件转换为 WAV 格式
        返回输出文件路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        # 如果没有指定输出路径，创建临时文件
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        # ffmpeg 转换命令
        # -i: 输入文件
        # -ar: 采样率
        # -ac: 声道数
        # -acodec pcm_s16le: 16位 PCM 编码
        # -y: 覆盖输出文件
        # -vn: 不处理视频流（对于视频文件只提取音频）
        cmd = [
            self._ffmpeg_path,
            "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-acodec", "pcm_s16le",
            "-vn",
            "-y",
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "未知错误"
                raise RuntimeError(f"ffmpeg 转换失败: {error_msg}")

            if not os.path.exists(output_path):
                raise RuntimeError("转换后文件不存在")

            logger.info(
                f"音频转换成功: {input_path} -> {output_path} "
                f"({sample_rate}Hz, {channels}ch)"
            )
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg 转换超时（超过120秒）")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg 未找到，请安装 ffmpeg。"
                "macOS: brew install ffmpeg"
            )

    def _find_ffprobe(self) -> str:
        """查找 ffprobe 可执行文件（与 ffmpeg 同目录优先）"""
        # 1. 与 ffmpeg 同目录
        ffmpeg_dir = Path(self._ffmpeg_path).parent
        ffprobe_path = ffmpeg_dir / "ffprobe"
        if ffprobe_path.exists() and os.access(ffprobe_path, os.X_OK):
            return str(ffprobe_path)

        # 2. 应用内置 bin 目录
        try:
            project_root = Path(__file__).resolve().parents[3]
            builtin = project_root / "bin" / "ffprobe"
            if builtin.exists() and os.access(builtin, os.X_OK):
                return str(builtin)
        except Exception:
            pass

        # 3. PATH 中查找
        try:
            result = subprocess.run(
                ["which", "ffprobe"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        return "ffprobe"  # 回退

    def get_audio_duration(self, file_path: str) -> float:
        """获取音频时长（秒）"""
        ffprobe_path = self._find_ffprobe()
        try:
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    def cleanup_temp_file(self, file_path: str) -> None:
        """清理临时文件"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:
            pass


# 全局转换器实例
converter = AudioConverter()
