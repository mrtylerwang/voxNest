"""
Qwen3-ASR 引擎
使用 mlx-audio 库运行 Qwen3-ASR-1.7B 模型
支持 52 种语言和方言、自动语言检测、长音频分块转录、时间戳
官方文档: https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B
MLX 版本: mlx-community/Qwen3-ASR-1.7B-8bit
"""
import time
from pathlib import Path
from typing import Optional, List

from .base import BaseASREngine, EngineType, ASRResult, ASRSegment, ModelCapability
from ..audio.preprocess import load_audio, get_audio_duration
from ..utils.logger import logger


class Qwen3ASREngine(BaseASREngine):
    """Qwen3-ASR 引擎（mlx-audio 实现）"""

    def __init__(self, model_key: str, engine_type: EngineType):
        super().__init__(model_key, engine_type)
        self._model = None
        self._model_path = None

    def load(self, model_path: str) -> None:
        """加载 Qwen3-ASR 模型"""
        if self._loaded:
            logger.info(f"模型已加载，跳过: {self.model_key}")
            return

        logger.info(f"正在加载 Qwen3-ASR 模型: {model_path}")

        try:
            from mlx_audio.stt import load as load_stt_model
        except ImportError:
            raise ImportError(
                "mlx-audio 未安装。请安装: pip install mlx-audio\n"
                "macOS Apple Silicon 推荐使用 MLX 加速"
            )

        # mlx-audio stt.load 支持本地路径，自动检测模型类型
        self._model = load_stt_model(model_path)
        self._model_path = model_path
        self._loaded = True

        logger.info(f"Qwen3-ASR 模型加载成功: {model_path}")

    def unload(self) -> None:
        """卸载模型，释放内存"""
        if self._model is not None:
            del self._model
            self._model = None

        # 强制清理 MLX 缓存
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

        self._loaded = False
        self._model_path = None
        logger.info(f"Qwen3-ASR 模型已卸载: {self.model_key}")

    def transcribe(self, audio_path: str) -> ASRResult:
        """
        转录音频
        Qwen3-ASR 支持时间戳输出（segments 中包含 start/end）
        language=None 时自动检测语言
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        start_time = time.time()
        logger.info(f"开始转录: {audio_path}")

        try:
            # mlx-audio Qwen3-ASR generate 方法
            # language=None 自动检测；chunk_duration 默认 1200 秒（20分钟）
            result = self._model.generate(
                audio_path,
                language=None,  # 自动语言检测
                temperature=0.0,  # ASR 用贪心解码
                verbose=False,
            )

            # 提取 segments（Qwen3-ASR 输出带时间戳的 segment 列表）
            segments = self._extract_segments(result)

            # 合并全文
            text = "".join(seg.text for seg in segments).strip()

            # 提取检测到的语言
            language = self._extract_language(result)

            # 生成 SRT
            srt = self._generate_srt(segments)

            processing_time = time.time() - start_time
            logger.info(
                f"转录完成: {len(text)} 字, {len(segments)} 段, "
                f"耗时 {processing_time:.2f}s, 语言: {language}"
            )

            return ASRResult(
                text=text,
                segments=segments,
                srt=srt,
                language=language,
                emotion="",  # Qwen3-ASR 不支持情感检测
                engine_used=f"qwen3-asr-{self.engine_type.value}",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise RuntimeError(f"Qwen3-ASR 转录失败: {e}")

    def _extract_segments(self, result) -> List[ASRSegment]:
        """从模型输出中提取带时间戳的 segments"""
        segments = []

        # Qwen3-ASR 的 STTOutput 有 segments 字段
        raw_segments = []
        if hasattr(result, "segments") and result.segments:
            raw_segments = result.segments
        elif isinstance(result, dict):
            raw_segments = result.get("segments", [])

        for seg in raw_segments:
            text = ""
            start = 0.0
            end = 0.0

            # segment 可能是对象或字典
            if hasattr(seg, "text"):
                text = seg.text or ""
            elif isinstance(seg, dict):
                text = seg.get("text", "") or ""

            if hasattr(seg, "start"):
                start = float(seg.start or 0.0)
            elif isinstance(seg, dict):
                start = float(seg.get("start", 0.0) or 0.0)

            if hasattr(seg, "end"):
                end = float(seg.end or 0.0)
            elif isinstance(seg, dict):
                end = float(seg.get("end", 0.0) or 0.0)

            if text.strip():
                segments.append(ASRSegment(text=text.strip(), start=start, end=end))

        # 如果没有 segments，尝试从 text 字段提取（整段作为一个 segment）
        if not segments:
            text = ""
            if hasattr(result, "text"):
                text = result.text or ""
            elif isinstance(result, dict):
                text = result.get("text", "") or ""

            if text.strip():
                segments.append(ASRSegment(text=text.strip(), start=0.0, end=0.0))

        return segments

    def _extract_language(self, result) -> str:
        """从模型输出中提取检测到的语言"""
        lang = ""
        if hasattr(result, "language"):
            lang = result.language
        elif isinstance(result, dict):
            lang = result.get("language", "") or result.get("lang", "") or ""

        # language 可能是列表（如 ['Chinese']），取第一个
        if isinstance(lang, list):
            lang = lang[0] if lang else ""

        return str(lang) if lang else ""

    def _generate_srt(self, segments: List[ASRSegment]) -> str:
        """生成 SRT 格式（只写时间和内容，不写序号）"""
        srt_lines = []
        for seg in segments:
            start = self._format_srt_time(seg.start)
            end = self._format_srt_time(seg.end)
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(seg.text)
            srt_lines.append("")
        return "\n".join(srt_lines).strip()

    def _format_srt_time(self, seconds: float) -> str:
        """格式化 SRT 时间戳"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def get_capability(self) -> ModelCapability:
        """获取模型能力"""
        return ModelCapability(
            supports_streaming=False,
            supports_cloning=False,
            supports_voice_design=False,
            supports_emotion=False,  # Qwen3-ASR 不支持情感检测
            supports_timestamps=True,  # 支持时间戳
            supported_languages=[
                "zh", "en", "yue", "ja", "ko", "de", "fr", "es", "pt",
                "ru", "it", "ar", "id", "th", "vi", "tr", "hi", "ms",
                "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el",
                "hu", "mk", "ro", "auto",
            ],
            min_memory_mb=2048,
            param_size="1.7B",
        )
