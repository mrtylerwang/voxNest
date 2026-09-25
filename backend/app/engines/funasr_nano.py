"""
Fun-ASR-Nano ASR 引擎
使用 mlx-audio 库运行 Fun-ASR-Nano-2512 模型
支持中文+7种方言+歌词识别，支持热词提示
注意：此模型不输出时间戳
"""
import time
import numpy as np
from pathlib import Path
from typing import Optional, List

from .base import BaseASREngine, EngineType, ASRResult, ASRSegment, ModelCapability
from ..audio.preprocess import load_audio, get_audio_duration
from ..utils.logger import logger


class FunASRNanoEngine(BaseASREngine):
    """Fun-ASR-Nano ASR 引擎（mlx-audio 实现）"""

    def __init__(self, model_key: str, engine_type: EngineType):
        super().__init__(model_key, engine_type)
        self._model = None
        self._model_path = None
        self._language = "auto"  # 默认自动检测
        self._hotwords: List[str] = []

    def load(self, model_path: str) -> None:
        """加载 Fun-ASR-Nano 模型"""
        if self._loaded:
            logger.info(f"模型已加载，跳过: {self.model_key}")
            return

        logger.info(f"正在加载 Fun-ASR-Nano 模型: {model_path}")

        try:
            from mlx_audio.stt.utils import load_model
        except ImportError:
            raise ImportError(
                "mlx-audio 未安装。请安装: pip install mlx-audio\n"
                "macOS Apple Silicon 推荐使用 MLX 加速"
            )

        # mlx-audio 的 load_model 支持本地路径
        self._model = load_model(model_path)
        self._model_path = model_path
        self._loaded = True

        logger.info(f"Fun-ASR-Nano 模型加载成功: {model_path}")

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
        logger.info(f"Fun-ASR-Nano 模型已卸载: {self.model_key}")

    def set_language(self, language: str) -> None:
        """设置识别语言"""
        self._language = language

    def set_hotwords(self, hotwords: List[str]) -> None:
        """设置热词列表"""
        self._hotwords = hotwords or []

    def transcribe(self, audio_path: str) -> ASRResult:
        """
        转录音频
        Fun-ASR-Nano 不输出时间戳，整段作为一个 segment
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        start_time = time.time()
        logger.info(f"开始转录: {audio_path}, 语言: {self._language}")

        try:
            # 构建 generate 参数
            generate_kwargs = {}

            # 语言参数（auto 时不传，让模型自动检测）
            if self._language and self._language != "auto":
                generate_kwargs["language"] = self._language

            # 热词参数
            if self._hotwords:
                generate_kwargs["hotwords"] = self._hotwords

            # 调用模型
            result = self._model.generate(audio_path, **generate_kwargs)

            # 提取文本
            text = self._extract_text(result)

            # 获取音频时长
            try:
                audio, sample_rate = load_audio(audio_path, target_sample_rate=16000)
                duration = get_audio_duration(audio, sample_rate)
            except Exception:
                duration = 0.0

            # Fun-ASR-Nano 不输出时间戳，整段作为一个 segment
            segments = [ASRSegment(
                text=text,
                start=0.0,
                end=duration,
            )]

            # 生成 SRT
            srt = self._generate_srt(segments)

            processing_time = time.time() - start_time
            logger.info(
                f"转录完成: {len(text)} 字, 耗时 {processing_time:.2f}s"
            )

            return ASRResult(
                text=text,
                segments=segments,
                srt=srt,
                language=self._language,
                emotion="",
                engine_used=f"funasr-nano-{self.engine_type.value}",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise RuntimeError(f"Fun-ASR-Nano 转录失败: {e}")

    def _extract_text(self, result) -> str:
        """从模型输出中提取文本"""
        if hasattr(result, "text"):
            return result.text or ""
        if isinstance(result, dict):
            return result.get("text", "") or result.get("transcript", "") or ""
        if isinstance(result, str):
            return result
        try:
            for item in result:
                if hasattr(item, "text"):
                    return item.text or ""
                if isinstance(item, dict):
                    return item.get("text", "") or ""
        except TypeError:
            pass
        return str(result) if result else ""

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
            supports_streaming=False,  # mlx-audio 实现暂不支持流式
            supports_cloning=False,
            supports_voice_design=False,
            supports_emotion=False,
            supported_languages=["zh", "en", "ja", "yue", "wuu", "nan", "hak", "gan", "hsn", "cjy", "auto"],
            min_memory_mb=1200,
            param_size="0.8B",
        )
