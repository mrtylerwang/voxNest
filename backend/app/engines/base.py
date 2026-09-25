"""
voxNest 推理引擎基类
定义统一的 ASR/TTS 引擎协议，所有引擎必须继承此基类
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass, field


class EngineType(str, Enum):
    """推理引擎类型"""
    MLX = "mlx"              # macOS Metal GPU 加速
    CPU = "cpu"              # 纯 CPU 降级（预留）


class TaskType(str, Enum):
    """任务类型"""
    ASR = "asr"
    TTS = "tts"


@dataclass
class ModelCapability:
    """模型能力描述"""
    supports_streaming: bool = False
    supports_cloning: bool = False
    supports_voice_design: bool = False
    supports_emotion: bool = False
    supports_timestamps: bool = False  # 是否支持时间戳输出
    requires_reference: bool = False  # 是否必须需要参考音频（如 Qwen3-TTS Base）
    preset_speakers: List[str] = field(default_factory=list)  # 内置预设音色列表（如 CustomVoice）
    preset_speaker_meta: Dict[str, dict] = field(default_factory=dict)  # 预置音色元数据（语言/性别/描述/情绪）
    supported_languages: List[str] = field(default_factory=list)
    min_memory_mb: int = 0
    param_size: str = ""


@dataclass
class ASRSegment:
    """ASR 分段结果"""
    text: str
    start: float  # 秒
    end: float    # 秒


@dataclass
class ASRResult:
    """ASR 完整结果"""
    text: str
    segments: List[ASRSegment] = field(default_factory=list)
    srt: str = ""
    language: str = ""
    emotion: str = ""
    engine_used: str = ""
    processing_time: float = 0.0


@dataclass
class TTSResult:
    """TTS 完整结果"""
    audio_path: str
    duration: float = 0.0
    sample_rate: int = 24000
    engine_used: str = ""
    processing_time: float = 0.0


class BaseASREngine(ABC):
    """ASR 引擎基类"""

    # 该引擎支持的推理引擎类型（子类可覆盖）
    # 当前所有引擎均基于 mlx-audio 实现，只支持 MLX
    supported_engine_types: List[EngineType] = [EngineType.MLX]

    def __init__(self, model_key: str, engine_type: EngineType):
        self.model_key = model_key
        self.engine_type = engine_type
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放内存"""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> ASRResult:
        """同步转录"""
        pass

    async def transcribe_stream(self, audio_path: str) -> AsyncGenerator[ASRSegment, None]:
        """流式转录（如果模型支持，默认不支持）"""
        result = self.transcribe(audio_path)
        for seg in result.segments:
            yield seg

    @abstractmethod
    def get_capability(self) -> ModelCapability:
        """获取模型能力"""
        pass


class BaseTTSEngine(ABC):
    """TTS 引擎基类"""

    # 该引擎支持的推理引擎类型（子类可覆盖）
    # 当前所有引擎均基于 mlx-audio 实现，只支持 MLX
    supported_engine_types: List[EngineType] = [EngineType.MLX]

    def __init__(self, model_key: str, engine_type: EngineType):
        self.model_key = model_key
        self.engine_type = engine_type
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放内存"""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        voice_description: Optional[str] = None,
    ) -> TTSResult:
        """同步合成"""
        pass

    def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        voice_description: Optional[str] = None,
    ):
        """流式合成（同步生成器，由 manager 放入线程池驱动；默认不支持）"""
        raise NotImplementedError("该模型不支持流式合成")

    @abstractmethod
    def get_capability(self) -> ModelCapability:
        """获取模型能力"""
        pass
