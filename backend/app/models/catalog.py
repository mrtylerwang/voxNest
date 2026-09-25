"""
模型清单（注册表）
定义所有内置模型的元数据，包括模型 ID、参数大小、能力、依赖等
新增模型只需在此文件中添加，无需修改其他代码
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ..engines.base import TaskType, ModelCapability


@dataclass
class ModelDependency:
    """模型依赖（辅助模型）"""
    key: str
    name: str
    description: str
    required: bool = True  # 是否必须下载


@dataclass
class ModelConfig:
    """模型配置"""
    key: str                    # 模型唯一标识
    name: str                   # 显示名称
    task_type: TaskType        # 任务类型
    param_size: str            # 参数大小（如 "0.5B"）
    description: str            # 描述
    mode: str                  # 模式：high_quality / balanced / lightweight

    # 下载源模型 ID
    modelscope_id: str = ""
    huggingface_id: str = ""   # mlx-community 版本
    official_hf_id: str = ""   # 官方 PyTorch 版本（备用）
    vendor: str = ""           # 开发厂家/团队

    # 模型能力
    capability: ModelCapability = field(default_factory=ModelCapability)

    # 依赖的辅助模型
    dependencies: List[ModelDependency] = field(default_factory=list)

    # 必需文件（用于检测是否已下载）
    required_files: List[str] = field(default_factory=list)

    # 模型文件大小估算（MB）
    estimated_size_mb: int = 0


# ===== ASR 模型清单 =====

ASR_MODELS: List[ModelConfig] = [
    ModelConfig(
        key="funasr-nano",
        name="Fun-ASR-Nano-2512",
        vendor="FunAudioLLM（通义实验室）",
        task_type=TaskType.ASR,
        param_size="0.8B",
        description="轻量高速中文ASR，支持7种方言、歌词/说唱识别，流式延迟极低",
        mode="balanced",
        modelscope_id="mlx-community/Fun-ASR-Nano-2512",
        huggingface_id="mlx-community/Fun-ASR-Nano-2512",
        official_hf_id="FunAudioLLM/Fun-ASR-Nano-2512",
        capability=ModelCapability(
            supports_streaming=True,
            supports_cloning=False,
            supports_voice_design=False,
            supports_emotion=False,
            supports_timestamps=False,
            supported_languages=["zh", "en", "ja", "auto"],
            min_memory_mb=1200,
            param_size="0.8B",
        ),
        dependencies=[],
        required_files=["config.json", "model.safetensors"],
        estimated_size_mb=1200,
    ),
    ModelConfig(
        key="qwen3-asr",
        name="Qwen3-ASR-1.7B",
        vendor="Qwen 通义千问（阿里巴巴）",
        task_type=TaskType.ASR,
        param_size="1.7B",
        description="高精度多语言ASR，支持52种语言和方言，自动语种检测，长音频分块，时间戳输出，媲美商业API",
        mode="high_quality",
        modelscope_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        huggingface_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        official_hf_id="Qwen/Qwen3-ASR-1.7B",
        capability=ModelCapability(
            supports_streaming=False,
            supports_cloning=False,
            supports_voice_design=False,
            supports_emotion=False,
            supports_timestamps=True,
            supported_languages=[
                "zh", "en", "yue", "ja", "ko", "de", "fr", "es", "pt",
                "ru", "it", "ar", "id", "th", "vi", "tr", "hi", "ms",
                "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el",
                "hu", "mk", "ro", "auto",
            ],
            min_memory_mb=2048,
            param_size="1.7B",
        ),
        dependencies=[],
        required_files=["config.json", "model.safetensors"],
        estimated_size_mb=1700,
    ),
]

# ===== TTS 模型清单 =====

TTS_MODELS: List[ModelConfig] = [
    ModelConfig(
        key="qwen3tts-base",
        name="Qwen3-TTS-1.7B-Base",
        vendor="Qwen 通义千问（阿里巴巴）",
        task_type=TaskType.TTS,
        param_size="1.7B",
        description="功能全面的开源中文TTS，支持3秒声纹克隆、流式生成、多方言",
        mode="high_quality",
        modelscope_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        huggingface_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        official_hf_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        capability=ModelCapability(
            supports_streaming=True,
            supports_cloning=True,
            supports_voice_design=False,
            supports_emotion=False,
            requires_reference=True,  # Base 模型无预设音色，必须参考音频
            supported_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
            min_memory_mb=2200,
            param_size="1.7B",
        ),
        dependencies=[
            ModelDependency(
                key="qwen3-tts-tokenizer",
                name="Qwen3-TTS-Tokenizer-12Hz",
                description="Qwen3-TTS 必须的语音分词器，下载主模型时自动下载",
                required=True,
            ),
        ],
        required_files=["config.json", "model.safetensors"],
        estimated_size_mb=2400,
    ),
    ModelConfig(
        key="qwen3tts-custom",
        name="Qwen3-TTS-1.7B-CustomVoice",
        vendor="Qwen 通义千问（阿里巴巴）",
        task_type=TaskType.TTS,
        param_size="1.7B",
        description="内置9个预设音色，无需参考音频即可合成；不支持参考音频声音克隆",
        mode="high_quality",
        modelscope_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        huggingface_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        official_hf_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        capability=ModelCapability(
            supports_streaming=True,
            supports_cloning=False,  # CustomVoice 的 generate 不支持 ref_audio ICL 克隆
            supports_voice_design=False,
            supports_emotion=False,
            preset_speakers=["Serena", "Vivian", "Uncle_Fu", "Ryan", "Aiden", "Ono_Anna", "Sohee", "Eric", "Dylan"],
            supported_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
            min_memory_mb=2200,
            param_size="1.7B",
        ),
        dependencies=[
            ModelDependency(
                key="qwen3-tts-tokenizer",
                name="Qwen3-TTS-Tokenizer-12Hz",
                description="Qwen3-TTS 必须的语音分词器，下载主模型时自动下载",
                required=True,
            ),
        ],
        required_files=["config.json", "model.safetensors"],
        estimated_size_mb=2400,
    ),
    ModelConfig(
        key="qwen3tts-voicedesign",
        name="Qwen3-TTS-1.7B-VoiceDesign",
        vendor="Qwen 通义千问（阿里巴巴）",
        task_type=TaskType.TTS,
        param_size="1.7B",
        description="支持自然语言音色设计（如'磁性男人的声音'、'温柔的女声'），通过文字描述生成目标音色",
        mode="high_quality",
        modelscope_id="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        huggingface_id="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        official_hf_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        capability=ModelCapability(
            supports_streaming=True,
            supports_cloning=False,
            supports_voice_design=True,
            supports_emotion=False,
            supported_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
            min_memory_mb=2200,
            param_size="1.7B",
        ),
        dependencies=[
            ModelDependency(
                key="qwen3-tts-tokenizer",
                name="Qwen3-TTS-Tokenizer-12Hz",
                description="Qwen3-TTS 必须的语音分词器，下载主模型时自动下载",
                required=True,
            ),
        ],
        required_files=["config.json", "model.safetensors"],
        estimated_size_mb=2400,
    ),
]

# ===== 辅助模型清单 =====

AUXILIARY_MODELS: List[ModelConfig] = [
    ModelConfig(
        key="qwen3-tts-tokenizer",
        name="Qwen3-TTS-Tokenizer-12Hz",
        vendor="Qwen 通义千问（阿里巴巴）",
        task_type=TaskType.TTS,
        param_size="—",
        description="Qwen3-TTS 必须的语音分词器，12.5Hz多码本分词器",
        mode="lightweight",
        modelscope_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        huggingface_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        official_hf_id="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        capability=ModelCapability(
            min_memory_mb=100,
        ),
        dependencies=[],
        required_files=["config.json", "model.safetensors"],  # 实际下载包含 model.safetensors (682MB)，无 tokenizer.json
        estimated_size_mb=700,
    ),
]


def get_all_models() -> List[ModelConfig]:
    """获取所有模型（主模型 + 辅助模型）"""
    return ASR_MODELS + TTS_MODELS + AUXILIARY_MODELS


def get_asr_models() -> List[ModelConfig]:
    """获取所有 ASR 主模型"""
    return ASR_MODELS


def get_tts_models() -> List[ModelConfig]:
    """获取所有 TTS 主模型"""
    return TTS_MODELS


def get_auxiliary_models() -> List[ModelConfig]:
    """获取所有辅助模型"""
    return AUXILIARY_MODELS


def get_model_by_key(key: str) -> Optional[ModelConfig]:
    """根据 key 获取模型配置"""
    for model in get_all_models():
        if model.key == key:
            return model
    return None


def get_models_by_mode(task_type: TaskType, mode: str) -> List[ModelConfig]:
    """根据任务类型和模式获取模型"""
    models = ASR_MODELS if task_type == TaskType.ASR else TTS_MODELS
    return [m for m in models if m.mode == mode]


def get_model_dependencies(model_key: str) -> List[ModelDependency]:
    """获取模型的依赖列表"""
    model = get_model_by_key(model_key)
    if model:
        return model.dependencies
    return []
