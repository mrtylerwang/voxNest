"""
引擎注册表
在应用启动时注册所有可用的引擎到调度器
"""
from typing import Dict, Type
from .base import BaseASREngine, BaseTTSEngine
from ..scheduler.scheduler import scheduler
from ..utils.logger import logger


def register_all_engines() -> None:
    """注册所有可用引擎"""
    # ASR 引擎
    asr_engines = _get_asr_engines()
    for model_key, engine_class in asr_engines.items():
        try:
            scheduler.register_asr_engine(model_key, engine_class)
            logger.debug(f"注册 ASR 引擎: {model_key} -> {engine_class.__name__}")
        except Exception as e:
            logger.warning(f"注册 ASR 引擎失败 {model_key}: {e}")

    # TTS 引擎
    tts_engines = _get_tts_engines()
    for model_key, engine_class in tts_engines.items():
        try:
            scheduler.register_tts_engine(model_key, engine_class)
            logger.debug(f"注册 TTS 引擎: {model_key} -> {engine_class.__name__}")
        except Exception as e:
            logger.warning(f"注册 TTS 引擎失败 {model_key}: {e}")

    logger.info(f"引擎注册完成: ASR {len(asr_engines)} 个, TTS {len(tts_engines)} 个")


def _get_asr_engines() -> Dict[str, Type[BaseASREngine]]:
    """获取所有 ASR 引擎类"""
    engines = {}

    # Fun-ASR-Nano
    try:
        from .funasr_nano import FunASRNanoEngine
        engines["funasr-nano"] = FunASRNanoEngine
    except ImportError as e:
        logger.warning(f"Fun-ASR-Nano 引擎不可用: {e}")

    # Qwen3-ASR
    try:
        from .qwen3asr import Qwen3ASREngine
        engines["qwen3-asr"] = Qwen3ASREngine
    except ImportError as e:
        logger.warning(f"Qwen3-ASR 引擎不可用: {e}")

    return engines


def _get_tts_engines() -> Dict[str, Type[BaseTTSEngine]]:
    """获取所有 TTS 引擎类"""
    engines = {}

    # Qwen3-TTS（Base / CustomVoice / VoiceDesign 共用同一个引擎类）
    try:
        from .qwen3tts import Qwen3TTSEngine
        engines["qwen3tts-base"] = Qwen3TTSEngine
        engines["qwen3tts-custom"] = Qwen3TTSEngine
        engines["qwen3tts-voicedesign"] = Qwen3TTSEngine
    except ImportError as e:
        logger.warning(f"Qwen3-TTS 引擎不可用: {e}")

    return engines
