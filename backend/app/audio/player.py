"""
音频播放和保存工具
"""
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional
from ..utils.logger import logger


def save_audio_to_wav(
    audio_array: np.ndarray,
    sample_rate: int,
    output_path: Optional[str] = None,
) -> str:
    """
    将音频数组保存为 WAV 文件
    audio_array: numpy 数组或 mx.array
    返回保存的文件路径
    """
    # 转换为 numpy 数组（如果是 mx.array）
    if hasattr(audio_array, "__array__"):
        audio_array = np.array(audio_array)
    elif not isinstance(audio_array, np.ndarray):
        try:
            import mlx.core as mx
            if isinstance(audio_array, mx.array):
                audio_array = np.array(audio_array)
        except ImportError:
            pass

    # 确保是 float32
    audio_array = audio_array.astype(np.float32)

    # 如果没有指定输出路径，创建临时文件
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 使用 soundfile 保存
    try:
        import soundfile as sf
        sf.write(output_path, audio_array, sample_rate)
    except ImportError:
        # soundfile 不可用时，使用 wave 模块手动写入
        _write_wav_manual(output_path, audio_array, sample_rate)

    logger.info(f"音频已保存: {output_path} ({len(audio_array) / sample_rate:.2f}s, {sample_rate}Hz)")
    return output_path


def _write_wav_manual(file_path: str, audio: np.ndarray, sample_rate: int) -> None:
    """手动写入 WAV 文件（soundfile 不可用时的回退）"""
    import wave
    import struct

    # 归一化到 int16 范围
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def get_audio_duration_from_array(audio_array: np.ndarray, sample_rate: int) -> float:
    """从音频数组获取时长（秒）"""
    if hasattr(audio_array, "shape"):
        return len(audio_array) / sample_rate
    return 0.0


def convert_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 24000,
) -> str:
    """
    将音频文件转换为 WAV 格式（用于参考音频预处理）
    使用 ffmpeg 转换
    """
    from .converter import converter
    return converter.convert_to_wav(
        input_path,
        output_path=output_path,
        sample_rate=sample_rate,
        channels=1,
    )
