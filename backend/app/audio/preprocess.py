"""
音频预处理
音频加载、重采样、归一化等预处理功能
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from ..utils.logger import logger


def load_audio(
    file_path: str,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件，返回 (audio_array, sample_rate)
    使用 soundfile 加载（支持 WAV/FLAC/OGG 等）
    对于 MP3/M4A/MP4 等格式，需要先用 ffmpeg 转换
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile 未安装，请安装: pip install soundfile")

    file_path = str(file_path)

    # 检查文件是否存在
    if not Path(file_path).exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    # 尝试直接加载
    try:
        audio, sample_rate = sf.read(file_path, dtype="float32")

        # 转换为单声道
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 重采样（如果需要）
        if sample_rate != target_sample_rate:
            audio = resample_audio(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        return audio.astype(np.float32), sample_rate

    except Exception as e:
        # soundfile 无法加载（可能是 MP3/MP4 等格式），使用 ffmpeg 转换
        logger.info(f"soundfile 无法直接加载，使用 ffmpeg 转换: {e}")
        from .converter import converter
        wav_path = converter.convert_to_wav(
            file_path,
            sample_rate=target_sample_rate,
            channels=1 if mono else 2,
        )
        try:
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            if mono and len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            return audio.astype(np.float32), sample_rate
        finally:
            # 清理临时文件
            converter.cleanup_temp_file(wav_path)


def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    简单的线性插值重采样
    对于高质量重采样建议使用 librosa.resample
    """
    if original_rate == target_rate:
        return audio

    duration = len(audio) / original_rate
    target_length = int(duration * target_rate)

    # 线性插值
    original_indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(original_indices, np.arange(len(audio)), audio)

    return resampled.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    音频峰值归一化
    将音频的峰值调整到 target_peak
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    min_silence_duration: float = 0.3,
) -> np.ndarray:
    """
    简单的静音裁剪
    移除开头和结尾的静音段
    """
    # 计算能量
    frame_size = int(0.01 * sample_rate)  # 10ms 帧
    if len(audio) < frame_size:
        return audio

    num_frames = len(audio) // frame_size
    energies = np.array([
        np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
        for i in range(num_frames)
    ])

    # 找到第一个和最后一个超过阈值的帧
    active_frames = np.where(energies > threshold)[0]

    if len(active_frames) == 0:
        return audio  # 全部是静音，返回原音频

    # 使用更保守的前后留白，避免截掉自然起音和尾音
    padding_frames = max(3, int((min_silence_duration / 0.01) * 0.5))
    start_frame = max(0, active_frames[0] - padding_frames)
    end_frame = min(num_frames, active_frames[-1] + padding_frames)

    start_sample = start_frame * frame_size
    end_sample = min(len(audio), end_frame * frame_size)

    return audio[start_sample:end_sample]


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """获取音频时长（秒）"""
    return len(audio) / sample_rate


def denoise_audio(
    audio: np.ndarray,
    sample_rate: int,
    prop_decrease: float = 0.2,
) -> np.ndarray:
    """
    音频降噪（使用 noisereduce）
    使用音频开头和结尾的静音段作为噪音样本估计
    prop_decrease: 降噪强度，0.0-1.0，TTS 生成音频建议 0.1-0.3（过度降噪会失真）
    """
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce 未安装，跳过后处理降噪")
        return audio

    if len(audio) < int(0.1 * sample_rate):
        return audio  # 音频太短，不降噪

    # 低噪音检测：TTS 模型生成的音频通常背景噪音很低
    # 整体 RMS < 0.008 说明音频已经很干净，跳过降噪避免声音发闷
    overall_rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0
    if overall_rms < 0.008:
        logger.info(f"音频整体 RMS {overall_rms:.5f} 低于阈值，跳过降噪")
        return audio

    # 检测静音段作为噪音样本
    noise_sample_duration = 0.05  # 50ms
    noise_samples = int(noise_sample_duration * sample_rate)

    start_energy = np.sqrt(np.mean(audio[:noise_samples] ** 2)) if len(audio) > noise_samples else 0
    end_energy = np.sqrt(np.mean(audio[-noise_samples:] ** 2)) if len(audio) > noise_samples else 0

    # 如果开头或结尾是静音，使用静音段作为噪音样本
    noise_threshold = 0.005
    if start_energy < noise_threshold or end_energy < noise_threshold:
        if start_energy < noise_threshold and end_energy < noise_threshold:
            noise_sample = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
        elif start_energy < noise_threshold:
            noise_sample = audio[:noise_samples]
        else:
            noise_sample = audio[-noise_samples:]

        try:
            denoised = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=prop_decrease,
            )
            return denoised.astype(np.float32)
        except Exception as e:
            logger.warning(f"使用静音样本降噪失败，使用全局降噪: {e}")

    # 全局降噪（没有检测到静音段时）
    try:
        denoised = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=prop_decrease,
        )
        return denoised.astype(np.float32)
    except Exception as e:
        logger.warning(f"音频降噪失败: {e}")
        return audio


def normalize_rms(
    audio: np.ndarray,
    target_rms: float = 0.15,
    max_gain: float = 3.0,
) -> np.ndarray:
    """
    RMS 音量归一化
    将音频的 RMS 音量调整到 target_rms（正常语音水平）
    max_gain: 最大增益限制，避免爆音
    """
    current_rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0
    if current_rms > 0.001:
        gain = min(target_rms / current_rms, max_gain)
        if gain > 1.0:
            audio = audio * gain
            logger.info(f"音量归一化: RMS {current_rms:.4f} → {current_rms * gain:.4f} (增益 {gain:.2f}x)")
    return audio


def apply_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_duration: float = 0.02,
) -> np.ndarray:
    """
    应用淡入淡出
    fade_duration: 淡入淡出时长（秒），默认20ms
    """
    fade_samples = int(fade_duration * sample_rate)
    if len(audio) > fade_samples * 2:
        # 淡入
        fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[:fade_samples] = audio[:fade_samples] * fade_curve
        # 淡出
        fade_out_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        audio[-fade_samples:] = audio[-fade_samples:] * fade_out_curve
    return audio


def postprocess_tts_audio(
    audio: np.ndarray,
    sample_rate: int,
    denoise: bool = True,
    denoise_strength: float = 0.4,
    normalize: bool = True,
    fade: bool = True,
    do_trim_silence: bool = True,
) -> np.ndarray:
    """
    TTS 生成音频统一后处理
    1. 静音裁剪（移除开头结尾的静音）
    2. 降噪（去除背景噪音）
    3. 音量归一化（RMS归一化到正常语音水平）
    4. 淡入淡出（避免爆音）
    """
    if len(audio) == 0:
        return audio

    # 1. 静音裁剪
    if do_trim_silence:
        audio = trim_silence(audio, sample_rate, threshold=0.01, min_silence_duration=0.25)

    # 2. 降噪
    if denoise:
        audio = denoise_audio(audio, sample_rate, prop_decrease=denoise_strength)

    # 3. 音量归一化
    if normalize:
        audio = normalize_rms(audio, target_rms=0.15, max_gain=3.0)

    # 4. 淡入淡出
    if fade:
        audio = apply_fade(audio, sample_rate, fade_duration=0.02)

    return audio.astype(np.float32)
