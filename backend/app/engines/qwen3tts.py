"""
Qwen3-TTS 引擎
使用 mlx-audio 库运行 Qwen3-TTS-12Hz-1.7B 模型
支持三种变体：
- Base: 零样本声音克隆（必须 ref_audio，自动 ASR 转写 ref_text）
- CustomVoice: 预设音色（9个内置音色，无需参考音频，不支持克隆）
- VoiceDesign: 自然语言声音设计（instruct 描述音色）

mlx-audio 0.4.8 API 规范：
- generate() 统一入口，按 config.tts_model_type 自动路由
- custom_voice: generate_custom_voice(speaker=, language=, instruct=)
- voice_design: generate_voice_design(instruct=, language=)
- base ICL: generate(ref_audio=, ref_text=) 必须同时提供
"""
import time
import numpy as np
from pathlib import Path
from typing import Optional, List

from .base import BaseTTSEngine, EngineType, TTSResult, ModelCapability
from ..audio.player import save_audio_to_wav, get_audio_duration_from_array
from ..audio.preprocess import apply_fade, normalize_audio
from ..utils.logger import logger


class Qwen3TTSEngine(BaseTTSEngine):
    """Qwen3-TTS 引擎（mlx-audio 实现）"""

    def __init__(self, model_key: str, engine_type: EngineType):
        super().__init__(model_key, engine_type)
        self._model = None
        self._model_path = None
        # 判断变体类型
        self._is_voice_design = "voicedesign" in model_key.lower() or "voice_design" in model_key.lower()
        self._is_custom_voice = "custom" in model_key.lower()
        # CustomVoice 模型的预设音色列表（与模型 config 中 spk_id 一致）
        self._preset_speakers = [
            "Serena", "Vivian", "Uncle_Fu", "Ryan", "Aiden",
            "Ono_Anna", "Sohee", "Eric", "Dylan",
        ]
        self._default_speaker = "Serena"
        # 预置音色元数据（来自 Qwen3-TTS 官方文档）
        self._preset_speaker_meta = {
            "Serena": {"gender": "female", "language": "zh", "description": "温暖温柔的年轻女声", "emotion": "温柔、平静"},
            "Vivian": {"gender": "female", "language": "zh", "description": "明亮略带个性的年轻女声", "emotion": "活泼、自信"},
            "Uncle_Fu": {"gender": "male", "language": "zh", "description": "低沉醇厚的成熟男声", "emotion": "稳重、温和"},
            "Dylan": {"gender": "male", "language": "zh", "description": "年轻北京男声，清晰自然", "emotion": "青春、活力", "dialect": "北京话"},
            "Eric": {"gender": "male", "language": "zh", "description": "活泼成都男声，略带沙哑", "emotion": "热情、开朗", "dialect": "四川话"},
            "Ryan": {"gender": "male", "language": "en", "description": "Dynamic male voice with strong rhythmic drive", "emotion": "Energetic, rhythmic"},
            "Aiden": {"gender": "male", "language": "en", "description": "Sunny American male voice with clear midrange", "emotion": "Bright, friendly"},
            "Ono_Anna": {"gender": "female", "language": "ja", "description": "年轻清澈的女声", "emotion": "清新、可爱"},
            "Sohee": {"gender": "female", "language": "ko", "description": "柔和温柔的女声", "emotion": "温柔、治愈"},
        }

    def load(self, model_path: str) -> None:
        """加载 Qwen3-TTS 模型"""
        if self._loaded:
            logger.info(f"模型已加载，跳过: {self.model_key}")
            return

        logger.info(f"正在加载 Qwen3-TTS 模型: {model_path}")

        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:
            raise ImportError(
                f"mlx-audio TTS 模块导入失败: {e}\n"
                "请确保已安装 mlx-audio==0.4.8 和 torch（mlx-audio 隐式依赖）\n"
                "安装命令: pip install mlx-audio==0.4.8 torch"
            )

        self._model = load_model(model_path)
        self._model_path = model_path
        self._loaded = True

        logger.info(f"Qwen3-TTS 模型加载成功: {model_path}")

    def unload(self) -> None:
        """卸载模型，释放内存"""
        if self._model is not None:
            del self._model
            self._model = None

        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

        self._loaded = False
        self._model_path = None
        logger.info(f"Qwen3-TTS 模型已卸载: {self.model_key}")

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        voice_description: Optional[str] = None,
    ) -> TTSResult:
        """
        合成语音
        - Base: 声音克隆（必须 ref_audio，自动转写 ref_text）
        - CustomVoice: 预设音色（voice 参数选择音色）
        - VoiceDesign: 自然语言声音设计（voice_description 描述音色）
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        # 清理文本
        text = self._clean_text(text)
        if not text:
            raise ValueError("合成文本不能为空")

        start_time = time.time()
        mode_name = "VoiceDesign" if self._is_voice_design else ("CustomVoice" if self._is_custom_voice else "Base")
        logger.info(f"开始合成: {len(text)} 字, 模式: {mode_name}")

        try:
            if self._is_voice_design:
                audio_array, sample_rate = self._synthesize_voice_design(
                    text, voice_description
                )
            elif self._is_custom_voice:
                audio_array, sample_rate = self._synthesize_custom(
                    text, reference_audio, voice
                )
            else:
                audio_array, sample_rate = self._synthesize_base(
                    text, reference_audio
                )

            # 统一后处理：静音裁剪 + 降噪 + 音量归一化 + 淡入淡出
            from ..audio.preprocess import postprocess_tts_audio
            audio_array = postprocess_tts_audio(
                audio_array,
                sample_rate,
                denoise=True,
                denoise_strength=0.2,
                normalize=True,
                fade=True,
                do_trim_silence=True,
            )

            # 保存为 WAV
            output_path = save_audio_to_wav(audio_array, sample_rate)
            duration = get_audio_duration_from_array(audio_array, sample_rate)

            processing_time = time.time() - start_time
            logger.info(
                f"合成完成: {duration:.2f}s 音频, 耗时 {processing_time:.2f}s, "
                f"采样率 {sample_rate}Hz"
            )

            return TTSResult(
                audio_path=output_path,
                duration=duration,
                sample_rate=sample_rate,
                engine_used=f"qwen3tts-{self.engine_type.value}",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"合成失败: {e}")
            raise RuntimeError(f"Qwen3-TTS 合成失败: {e}")

    def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        voice_description: Optional[str] = None,
    ):
        """
        流式合成（同步生成器，由 manager 放入线程池驱动）
        yield 16-bit PCM 音频块（bytes）
        """
        if not self._model:
            raise RuntimeError("模型未加载")

        text = self._clean_text(text)
        if not text:
            raise RuntimeError("文本不能为空")

        try:
            max_tokens = self._calc_max_tokens(text)
            sample_rate = 24000
            overlap_samples = int(sample_rate * 0.01)  # 10ms 交叉淡化
            # 根据模型类型选择流式生成方法
            if self._is_custom_voice:
                generator = self._model.generate_custom_voice(
                    text,
                    speaker=voice or self._default_speaker,
                    language="auto",
                    temperature=0.3,
                    top_k=30,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    max_tokens=max_tokens,
                    stream=True,
                    streaming_interval=2.0,
                    verbose=False,
                )
            elif self._is_voice_design:
                generator = self._model.generate(
                    text,
                    instruct=voice_description or "A natural, clear female voice speaking in Chinese.",
                    lang_code="auto",
                    temperature=0.3,
                    top_k=30,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    max_tokens=max_tokens,
                    stream=True,
                    streaming_interval=2.0,
                    verbose=False,
                )
            else:
                # Base ICL 模式
                if not reference_audio:
                    raise RuntimeError(
                        "Qwen3-TTS Base 模型必须上传参考音频进行声音克隆。"
                    )
                ref_text = self._get_ref_text(reference_audio)
                generator = self._model.generate(
                    text,
                    ref_audio=reference_audio,
                    ref_text=ref_text,
                    lang_code="auto",
                    temperature=0.3,
                    top_k=30,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    max_tokens=max_tokens,
                    stream=True,
                    streaming_interval=2.0,
                    verbose=False,
                )

            pending_chunk = None
            is_first_chunk = True

            # 逐块输出音频，使用流式安全的轻量后处理和平滑拼接
            for result in generator:
                if hasattr(result, "audio") and result.audio is not None:
                    audio_array = self._prepare_stream_chunk(
                        np.asarray(result.audio, dtype=np.float32),
                        sample_rate,
                        is_first_chunk=is_first_chunk,
                    )
                    if audio_array.size == 0:
                        continue

                    is_first_chunk = False
                    if pending_chunk is None:
                        pending_chunk = audio_array
                        continue

                    emit_chunk, pending_chunk = self._merge_stream_chunks(
                        pending_chunk,
                        audio_array,
                        overlap_samples,
                    )
                    if emit_chunk.size > 0:
                        yield self._pcm_bytes(emit_chunk)

            if pending_chunk is not None and pending_chunk.size > 0:
                final_chunk = self._finalize_stream_chunk(pending_chunk, sample_rate)
                if final_chunk.size > 0:
                    yield self._pcm_bytes(final_chunk)

        except Exception as e:
            logger.error(f"流式合成失败: {e}")
            raise RuntimeError(f"Qwen3-TTS 流式合成失败: {e}")

    def _prepare_stream_chunk(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        is_first_chunk: bool = False,
    ) -> np.ndarray:
        """流式块后处理：限幅、首块轻淡入，避免首包爆音。"""
        if audio_array.size == 0:
            return audio_array.astype(np.float32)

        audio_array = np.clip(audio_array.astype(np.float32), -1.0, 1.0)
        peak = np.max(np.abs(audio_array))
        if peak > 0.98:
            audio_array = normalize_audio(audio_array, target_peak=0.95)

        if is_first_chunk:
            fade_samples = min(int(sample_rate * 0.01), len(audio_array) // 2)
            if fade_samples > 0:
                fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                audio_array[:fade_samples] = audio_array[:fade_samples] * fade_curve

        return audio_array

    def _merge_stream_chunks(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        overlap_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对相邻块做短交叉淡化，减少块边界爆音和断裂感。"""
        overlap = min(overlap_samples, len(previous), len(current))
        if overlap <= 0:
            return previous, current

        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        blended = previous[-overlap:] * fade_out + current[:overlap] * fade_in
        emit_chunk = np.concatenate([previous[:-overlap], blended]).astype(np.float32)
        next_pending = current[overlap:].astype(np.float32)
        return emit_chunk, next_pending

    def _finalize_stream_chunk(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """最后一块补轻淡出，避免尾部点击声。"""
        if audio_array.size == 0:
            return audio_array.astype(np.float32)
        return apply_fade(audio_array.astype(np.float32), sample_rate, fade_duration=0.01)

    def _pcm_bytes(self, audio_array: np.ndarray) -> bytes:
        """将 float32 音频块转换为 16-bit PCM。"""
        return (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def _calc_max_tokens(self, text: str) -> int:
        """
        计算合适的 max_tokens，避免尾部丢音
        mlx-audio 默认: min(4096, max(75, text_len*6))
        这里放宽到 text_len*8，确保短文本也有足够 token 生成完整音频
        """
        return min(4096, max(512, len(text) * 8))

    def _synthesize_base(
        self,
        text: str,
        reference_audio: Optional[str],
    ) -> tuple:
        """
        Base 变体：零样本声音克隆（ICL 模式）
        mlx-audio 的 ICL 模式要求 ref_audio + ref_text 同时提供。
        Base 模型没有预设音色，必须提供参考音频。
        """
        if not reference_audio:
            raise RuntimeError(
                "Qwen3-TTS Base 模型必须上传参考音频进行声音克隆。"
                "如需无参考音频合成，请切换到 Qwen3-TTS CustomVoice（内置9个预设音色）。"
            )

        # 自动使用已加载的 Qwen3-ASR 转写参考音频获取 ref_text
        ref_text = self._get_ref_text(reference_audio)
        if not ref_text:
            logger.warning("参考音频 ASR 转写为空，ICL 模式可能质量下降")

        # 调用 mlx-audio 统一入口，自动路由到 Base ICL 模式
        # ref_audio 传文件路径，mlx-audio 内部自动加载为 24kHz
        # 采样参数基于社区优化：低 temperature 减少呼吸声/重复，repetition_penalty 抑制重复
        results = list(self._model.generate(
            text,
            ref_audio=reference_audio,
            ref_text=ref_text,
            lang_code="auto",
            temperature=0.3,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.3,
            max_tokens=self._calc_max_tokens(text),
            verbose=False,
        ))
        return self._collect_audio(results)

    def _synthesize_custom(
        self,
        text: str,
        reference_audio: Optional[str],
        speaker: Optional[str],
    ) -> tuple:
        """
        CustomVoice 变体：预设音色合成
        - 不支持参考音频克隆（mlx-audio 的 custom_voice 分支不接受 ref_audio）
        - 使用 generate_custom_voice(speaker=, language=) 直接调用
        """
        if reference_audio:
            raise RuntimeError(
                "Qwen3-TTS CustomVoice 不支持参考音频声音克隆。"
                "如需声音克隆，请切换到 Qwen3-TTS Base 模型。"
            )

        # 选择音色：用户指定优先，否则使用默认 Serena
        selected_speaker = self._default_speaker
        if speaker:
            # 大小写不敏感匹配
            for s in self._preset_speakers:
                if s.lower() == speaker.lower():
                    selected_speaker = s
                    break
            if selected_speaker != speaker:
                logger.warning(f"未知音色 '{speaker}'，使用默认音色 '{selected_speaker}'")

        logger.info(f"CustomVoice 模式：使用预设音色 '{selected_speaker}'")

        # 直接调用 generate_custom_voice（mlx-audio 官方 API）
        # 采样参数基于社区优化：低 temperature 减少呼吸声/重复
        results = list(self._model.generate_custom_voice(
            text=text,
            speaker=selected_speaker,
            language="auto",
            temperature=0.3,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.3,
            max_tokens=self._calc_max_tokens(text),
            verbose=False,
        ))
        return self._collect_audio(results)

    def _synthesize_voice_design(
        self,
        text: str,
        voice_description: Optional[str],
    ) -> tuple:
        """
        VoiceDesign 变体：自然语言声音设计
        使用 mlx-audio 统一入口 generate(instruct=)，自动路由到 voice_design 分支
        """
        # 默认音色描述（用户未提供时使用）
        if not voice_description or not voice_description.strip():
            voice_description = "自然、清晰的中文女声，语速适中，语调平稳"
            logger.info("VoiceDesign 模式：未提供音色描述，使用默认描述")

        logger.info(f"VoiceDesign 模式：instruct={voice_description}")

        # 通过统一入口 generate() 调用，mlx-audio 自动路由到 generate_voice_design
        # 采样参数基于社区优化（韩国博客实测 + learntoprompt.org）：
        # temperature=0.3 减少呼吸声/随机性，top_k=30 缩小候选范围，
        # top_p=0.85 过滤低概率token，repetition_penalty=1.3 抑制重复
        results = list(self._model.generate(
            text=text,
            instruct=voice_description,
            lang_code="auto",
            temperature=0.3,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.3,
            max_tokens=self._calc_max_tokens(text),
            verbose=False,
        ))
        return self._collect_audio(results)

    def _get_ref_text(self, reference_audio_path: str) -> str:
        """
        使用已加载的 ASR 引擎自动转写参考音频，获取 ref_text
        Base 模型的 ICL 模式需要 ref_text 与参考音频内容对应。
        优先复用 scheduler 中已加载的 ASR（Qwen3-ASR），避免每次合成都
        重新加载/卸载 ASR 导致的慢启动、内存峰值与临时文件泄漏。
        """
        import os
        import tempfile
        import soundfile as sf
        from ..audio.preprocess import load_audio
        from ..scheduler.scheduler import scheduler
        from .base import TaskType

        temp_path = None
        try:
            # 加载参考音频为 16kHz
            audio_np, sr = load_audio(reference_audio_path, target_sample_rate=16000)

            # 保存为临时 WAV 供 ASR 使用
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            sf.write(temp_path, audio_np, 16000)

            # 1) 优先复用已加载的 ASR 引擎
            asr = scheduler.get_loaded_engine("qwen3-asr", TaskType.ASR)
            if asr is None or not asr.is_loaded:
                # 2) 未加载则经 model_manager 加载一次（加载后由 scheduler 缓存复用，不立即 unload）
                try:
                    from ..models.manager import model_manager
                    model_manager.load_model("qwen3-asr")
                    asr = scheduler.get_loaded_engine("qwen3-asr", TaskType.ASR)
                except Exception as e:
                    logger.warning(f"自动加载 Qwen3-ASR 用于参考音频转写失败: {e}")
                    return ""

            if asr is None:
                logger.warning("Qwen3-ASR 不可用，跳过参考音频转写")
                return ""

            result = asr.transcribe(temp_path)
            ref_text = (result.text or "").strip()

            if not ref_text:
                logger.warning("ASR 转写参考音频为空")
                return ""

            logger.info(f"参考音频 ASR 转写成功: {len(ref_text)} 字")
            return ref_text

        except Exception as e:
            logger.warning(f"参考音频 ASR 转写失败: {e}")
            return ""
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _collect_audio(self, results: list) -> tuple:
        """从生成结果中收集音频数据"""
        if not results:
            raise RuntimeError("模型未生成音频")

        audio_chunks = []
        sample_rate = 24000  # Qwen3-TTS 默认 24000Hz

        for result in results:
            if hasattr(result, "audio") and result.audio is not None:
                audio_chunks.append(np.array(result.audio))
            if hasattr(result, "sample_rate") and result.sample_rate:
                sample_rate = result.sample_rate

        if not audio_chunks:
            raise RuntimeError("生成结果中没有音频数据")

        audio_array = np.concatenate(audio_chunks)
        return audio_array, sample_rate

    def _clean_text(self, text: str) -> str:
        """清理文本（移除换行符、Markdown符号，合并多余空格，确保末尾有标点符号防止尾部丢音）"""
        if not text:
            return text
        # 移除换行符
        text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        # 移除 Markdown 格式符号（#、*、_、`、>、- 等），这些会被模型念出奇怪声音
        import re
        text = re.sub(r'[#*_`~]', '', text)
        # 移除 Markdown 链接格式 [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 合并多余空格
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
        # 确保文本末尾有标点符号，防止模型因缺少结束标记而尾部丢音
        if text and text[-1] not in "。！？.!?；;，,":
            text += "。"
        return text

    def get_capability(self) -> ModelCapability:
        """获取模型能力"""
        if self._is_custom_voice:
            return ModelCapability(
                supports_streaming=True,
                supports_cloning=False,
                supports_voice_design=False,
                supports_emotion=False,
                requires_reference=False,
                preset_speakers=list(self._preset_speakers),
                preset_speaker_meta=dict(self._preset_speaker_meta),
                supported_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                min_memory_mb=2200,
                param_size="1.7B",
            )
        elif self._is_voice_design:
            return ModelCapability(
                supports_streaming=True,
                supports_cloning=False,
                supports_voice_design=True,
                supports_emotion=False,
                requires_reference=False,
                supported_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                min_memory_mb=2200,
                param_size="1.7B",
            )
        else:
            return ModelCapability(
                supports_streaming=True,
                supports_cloning=True,
                supports_voice_design=False,
                supports_emotion=False,
                requires_reference=True,
                supported_languages=["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
                min_memory_mb=2200,
                param_size="1.7B",
            )
