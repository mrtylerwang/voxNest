"""
模型管理器
负责模型的加载、卸载、切换、状态查询
所有模型操作必须通过此管理器，禁止在 API 层直接操作引擎
"""
from typing import Optional, Dict, List
import os
from ..engines.base import BaseASREngine, BaseTTSEngine, TaskType, ASRResult, TTSResult
from ..scheduler.scheduler import scheduler
from ..utils.logger import logger
from ..audio.converter import converter
from .catalog import (
    get_model_by_key, get_asr_models, get_tts_models,
    get_auxiliary_models, get_all_models, ModelConfig
)
from .storage import storage


class ModelManager:
    """模型管理器单例"""

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """初始化"""
        self._current_asr: Optional[str] = None
        self._current_tts: Optional[str] = None
        self._engine_registry: Dict[str, type] = {}
        logger.info("模型管理器初始化完成")

    def register_engine(self, model_key: str, engine_class: type) -> None:
        """注册模型引擎类"""
        self._engine_registry[model_key] = engine_class
        model_config = get_model_by_key(model_key)
        if model_config:
            if model_config.task_type == TaskType.ASR:
                scheduler.register_asr_engine(model_key, engine_class)
            else:
                scheduler.register_tts_engine(model_key, engine_class)
        logger.debug(f"注册模型引擎: {model_key} -> {engine_class.__name__}")

    def _build_source_info(self, model) -> dict:
        """构造下载来源信息（厂家、模型ID、各源支持情况）

        静态反映模型在 catalog 中声明的下载源，不依赖全局 download_source 设置，
        避免所有模型卡片的来源信息随全局设置漂移。
        """
        sources = {}
        if model.modelscope_id:
            sources["modelscope"] = {
                "model_id": model.modelscope_id,
                "url": f"https://www.modelscope.cn/models/{model.modelscope_id}",
            }
        if model.huggingface_id:
            sources["huggingface"] = {
                "model_id": model.huggingface_id,
                "url": f"https://huggingface.co/{model.huggingface_id}",
                "mirror_url": f"https://hf-mirror.com/{model.huggingface_id}",
            }
        return {
            "vendor": model.vendor,
            # 规范模型 ID（用于团队筛选等），优先 HF（mlx-community 社区版）
            "source_model_id": model.huggingface_id or model.modelscope_id or model.official_hf_id,
            "supports_modelscope": bool(model.modelscope_id),
            "supports_huggingface": bool(model.huggingface_id),
            "sources": sources,
        }

    def get_model_list(self, task_type: Optional[TaskType] = None) -> List[dict]:
        """
        获取模型列表（含下载状态）
        task_type=None 时返回所有主模型
        """
        if task_type == TaskType.ASR:
            models = get_asr_models()
        elif task_type == TaskType.TTS:
            models = get_tts_models()
        else:
            models = get_asr_models() + get_tts_models()

        result = []
        available_memory_mb = scheduler.hardware.available_memory_mb or scheduler.hardware.total_memory_mb

        for model in models:
            is_downloaded = storage.is_model_downloaded(model.key)
            model_size = storage.get_model_size(model.key) if is_downloaded else 0

            # 内存不足时标记不推荐（性能分档/低配置模式已移除，仅保留内存阈值提示）
            min_memory = model.capability.min_memory_mb or 1000
            memory_ratio = min_memory / max(available_memory_mb, 1)
            recommended = False
            not_recommended = False
            not_recommended_reason = None
            if memory_ratio > 0.9:
                not_recommended = True
                not_recommended_reason = f"模型需约 {min_memory}MB 内存，当前可用内存可能不足"

            # 如果模型已加载，从引擎实例获取最新 capability（含 preset_speaker_meta 等动态字段）
            loaded_engine = scheduler.get_loaded_engine(model.key, model.task_type)
            cap = loaded_engine.get_capability() if loaded_engine else model.capability

            result.append({
                "key": model.key,
                "name": model.name,
                **self._build_source_info(model),
                "task_type": model.task_type.value,
                "param_size": model.param_size,
                "description": model.description,
                "mode": model.mode,
                "is_downloaded": is_downloaded,
                "model_size_mb": int(model_size / (1024 * 1024)) if model_size else 0,
                "estimated_size_mb": model.estimated_size_mb,
                "capability": {
                    "supports_streaming": cap.supports_streaming,
                    "supports_cloning": cap.supports_cloning,
                    "supports_voice_design": cap.supports_voice_design,
                    "supports_emotion": cap.supports_emotion,
                    "supports_timestamps": cap.supports_timestamps,
                    "requires_reference": cap.requires_reference,
                    "preset_speakers": cap.preset_speakers,
                    "preset_speaker_meta": cap.preset_speaker_meta,
                    "supported_languages": cap.supported_languages,
                    "min_memory_mb": cap.min_memory_mb,
                },
                "recommended": recommended,
                "not_recommended": not_recommended,
                "not_recommended_reason": not_recommended_reason,
                "dependencies": [
                    {
                        "key": dep.key,
                        "name": dep.name,
                        "description": dep.description,
                        "required": dep.required,
                        "is_downloaded": storage.is_model_downloaded(dep.key),
                    }
                    for dep in model.dependencies
                ],
                "is_current": (
                    (task_type == TaskType.ASR and self._current_asr == model.key) or
                    (task_type == TaskType.TTS and self._current_tts == model.key)
                ),
            })

        return result

    def get_auxiliary_model_list(self) -> List[dict]:
        """获取辅助模型列表"""
        models = get_auxiliary_models()
        result = []
        for model in models:
            is_downloaded = storage.is_model_downloaded(model.key)
            model_size = storage.get_model_size(model.key) if is_downloaded else 0
            result.append({
                "key": model.key,
                "name": model.name,
                **self._build_source_info(model),
                "task_type": "aux",
                "param_size": model.param_size,
                "description": model.description,
                "is_downloaded": is_downloaded,
                "model_size_mb": int(model_size / (1024 * 1024)) if model_size else 0,
                "estimated_size_mb": model.estimated_size_mb,
                "dependencies": [],
            })
        return result

    def load_model(self, model_key: str) -> dict:
        """
        加载模型
        返回加载结果信息
        """
        model_config = get_model_by_key(model_key)
        if model_config is None:
            raise ValueError(f"未知模型: {model_key}")

        if not storage.is_model_downloaded(model_key):
            raise RuntimeError(f"模型未下载: {model_key}")

        # 检查依赖
        for dep in model_config.dependencies:
            if dep.required and not storage.is_model_downloaded(dep.key):
                raise RuntimeError(
                    f"依赖模型未下载: {dep.name} ({dep.key})，请先下载"
                )

        model_path = str(storage.get_model_dir(model_key))

        try:
            engine = scheduler.get_best_engine(model_key, model_config.task_type, model_path)

            if model_config.task_type == TaskType.ASR:
                # 卸载当前 ASR 引擎（如果不同）
                if self._current_asr and self._current_asr != model_key:
                    scheduler.unload_engine(self._current_asr, TaskType.ASR)
                self._current_asr = model_key
            else:
                if self._current_tts and self._current_tts != model_key:
                    scheduler.unload_engine(self._current_tts, TaskType.TTS)
                self._current_tts = model_key

            logger.info(f"模型加载成功: {model_key} ({model_config.task_type.value})")
            return {
                "success": True,
                "model_key": model_key,
                "task_type": model_config.task_type.value,
                "engine_type": engine.engine_type.value,
                "message": "模型加载成功",
            }
        except Exception as e:
            logger.error(f"模型加载失败: {model_key}, error: {e}")
            raise RuntimeError(f"模型加载失败: {e}") from e

    def unload_model(self, model_key: str) -> dict:
        """卸载模型"""
        model_config = get_model_by_key(model_key)
        if model_config is None:
            raise ValueError(f"未知模型: {model_key}")

        scheduler.unload_engine(model_key, model_config.task_type)

        if model_config.task_type == TaskType.ASR and self._current_asr == model_key:
            self._current_asr = None
        elif model_config.task_type == TaskType.TTS and self._current_tts == model_key:
            self._current_tts = None

        return {
            "success": True,
            "model_key": model_key,
            "message": "模型已卸载",
        }

    def get_current_model(self, task_type: TaskType) -> Optional[str]:
        """获取当前使用的模型 key"""
        if task_type == TaskType.ASR:
            return self._current_asr
        return self._current_tts

    def get_loaded_models(self) -> List[str]:
        """获取当前已加载的模型列表"""
        return scheduler.get_loaded_engines()

    def transcribe(self, audio_path: str, model_key: Optional[str] = None) -> ASRResult:
        """
        ASR 转录
        model_key=None 时使用当前已加载的模型
        自动将非 WAV 格式（MP4/MP3/M4A/FLAC/OGG等）转换为 WAV
        """
        if model_key:
            self.load_model(model_key)
        elif not self._current_asr:
            raise RuntimeError("未选择 ASR 模型，请先加载模型")

        engine = scheduler.get_loaded_engine(self._current_asr, TaskType.ASR)
        if engine is None or not engine.is_loaded:
            self.load_model(self._current_asr)
            engine = scheduler.get_loaded_engine(self._current_asr, TaskType.ASR)

        # 非 WAV 格式自动转换（MP4/MP3/M4A/FLAC/OGG等）
        wav_path = audio_path
        temp_wav = None
        ext = os.path.splitext(audio_path)[1].lower()
        if ext != ".wav":
            logger.info(f"非 WAV 格式检测到 ({ext})，使用 ffmpeg 转换为 WAV")
            try:
                temp_wav = converter.convert_to_wav(audio_path)
                wav_path = temp_wav
                logger.info(f"格式转换完成: {audio_path} -> {wav_path}")
            except Exception as e:
                logger.error(f"ffmpeg 格式转换失败: {e}")
                raise RuntimeError(f"音频格式转换失败（{ext}），请确保 ffmpeg 可用: {e}")

        try:
            return engine.transcribe(wav_path)
        finally:
            # 清理临时转换文件
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except OSError:
                    pass

    def synthesize(
        self,
        text: str,
        model_key: Optional[str] = None,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        voice_description: Optional[str] = None,
    ) -> TTSResult:
        """
        TTS 合成
        model_key=None 时使用当前已加载的模型
        """
        if model_key:
            self.load_model(model_key)
        elif not self._current_tts:
            raise RuntimeError("未选择 TTS 模型，请先加载模型")

        engine = scheduler.get_loaded_engine(self._current_tts, TaskType.TTS)
        if engine is None or not engine.is_loaded:
            self.load_model(self._current_tts)
            engine = scheduler.get_loaded_engine(self._current_tts, TaskType.TTS)

        return engine.synthesize(text, voice, reference_audio, voice_description)

    async def synthesize_stream(
        self,
        text: str,
        model_key: Optional[str] = None,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        voice_description: Optional[str] = None,
    ):
        """
        TTS 流式合成（异步生成器）
        引擎 synthesize_stream 是同步生成器（MLX 推理阻塞），
        用线程池 + asyncio.Queue 桥接，避免阻塞事件循环。
        """
        import asyncio
        if model_key:
            self.load_model(model_key)
        elif not self._current_tts:
            raise RuntimeError("未选择 TTS 模型，请先加载模型")

        engine = scheduler.get_loaded_engine(self._current_tts, TaskType.TTS)
        if engine is None or not engine.is_loaded:
            self.load_model(self._current_tts)
            engine = scheduler.get_loaded_engine(self._current_tts, TaskType.TTS)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        SENTINEL = object()

        def _producer():
            try:
                for chunk in engine.synthesize_stream(text, voice, reference_audio, voice_description):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)

        import threading
        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def get_system_status(self) -> dict:
        """获取系统状态"""
        return {
            "current_asr": self._current_asr,
            "current_tts": self._current_tts,
            "loaded_engines": scheduler.get_loaded_engines(),
            "memory": scheduler.get_memory_usage(),
            "storage": storage.get_storage_info(),
            "platform": scheduler.platform.value,
            "arch": scheduler.arch.value,
            "hardware": {
                "gpu_type": scheduler.hardware.gpu_type,
                "gpu_model": scheduler.hardware.gpu_model,
                "total_memory_mb": scheduler.hardware.total_memory_mb,
            },
            "versions": self._get_component_versions(),
        }

    def _get_component_versions(self) -> dict:
        """获取推理加速组件版本（MLX/mlx-audio 等）"""
        import sys
        from importlib.metadata import version as pkg_version, PackageNotFoundError
        versions = {"python": sys.version.split()[0]}
        for pkg in ("mlx-audio", "mlx", "transformers", "torch"):
            try:
                versions[pkg.replace("-", "_")] = pkg_version(pkg)
            except PackageNotFoundError:
                versions[pkg.replace("-", "_")] = None
        return versions


# 全局模型管理器实例
model_manager = ModelManager()
