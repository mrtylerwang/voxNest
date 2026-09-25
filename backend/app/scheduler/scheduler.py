"""
引擎调度器
负责平台检测、硬件检测、引擎优先级排序、自动降级
所有模型加载必须通过此调度器，禁止在业务代码中直接实例化引擎
"""
from typing import Optional, Dict, Type, List
from ..engines.base import (
    BaseASREngine, BaseTTSEngine, EngineType, TaskType, ModelCapability
)
from .platform import detect_platform, detect_arch, Platform, Arch, is_apple_silicon
from .hardware import detect_hardware, HardwareInfo
from ..utils.logger import logger


class EngineScheduler:
    """引擎调度器单例"""

    _instance: Optional["EngineScheduler"] = None

    def __new__(cls) -> "EngineScheduler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """初始化调度器"""
        self._platform = detect_platform()
        self._arch = detect_arch()
        self._hardware = detect_hardware()
        self._asr_engines: Dict[str, Type[BaseASREngine]] = {}
        self._tts_engines: Dict[str, Type[BaseTTSEngine]] = {}
        self._loaded_engines: Dict[str, BaseASREngine | BaseTTSEngine] = {}

        logger.info(
            f"引擎调度器初始化: platform={self._platform.value}, "
            f"arch={self._arch.value}, gpu={self._hardware.gpu_type}, "
            f"memory={self._hardware.total_memory_mb}MB"
        )

    def register_asr_engine(self, model_key: str, engine_class: Type[BaseASREngine]) -> None:
        """注册 ASR 引擎"""
        self._asr_engines[model_key] = engine_class
        logger.debug(f"注册 ASR 引擎: {model_key} -> {engine_class.__name__}")

    def register_tts_engine(self, model_key: str, engine_class: Type[BaseTTSEngine]) -> None:
        """注册 TTS 引擎"""
        self._tts_engines[model_key] = engine_class
        logger.debug(f"注册 TTS 引擎: {model_key} -> {engine_class.__name__}")

    def get_engine_priority(self, model_key: str, task_type: TaskType) -> List[EngineType]:
        """
        返回引擎优先级列表。当前仅支持 macOS + MLX，CPU 作为预留降级。
        """
        if is_apple_silicon():
            return [EngineType.MLX, EngineType.CPU]
        return [EngineType.CPU]

    def get_best_engine(
        self,
        model_key: str,
        task_type: TaskType,
        model_path: str
    ) -> BaseASREngine | BaseTTSEngine:
        """
        获取最优引擎，带自动降级
        1. 按优先级尝试创建引擎
        2. 初始化失败 → 记录日志 → 尝试下一级引擎
        3. 全部失败 → 抛出异常
        """
        # 检查是否已加载
        cache_key = f"{model_key}_{task_type.value}"
        if cache_key in self._loaded_engines:
            engine = self._loaded_engines[cache_key]
            if engine.is_loaded:
                return engine

        # 获取引擎类
        if task_type == TaskType.ASR:
            engine_class = self._asr_engines.get(model_key)
        else:
            engine_class = self._tts_engines.get(model_key)

        if engine_class is None:
            raise ValueError(f"未注册的模型引擎: {model_key} ({task_type.value})")

        # 按优先级尝试
        priorities = self.get_engine_priority(model_key, task_type)
        last_error = None

        for engine_type in priorities:
            # 检查引擎类是否支持该类型
            if engine_type not in engine_class.supported_engine_types:
                logger.debug(
                    f"引擎 {engine_class.__name__} 不支持 {engine_type.value}，跳过"
                )
                continue

            try:
                engine = engine_class(model_key, engine_type)
                engine.load(model_path)
                self._loaded_engines[cache_key] = engine
                logger.info(
                    f"引擎加载成功: model={model_key}, type={task_type.value}, "
                    f"engine={engine_type.value}"
                )
                return engine
            except Exception as e:
                last_error = e
                logger.warning(
                    f"引擎 {engine_type.value} 初始化失败: {e}，尝试降级"
                )
                continue

        raise RuntimeError(
            f"所有引擎均初始化失败，模型 {model_key} 无法加载。最后错误: {last_error}"
        )

    def unload_engine(self, model_key: str, task_type: TaskType) -> None:
        """卸载指定引擎，释放内存"""
        cache_key = f"{model_key}_{task_type.value}"
        if cache_key in self._loaded_engines:
            engine = self._loaded_engines.pop(cache_key)
            try:
                engine.unload()
                logger.info(f"引擎已卸载: model={model_key}, type={task_type.value}")
            except Exception as e:
                logger.warning(f"引擎卸载失败: {e}")

    def get_loaded_engine(self, model_key: str, task_type: TaskType):
        """获取已加载的引擎实例（未加载返回 None）"""
        cache_key = f"{model_key}_{task_type.value}"
        return self._loaded_engines.get(cache_key)

    def unload_all(self) -> None:
        """卸载所有已加载引擎"""
        keys = list(self._loaded_engines.keys())
        for key in keys:
            engine = self._loaded_engines.pop(key)
            try:
                engine.unload()
            except Exception:
                pass
        logger.info(f"已卸载所有引擎，共 {len(keys)} 个")

    def get_loaded_engines(self) -> List[str]:
        """获取当前已加载的引擎列表"""
        return list(self._loaded_engines.keys())

    def get_memory_usage(self) -> dict:
        """获取内存占用情况（用于性能监控）"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_mb": int(mem.total / (1024 * 1024)),
                "available_mb": int(mem.available / (1024 * 1024)),
                "used_percent": mem.percent,
                "loaded_engines": len(self._loaded_engines),
            }
        except ImportError:
            return {
                "total_mb": self._hardware.total_memory_mb,
                "available_mb": self._hardware.available_memory_mb,
                "used_percent": 0,
                "loaded_engines": len(self._loaded_engines),
            }

    @property
    def platform(self) -> Platform:
        return self._platform

    @property
    def arch(self) -> Arch:
        return self._arch

    @property
    def hardware(self) -> HardwareInfo:
        return self._hardware


# 全局调度器实例
scheduler = EngineScheduler()
