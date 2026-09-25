"""
voxNest 配置管理
统一管理应用配置，支持环境变量、配置文件、默认值三级覆盖
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional
from enum import Enum


class DownloadSource(str, Enum):
    """下载源枚举"""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"


class Config:
    """应用配置单例"""

    _instance: Optional["Config"] = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """初始化配置"""
        # 项目根目录（backend/app/config.py → 上溯三级到项目根）
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # 应用数据目录
        self._app_dir = Path(os.environ.get("VOXNEST_APP_DIR", self._default_app_dir()))
        self._app_dir.mkdir(parents=True, exist_ok=True)

        # 配置文件路径
        self._config_file = self._app_dir / "config.json"

        # 加载配置文件（必须在读取 models_dir 之前）
        self._config = self._load_config()

        # 模型存储目录（环境变量优先，其次配置文件，最后默认）
        env_models_dir = os.environ.get("VOXNEST_MODELS_DIR")
        if env_models_dir:
            self._models_dir = Path(env_models_dir)
        else:
            config_models_dir = self._config.get("models_dir")
            if config_models_dir:
                self._models_dir = Path(config_models_dir)
            else:
                self._models_dir = self._app_dir / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def _default_app_dir(self) -> Path:
        """获取默认应用数据目录 — macOS only"""
        # macOS: ~/.voxNest
        return Path.home() / ".voxNest"

    def _load_config(self) -> dict:
        """加载配置文件"""
        defaults = {
            "download_source": DownloadSource.MODELSCOPE.value,
            "use_hf_mirror": False,
            "max_concurrent_downloads": 5,
            "language": "auto",  # auto/zh/en
            "window_width": 1280,
            "window_height": 820,
            "tts_streaming_default": True,  # TTS 流式合成默认开启
        }

        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                defaults.update(user_config)
            except (json.JSONDecodeError, IOError) as e:
                # logger 在 import config 时尚未就绪，用 stderr 兜底提示
                print(f"[voxNest] 警告：配置文件读取失败，使用默认配置: {e}", file=sys.stderr)

        return defaults

    def save(self) -> None:
        """保存配置到文件"""
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[voxNest] 警告：配置文件写入失败: {e}", file=sys.stderr)

    def get(self, key: str, default=None):
        """获取配置项"""
        return self._config.get(key, default)

    def set(self, key: str, value) -> None:
        """设置配置项并保存"""
        self._config[key] = value
        self.save()

    def update(self, updates: dict) -> None:
        """批量更新配置并保存"""
        self._config.update(updates)
        self.save()

    # ===== 便捷属性 =====

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def app_dir(self) -> Path:
        return self._app_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    @models_dir.setter
    def models_dir(self, path: Path) -> None:
        self._models_dir = Path(path)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self.set("models_dir", str(self._models_dir))

    @property
    def download_source(self) -> DownloadSource:
        return DownloadSource(self._config.get("download_source", "modelscope"))

    @property
    def use_hf_mirror(self) -> bool:
        return self._config.get("use_hf_mirror", False)

    @property
    def max_concurrent_downloads(self) -> int:
        return self._config.get("max_concurrent_downloads", 5)

    @property
    def language(self) -> str:
        return self._config.get("language", "auto")

    def to_dict(self) -> dict:
        """导出全部配置（用于 API 返回）"""
        result = dict(self._config)
        # 始终返回当前实际使用的模型目录（包括默认值）
        result["models_dir"] = str(self._models_dir)
        return result


# 全局配置实例
config = Config()
