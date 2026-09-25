"""
平台检测 — macOS (Apple Silicon) only
MLX 推理框架硬依赖 arm64 + Apple Silicon，不支持 x86_64 / Windows / Linux
"""
import platform
import sys
from enum import Enum


class Platform(str, Enum):
    """平台枚举"""
    MACOS = "macos"
    UNKNOWN = "unknown"


class Arch(str, Enum):
    """架构枚举"""
    ARM64 = "arm64"      # Apple Silicon
    UNKNOWN = "unknown"


def detect_platform() -> Platform:
    """检测当前操作系统"""
    if platform.system().lower() == "darwin":
        return Platform.MACOS
    return Platform.UNKNOWN


def detect_arch() -> Arch:
    """检测当前 CPU 架构"""
    if platform.machine().lower() in ("arm64", "aarch64"):
        return Arch.ARM64
    return Arch.UNKNOWN


def is_apple_silicon() -> bool:
    """是否为 Apple Silicon（macOS + arm64）"""
    return detect_platform() == Platform.MACOS and detect_arch() == Arch.ARM64


def assert_supported() -> None:
    """断言当前环境为 voxNest 唯一受支持的平台"""
    if not is_apple_silicon():
        raise RuntimeError(
            f"voxNest 仅支持 Apple Silicon macOS (arm64)。"
            f"当前环境: {platform.system()} {platform.machine()}。"
            f"MLX 推理框架在此平台不可用。"
        )


def get_python_version() -> str:
    """获取 Python 版本"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
