"""
硬件检测 — Apple Silicon macOS only
MLX 推理框架硬依赖 Metal + AMX（仅 Apple Silicon 可用）
"""
import os
import platform
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """硬件信息"""
    cpu_model: str = ""
    cpu_cores: int = 0
    total_memory_mb: int = 0
    available_memory_mb: int = 0
    gpu_type: str = ""          # apple_silicon
    gpu_model: str = ""
    gpu_memory_mb: int = 0      # 统一内存，约为总内存一半
    has_metal: bool = True      # Apple Silicon 始终有 Metal
    is_low_end: bool = False


def detect_hardware() -> HardwareInfo:
    """检测硬件信息（Apple Silicon macOS）"""
    info = HardwareInfo()

    # CPU
    try:
        info.cpu_model = platform.processor() or platform.machine()
        info.cpu_cores = os.cpu_count() or 0
    except Exception:
        pass

    # 内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.total_memory_mb = int(mem.total / (1024 * 1024))
        info.available_memory_mb = int(mem.available / (1024 * 1024))
    except ImportError:
        import subprocess
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            info.total_memory_mb = int(result.stdout.strip()) // (1024 * 1024)
        except Exception:
            info.total_memory_mb = 8192
        info.available_memory_mb = info.total_memory_mb // 2

    # GPU（Apple Silicon 统一内存）
    info.gpu_type = "apple_silicon"
    info.gpu_model = info.cpu_model  # e.g. "Apple M3 Pro"
    info.gpu_memory_mb = info.total_memory_mb // 2

    # 低配置判断
    info.is_low_end = info.total_memory_mb < 8192

    return info
