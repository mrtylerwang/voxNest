"""
voxNest 后端主入口 — Apple Silicon macOS only
统一 ASR + TTS 服务，FastAPI 应用
"""
import os
import sys
import argparse
from contextlib import asynccontextmanager

# === 平台前置校验（MLX 硬依赖 Apple Silicon arm64）===
from app.scheduler.platform import assert_supported
assert_supported()

# 版本号单一事实来源：FastAPI version 与 health 返回均读取此处
APP_VERSION = "0.8.14"

# === 下载相关环境变量（必须在 import huggingface_hub / modelscope 之前设置）===
# HuggingFace: 禁用 xet 协议（会返回 401 Unauthorized），慢网络调高超时
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
# ModelScope: 大文件并行下载
os.environ.setdefault('MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS', '4')
os.environ.setdefault('MODELSCOPE_DOWNLOAD_PARALLEL_THRESHOLD_MB', '100')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import config
from app.utils.logger import logger
from app.api import asr, tts, models, settings, logs
from app.models.manager import model_manager
from app.scheduler.scheduler import scheduler
from app.engines.registry import register_all_engines


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("=" * 60)
    logger.info("voxNest 后端服务启动")
    logger.info(f"平台: {scheduler.platform.value}, 架构: {scheduler.arch.value}")
    logger.info(f"GPU: {scheduler.hardware.gpu_type}, 内存: {scheduler.hardware.total_memory_mb}MB")
    logger.info(f"模型目录: {config.models_dir}")
    logger.info(f"下载源: {config.download_source.value}, HF-Mirror: {config.use_hf_mirror}")

    # 注册所有引擎
    register_all_engines()

    logger.info("=" * 60)

    yield

    # 关闭时
    logger.info("voxNest 后端服务关闭")
    scheduler.unload_all()


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="voxNest API",
        description="本地语音 AI 桌面应用 - 统一后端服务（ASR + TTS）",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    # CORS 中间件
    # 安全基线：禁止 allow_origins=["*"] 与 allow_credentials=True 同时使用。
    # 本服务仅监听 127.0.0.1、不使用 Cookie/凭证，因此关闭 credentials，
    # 允许 Tauri webview (tauri://localhost) 与浏览器 file:// 直连调试。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(asr.router)
    app.include_router(tts.router)
    app.include_router(models.router)
    app.include_router(settings.router)
    app.include_router(logs.router)

    # 健康检查
    @app.get("/api/v1/health")
    async def health_check():
        return {
            "status": "ok",
            "version": APP_VERSION,
            "platform": scheduler.platform.value,
            "arch": scheduler.arch.value,
            "gpu_type": scheduler.hardware.gpu_type,
        }

    return app


# 全局应用实例
app = create_app()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="voxNest 后端服务")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认 127.0.0.1，仅本地访问）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="监听端口（默认 0，自动选择空闲端口）",
    )
    args = parser.parse_args()

    import uvicorn

    # 自动选择空闲端口
    port = args.port
    if port == 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    logger.info(f"启动 voxNest 后端服务: {args.host}:{port}")

    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
