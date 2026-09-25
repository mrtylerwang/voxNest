"""
设置 API 路由
"""
from fastapi import APIRouter, Form, HTTPException
from typing import Optional
from ..config import config, DownloadSource
from ..models.manager import model_manager
from ..utils.logger import logger

router = APIRouter(prefix="/api/v1/settings", tags=["Settings"])


@router.get("")
async def get_settings():
    """获取所有设置"""
    return config.to_dict()


@router.post("")
async def update_settings(
    download_source: Optional[str] = Form(None),
    use_hf_mirror: Optional[bool] = Form(None),
    max_concurrent_downloads: Optional[int] = Form(None),
    language: Optional[str] = Form(None),
):
    """更新设置"""
    updates = {}

    if download_source is not None:
        try:
            updates["download_source"] = DownloadSource(download_source).value
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的下载源: {download_source}")

    if use_hf_mirror is not None:
        updates["use_hf_mirror"] = use_hf_mirror

    if max_concurrent_downloads is not None:
        if max_concurrent_downloads < 2 or max_concurrent_downloads > 8:
            raise HTTPException(status_code=400, detail="并发下载数必须在 2-8 之间")
        updates["max_concurrent_downloads"] = max_concurrent_downloads

    if language is not None:
        if language not in ("auto", "zh", "en"):
            raise HTTPException(status_code=400, detail=f"无效的语言: {language}")
        updates["language"] = language

    if updates:
        config.update(updates)
        logger.info(f"设置已更新: {updates}")

    return {"success": True, "settings": config.to_dict()}


@router.get("/system/status")
async def get_system_status():
    """获取系统状态（硬件、内存、已加载模型等）"""
    return model_manager.get_system_status()


@router.get("/download-source/options")
async def get_download_source_options():
    """获取下载源选项列表"""
    return {
        "options": [
            {
                "value": DownloadSource.MODELSCOPE.value,
                "label": "ModelScope",
                "description": "国内直连，速度快",
                "supports_mirror": False,
            },
            {
                "value": DownloadSource.HUGGINGFACE.value,
                "label": "HuggingFace",
                "description": "国际平台，国内需启用 HF-Mirror 加速",
                "supports_mirror": True,
                "mirror_label": "启用 HF-Mirror 加速",
                "mirror_description": "使用 hf-mirror.com 作为 HuggingFace 的镜像代理，国内用户建议开启；如有 VPN 环境建议关闭",
            },
        ]
    }
