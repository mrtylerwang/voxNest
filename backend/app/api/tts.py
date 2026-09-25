"""
TTS API 路由
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional
import os
import tempfile
from ..models.manager import model_manager
from ..engines.base import TaskType
from ..utils.logger import logger

router = APIRouter(prefix="/api/v1/tts", tags=["TTS"])

# 参考音频大小上限（50MB），防止无界读入内存
MAX_REF_AUDIO_BYTES = 50 * 1024 * 1024
ALLOWED_REF_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@router.get("/models")
def get_tts_models():
    """获取 TTS 模型列表"""
    models = model_manager.get_model_list(TaskType.TTS)
    return {"models": models}


@router.post("/load")
def load_tts_model(model_key: str = Form(...)):
    """加载 TTS 模型"""
    try:
        result = model_manager.load_model(model_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"加载 TTS 模型失败: {e}")
        raise HTTPException(status_code=500, detail="加载 TTS 模型失败")


@router.post("/unload")
def unload_tts_model(model_key: str = Form(...)):
    """卸载 TTS 模型"""
    try:
        result = model_manager.unload_model(model_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _save_reference_audio(reference_audio: Optional[UploadFile]) -> Optional[str]:
    """同步保存参考音频并校验大小上限，返回临时路径"""
    if not reference_audio:
        return None
    suffix = os.path.splitext(reference_audio.filename or "")[1].lower()
    if suffix and suffix not in ALLOWED_REF_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的参考音频格式: {suffix}")
    content = reference_audio.file.read(MAX_REF_AUDIO_BYTES + 1)
    if len(content) > MAX_REF_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="参考音频过大（上限 50MB）")
    suffix = suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        return tmp.name


def _cleanup(path: Optional[str]) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


@router.post("/synthesize")
def synthesize(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    model_key: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    voice_description: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    """
    TTS 合成（同步路由，FastAPI 自动放入线程池）
    - text: 要合成的文本
    - model_key: 模型 key（可选，不填使用当前模型）
    - voice: 内置音色名称（可选）
    - voice_description: 音色描述（用于 VoiceDesign 模型）
    - reference_audio: 参考音频（用于声音克隆）
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    ref_audio_path = _save_reference_audio(reference_audio)

    try:
        result = model_manager.synthesize(
            text=text,
            model_key=model_key,
            voice=voice,
            reference_audio=ref_audio_path,
            voice_description=voice_description,
        )

        if os.path.exists(result.audio_path):
            # FileResponse 发送完成后由 BackgroundTask 删除合成临时 wav
            background_tasks.add_task(_cleanup, result.audio_path)
            return FileResponse(
                result.audio_path,
                media_type="audio/wav",
                filename="voxNest_tts.wav",
                headers={
                    "X-Engine": result.engine_used,
                    "X-Duration": str(result.duration),
                    "X-Processing-Time": str(result.processing_time),
                },
            )
        else:
            raise HTTPException(status_code=500, detail="合成音频文件不存在")

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS 合成失败: {e}")
        raise HTTPException(status_code=500, detail="TTS 合成失败")
    finally:
        _cleanup(ref_audio_path)


@router.get("/current")
def get_current_tts_model():
    """获取当前使用的 TTS 模型"""
    current = model_manager.get_current_model(TaskType.TTS)
    return {"current_model": current}


@router.post("/synthesize/stream")
async def synthesize_stream(
    text: str = Form(...),
    model_key: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    voice_description: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    """
    TTS 流式合成（SSE）
    - 逐块返回 16-bit PCM 音频数据（24kHz, mono, base64）
    - 前端使用 Web Audio API 播放
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    ref_audio_path = _save_reference_audio(reference_audio)

    async def generate():
        try:
            async for chunk in model_manager.synthesize_stream(
                text=text,
                model_key=model_key,
                voice=voice,
                reference_audio=ref_audio_path,
                voice_description=voice_description,
            ):
                import base64
                yield f"data: {base64.b64encode(chunk).decode()}\n\n"
            yield "data: [DONE]\n\n"
        except RuntimeError as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        except Exception as e:
            logger.error(f"TTS 流式合成失败: {e}")
            yield f"event: error\ndata: 流式合成失败\n\n"
        finally:
            _cleanup(ref_audio_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Sample-Rate": "24000",
            "X-Bits-Per-Sample": "16",
            "X-Channels": "1",
        },
    )
