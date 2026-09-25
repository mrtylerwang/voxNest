"""
ASR API 路由
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import os
import tempfile
from ..models.manager import model_manager
from ..models.storage import storage
from ..engines.base import TaskType, ASRResult, ASRSegment
from ..utils.logger import logger

router = APIRouter(prefix="/api/v1/asr", tags=["ASR"])

# 上传文件大小上限（100MB），防止无界读入内存
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".mov", ".webm"}


@router.get("/models")
def get_asr_models():
    """获取 ASR 模型列表"""
    models = model_manager.get_model_list(TaskType.ASR)
    return {"models": models}


@router.post("/load")
def load_asr_model(model_key: str = Form(...)):
    """加载 ASR 模型"""
    try:
        result = model_manager.load_model(model_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"加载 ASR 模型失败: {e}")
        raise HTTPException(status_code=500, detail="加载 ASR 模型失败")


@router.post("/unload")
def unload_asr_model(model_key: str = Form(...)):
    """卸载 ASR 模型"""
    try:
        result = model_manager.unload_model(model_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    model_key: Optional[str] = Form(None),
):
    """
    ASR 转录（同步路由，FastAPI 自动放入线程池，避免阻塞事件循环）
    支持 WAV、MP3、MP4、M4A、FLAC 等格式
    """
    # 校验扩展名
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix and suffix not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {suffix}")
    if not suffix:
        suffix = ".wav"

    # 同步读取并校验大小上限
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="音频文件过大（上限 100MB）")

    # 保存上传文件到临时目录
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. 基础转录
        result = model_manager.transcribe(tmp_path, model_key)

        # 2. 按标点重新分割 segments（如果只有一个 segment）
        if len(result.segments) <= 1 and result.text:
            result = _split_by_punctuation(result)

        # 3. 重新生成 SRT（按项目规范：不写序号行）
        result.srt = _generate_srt(result.segments)

        return {
            "success": True,
            "text": result.text,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                }
                for seg in result.segments
            ],
            "srt": result.srt,
            "language": result.language,
            "emotion": result.emotion,
            "engine_used": result.engine_used,
            "processing_time": result.processing_time,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"ASR 转录失败: {e}")
        raise HTTPException(status_code=500, detail="ASR 转录失败")
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/current")
def get_current_asr_model():
    """获取当前使用的 ASR 模型"""
    current = model_manager.get_current_model(TaskType.ASR)
    return {"current_model": current}


def _split_by_punctuation(result: ASRResult) -> ASRResult:
    """
    按标点分割文本为多个 segments
    时间戳按文本长度均匀分配
    """
    if not result.text or not result.segments:
        return result

    # 获取总时长
    total_duration = result.segments[-1].end if result.segments else 0.0
    if total_duration <= 0:
        return result

    # 使用正则分割标点
    sentences = _regex_split_by_punctuation(result.text)

    if len(sentences) <= 1:
        return result

    # 按文本长度比例分配时间
    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return result

    new_segments = []
    current_time = 0.0
    for sentence in sentences:
        char_ratio = len(sentence) / total_chars
        segment_duration = total_duration * char_ratio
        new_segments.append(ASRSegment(
            text=sentence,
            start=current_time,
            end=current_time + segment_duration,
        ))
        current_time += segment_duration

    result.segments = new_segments
    logger.info(f"按标点分割为 {len(new_segments)} 个片段")
    return result


def _regex_split_by_punctuation(text: str) -> List[str]:
    """用正则按标点分割文本"""
    import re
    if not text:
        return []

    # 按中文和英文标点分割，保留标点
    pattern = r'([，。！？；：、,.!?;:])'
    parts = re.split(pattern, text)

    sentences = []
    current = ""
    for part in parts:
        if not part:
            continue
        if re.match(pattern, part):
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part

    if current.strip():
        sentences.append(current.strip())

    return sentences


def _generate_srt(segments: List[ASRSegment]) -> str:
    """生成 SRT 格式（按项目规范：只写时间轴与文本，不写序号行）"""
    srt_lines = []
    for seg in segments:
        start = _format_srt_time(seg.start)
        end = _format_srt_time(seg.end)
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg.text)
        srt_lines.append("")
    return "\n".join(srt_lines).strip()


def _format_srt_time(seconds: float) -> str:
    """格式化 SRT 时间戳"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
