"""
日志 API 路由
"""
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
from ..config import config
from ..utils.logger import logger

router = APIRouter(prefix="/api/v1/logs", tags=["Logs"])


@router.get("")
async def get_logs(
    lines: int = Query(500, ge=1, le=5000, description="返回最近的日志行数"),
    level: str = Query(None, description="按日志级别过滤（INFO/WARNING/ERROR）"),
):
    """获取最近的应用日志（从文件尾部按行读取，避免大文件全量入内存）"""
    log_file = config.app_dir / "logs" / "voxnest.log"

    if not log_file.exists():
        return {"logs": [], "total": 0}

    try:
        # 从尾部按需读行：最多读 lines*4 行（过滤后不足），仍有限
        collected = _tail_lines(log_file, max_lines=lines * 4)
    except (IOError, OSError) as e:
        logger.error(f"读取日志文件失败: {e}")
        return {"logs": [], "total": 0, "error": "读取日志失败"}

    # 过滤级别
    if level:
        level_upper = level.upper()
        filtered = [line.rstrip("\n") for line in collected if f"[{level_upper}]" in line]
    else:
        filtered = [line.rstrip("\n") for line in collected]

    recent = filtered[-lines:] if lines < len(filtered) else filtered

    return {
        "logs": recent,
        "total": len(filtered),
    }


def _tail_lines(path, max_lines: int) -> list:
    """从文件尾部读取最多 max_lines 行（分块反向读取）"""
    chunksize = 64 * 1024
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        data = b""
        while end > 0 and data.count(b"\n") <= max_lines:
            step = min(chunksize, end)
            end -= step
            f.seek(end)
            data = f.read(step) + data
        return data.decode("utf-8", errors="replace").splitlines()


@router.post("/clear")
async def clear_logs():
    """清空日志文件（清空后不再追加"已清空"一行，避免刷新后日志复现）"""
    log_file = config.app_dir / "logs" / "voxnest.log"
    try:
        if log_file.exists():
            with open(log_file, "w", encoding="utf-8"):
                pass
        return {"success": True}
    except (IOError, OSError) as e:
        logger.error(f"清空日志文件失败: {e}")
        return {"success": False, "error": "清空日志失败"}


@router.get("/export")
async def export_logs():
    """导出日志文件"""
    log_file = config.app_dir / "logs" / "voxnest.log"
    if not log_file.exists():
        return {"success": False, "error": "日志文件不存在"}
    return FileResponse(
        path=str(log_file),
        media_type="text/plain",
        filename="voxnest.log",
    )
