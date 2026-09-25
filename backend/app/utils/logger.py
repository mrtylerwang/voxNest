"""
voxNest 日志系统
统一日志格式，支持控制台和文件输出
"""
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from ..config import config


def setup_logger(name: str = "voxnest") -> logging.Logger:
    """设置并返回日志器"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（带大小轮转，避免无限增长）
    try:
        log_dir = config.app_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "voxnest.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError):
        pass

    return logger


# 全局日志器
logger = setup_logger()
