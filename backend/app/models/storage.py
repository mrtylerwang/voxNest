"""
模型存储管理
负责模型目录管理、已下载检测、目录迁移
"""
import os
import shutil
from pathlib import Path
from typing import Optional, List
from ..config import config
from ..utils.logger import logger
from .catalog import get_model_by_key, ModelConfig


class ModelStorage:
    """模型存储管理器"""

    def __init__(self):
        self._models_dir = config.models_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def get_model_dir(self, model_key: str) -> Path:
        """获取指定模型的存储目录"""
        return self._models_dir / model_key

    def is_model_downloaded(self, model_key: str) -> bool:
        """检查模型是否已下载（检查必需文件是否存在，且无未完成的大文件临时文件）"""
        model_config = get_model_by_key(model_key)
        if model_config is None:
            return False

        model_dir = self.get_model_dir(model_key)
        if not model_dir.exists():
            return False

        # 检查是否有未完成的大文件临时文件（.incomplete / .partial）
        # 注意：不检查 .lock 文件，因为它是下载锁文件，中断后会残留，不代表下载未完成
        for f in model_dir.rglob("*"):
            if f.is_file() and (f.name.endswith(".incomplete") or f.name.endswith(".partial")):
                return False

        # 检查必需文件
        for required_file in model_config.required_files:
            file_path = model_dir / required_file
            if not file_path.exists():
                return False

        return True

    def get_downloaded_models(self) -> List[str]:
        """获取所有已下载的模型 key 列表"""
        downloaded = []
        if not self._models_dir.exists():
            return downloaded

        for item in self._models_dir.iterdir():
            if item.is_dir():
                model_key = item.name
                if self.is_model_downloaded(model_key):
                    downloaded.append(model_key)

        return downloaded

    def get_model_size(self, model_key: str) -> int:
        """获取模型目录大小（字节）"""
        model_dir = self.get_model_dir(model_key)
        if not model_dir.exists():
            return 0

        total_size = 0
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                except OSError:
                    pass
        return total_size

    def delete_model(self, model_key: str) -> bool:
        """删除已下载的模型"""
        model_dir = self.get_model_dir(model_key)
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
                logger.info(f"模型已删除: {model_key}")
                return True
            except Exception as e:
                logger.error(f"模型删除失败: {e}")
                return False
        return False

    def change_models_dir(self, new_dir: Path, migrate: bool = True) -> dict:
        """
        修改模型存储目录
        migrate=True 时自动迁移已下载的模型文件
        返回迁移进度信息
        """
        new_dir = Path(new_dir)
        new_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "success": False,
            "migrated": 0,
            "total": 0,
            "failed": [],
            "new_dir": str(new_dir),
        }

        if migrate and self._models_dir.exists() and self._models_dir != new_dir:
            # 获取所有已下载模型
            downloaded_models = self.get_downloaded_models()
            result["total"] = len(downloaded_models)

            for model_key in downloaded_models:
                src_dir = self.get_model_dir(model_key)
                dst_dir = new_dir / model_key
                try:
                    if dst_dir.exists():
                        # 目标已存在，合并（跳过已存在的文件）
                        for item in src_dir.iterdir():
                            dst_item = dst_dir / item.name
                            if not dst_item.exists():
                                if item.is_dir():
                                    shutil.copytree(item, dst_item)
                                else:
                                    shutil.copy2(item, dst_item)
                    else:
                        shutil.copytree(src_dir, dst_dir)

                    # 验证新目录的文件完整性
                    model_config = self._catalog.get_model(model_key)
                    if model_config:
                        for required_file in model_config.required_files:
                            if not (dst_dir / required_file).exists():
                                raise RuntimeError(f"验证失败: 缺少必需文件 {required_file}")

                    # 验证通过后，删除原目录的模型文件（释放磁盘空间）
                    try:
                        shutil.rmtree(src_dir)
                        logger.info(f"原目录模型已清理: {model_key}")
                    except Exception as e:
                        logger.warning(f"清理原目录失败（不影响迁移）: {model_key}, error: {e}")

                    result["migrated"] += 1
                    logger.info(f"模型迁移成功: {model_key}")
                except Exception as e:
                    result["failed"].append({"model": model_key, "error": str(e)})
                    logger.error(f"模型迁移失败: {model_key}, error: {e}")

        # 更新配置
        self._models_dir = new_dir
        config.models_dir = new_dir
        result["success"] = True

        logger.info(
            f"模型存储目录已修改: {self._models_dir}, "
            f"迁移: {result['migrated']}/{result['total']}, "
            f"失败: {len(result['failed'])}"
        )

        return result

    def get_storage_info(self) -> dict:
        """获取存储信息"""
        total_size = 0
        model_count = 0

        if self._models_dir.exists():
            for item in self._models_dir.iterdir():
                if item.is_dir():
                    model_count += 1
                    try:
                        for root, dirs, files in os.walk(item):
                            for file in files:
                                try:
                                    total_size += (Path(root) / file).stat().st_size
                                except OSError:
                                    pass
                    except Exception:
                        pass

        # 获取磁盘可用空间
        try:
            disk_usage = shutil.disk_usage(self._models_dir)
            free_space = disk_usage.free
        except Exception:
            free_space = 0

        return {
            "models_dir": str(self._models_dir),
            "model_count": model_count,
            "total_size_bytes": total_size,
            "total_size_mb": int(total_size / (1024 * 1024)),
            "free_space_bytes": free_space,
            "free_space_mb": int(free_space / (1024 * 1024)),
        }


# 全局存储管理器实例
storage = ModelStorage()
