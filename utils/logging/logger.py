# utils/logging/logger.py

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional


class LoggerSetup:
    """日志系统配置"""

    _configured = False

    @classmethod
    def setup(cls,
              console_level: str = "INFO",
              file_level: str = "DEBUG",
              log_dir: str = "logs",
              rotation: str = "10 MB",
              retention: str = "30 days",
              enable_file: bool = True):
        """
        配置日志系统

        Args:
            console_level: 控制台日志级别
            file_level: 文件日志级别
            log_dir: 日志目录
            rotation: 日志轮转大小
            retention: 日志保留时间
            enable_file: 是否启用文件日志
        """
        if cls._configured:
            return

        # 移除默认处理器
        logger.remove()

        # 控制台输出格式
        console_format = (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # 添加控制台处理器
        logger.add(
            sys.stdout,
            format=console_format,
            level=console_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )

        # 添加文件处理器
        if enable_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # 普通日志文件
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )

            logger.add(
                log_path / "app_{time:YYYY-MM-DD}.log",
                format=file_format,
                level=file_level,
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                backtrace=True,
                diagnose=True
            )

            # 错误日志单独文件
            logger.add(
                log_path / "error_{time:YYYY-MM-DD}.log",
                format=file_format,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                backtrace=True,
                diagnose=True
            )

            # 仿真专用日志
            logger.add(
                log_path / "simulation_{time:YYYY-MM-DD}.log",
                format=file_format,
                level="DEBUG",
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                filter=lambda record: "simulation" in record["extra"].get("module", "")
            )

        cls._configured = True
        logger.info("日志系统初始化完成")

    @classmethod
    def get_logger(cls, name: str):
        """获取带模块名的logger"""
        return logger.bind(module=name)


def setup_logging(**kwargs):
    """便捷函数：设置日志系统"""
    LoggerSetup.setup(**kwargs)


def get_logger(name: str):
    """便捷函数：获取logger"""
    return LoggerSetup.get_logger(name)