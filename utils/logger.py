"""
日志系统模块

提供结构化日志记录功能，支持不同级别的日志输出和文件保存。
实现统一的日志格式和多种输出方式。
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerManager:
    """日志管理器
    
    负责创建和管理应用程序的日志记录器，支持控制台输出和文件保存。
    提供结构化日志记录和不同级别的日志输出功能。
    """
    
    _instance: Optional['LoggerManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'LoggerManager':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志管理器"""
        if not self._initialized:
            self._loggers: Dict[str, logging.Logger] = {}
            self._log_level = LogLevel.INFO
            self._log_file_path: Optional[str] = None
            self._console_handler: Optional[logging.Handler] = None
            self._file_handler: Optional[logging.Handler] = None
            self._formatter: Optional[logging.Formatter] = None
            self._setup_formatter()
            LoggerManager._initialized = True
    
    def _setup_formatter(self) -> None:
        """设置日志格式器"""
        # 创建详细的日志格式
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
        
        # 设置时间格式
        date_format = "%Y-%m-%d %H:%M:%S"
        
        self._formatter = logging.Formatter(
            fmt=log_format,
            datefmt=date_format
        )
    
    def configure(
        self,
        log_level: str = "INFO",
        log_file_path: Optional[str] = None,
        enable_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """配置日志系统
        
        Args:
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file_path: 日志文件路径，None表示不保存到文件
            enable_console: 是否启用控制台输出
            max_file_size: 日志文件最大大小（字节）
            backup_count: 日志文件备份数量
        """
        try:
            # 设置日志级别
            self._log_level = LogLevel(log_level.upper())
        except ValueError:
            self._log_level = LogLevel.INFO
            print(f"警告: 无效的日志级别 '{log_level}'，使用默认级别 INFO")
        
        self._log_file_path = log_file_path
        
        # 清理现有的处理器
        self._cleanup_handlers()
        
        # 设置控制台处理器
        if enable_console:
            self._setup_console_handler()
        
        # 设置文件处理器
        if log_file_path:
            self._setup_file_handler(log_file_path, max_file_size, backup_count)
        
        # 更新所有现有日志记录器的配置
        self._update_existing_loggers()
    
    def _cleanup_handlers(self) -> None:
        """清理现有的日志处理器"""
        if self._console_handler:
            self._console_handler.close()
            self._console_handler = None
        
        if self._file_handler:
            self._file_handler.close()
            self._file_handler = None
    
    def _setup_console_handler(self) -> None:
        """设置控制台日志处理器"""
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(getattr(logging, self._log_level.value))
        self._console_handler.setFormatter(self._formatter)
    
    def _setup_file_handler(
        self,
        log_file_path: str,
        max_file_size: int,
        backup_count: int
    ) -> None:
        """设置文件日志处理器
        
        Args:
            log_file_path: 日志文件路径
            max_file_size: 文件最大大小
            backup_count: 备份文件数量
        """
        try:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # 使用轮转文件处理器
            self._file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            self._file_handler.setLevel(getattr(logging, self._log_level.value))
            self._file_handler.setFormatter(self._formatter)
            
        except Exception as e:
            print(f"警告: 无法设置文件日志处理器: {e}")
            self._file_handler = None
    
    def _update_existing_loggers(self) -> None:
        """更新所有现有日志记录器的配置"""
        for logger in self._loggers.values():
            self._configure_logger(logger)
    
    def _configure_logger(self, logger: logging.Logger) -> None:
        """配置单个日志记录器
        
        Args:
            logger: 要配置的日志记录器
        """
        # 清除现有处理器
        logger.handlers.clear()
        
        # 设置日志级别
        logger.setLevel(getattr(logging, self._log_level.value))
        
        # 添加处理器
        if self._console_handler:
            logger.addHandler(self._console_handler)
        
        if self._file_handler:
            logger.addHandler(self._file_handler)
        
        # 防止日志传播到根日志记录器
        logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取日志记录器
        
        Args:
            name: 日志记录器名称，通常使用模块名
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._configure_logger(logger)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def log_system_info(self) -> None:
        """记录系统信息"""
        logger = self.get_logger("system")
        logger.info("=" * 50)
        logger.info("视频物体检测系统启动")
        logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"日志级别: {self._log_level.value}")
        
        if self._log_file_path:
            logger.info(f"日志文件: {self._log_file_path}")
        else:
            logger.info("日志文件: 未启用")
        
        logger.info("=" * 50)
    
    def log_config_info(self, config: Dict[str, Any]) -> None:
        """记录配置信息
        
        Args:
            config: 配置字典
        """
        logger = self.get_logger("config")
        logger.info("系统配置信息:")
        
        # 安全地记录配置信息（隐藏敏感信息）
        safe_config = self._sanitize_config(config)
        for key, value in safe_config.items():
            logger.info(f"  {key}: {value}")
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """清理配置信息中的敏感数据
        
        Args:
            config: 原始配置字典
            
        Returns:
            Dict[str, Any]: 清理后的配置字典
        """
        # 创建配置副本
        safe_config = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                safe_config[key] = self._sanitize_config(value)
            elif isinstance(value, (str, int, float, bool, list)):
                safe_config[key] = value
            else:
                safe_config[key] = str(type(value))
        
        return safe_config
    
    def log_performance_info(
        self,
        operation: str,
        duration: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录性能信息
        
        Args:
            operation: 操作名称
            duration: 执行时间（秒）
            additional_info: 额外信息
        """
        logger = self.get_logger("performance")
        
        info_parts = [f"操作: {operation}", f"耗时: {duration:.3f}秒"]
        
        if additional_info:
            for key, value in additional_info.items():
                info_parts.append(f"{key}: {value}")
        
        logger.info(" | ".join(info_parts))
    
    def log_error_with_context(
        self,
        error: Exception,
        context: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """记录带上下文的错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文描述
            additional_info: 额外信息
        """
        logger = self.get_logger("error")
        
        error_info = [
            f"错误类型: {type(error).__name__}",
            f"错误信息: {str(error)}",
            f"上下文: {context}"
        ]
        
        if additional_info:
            for key, value in additional_info.items():
                error_info.append(f"{key}: {value}")
        
        logger.error(" | ".join(error_info), exc_info=True)
    
    def shutdown(self) -> None:
        """关闭日志系统"""
        logger = self.get_logger("system")
        logger.info("日志系统关闭")
        
        # 关闭所有处理器
        self._cleanup_handlers()
        
        # 清理日志记录器
        for logger in self._loggers.values():
            logger.handlers.clear()
        
        self._loggers.clear()


# 全局日志管理器实例
logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return logger_manager.get_logger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file_path: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """配置日志系统的便捷函数
    
    Args:
        log_level: 日志级别
        log_file_path: 日志文件路径
        enable_console: 是否启用控制台输出
    """
    logger_manager.configure(
        log_level=log_level,
        log_file_path=log_file_path,
        enable_console=enable_console
    )


def log_system_startup() -> None:
    """记录系统启动信息的便捷函数"""
    logger_manager.log_system_info()


def log_performance(
    operation: str,
    duration: float,
    **kwargs
) -> None:
    """记录性能信息的便捷函数
    
    Args:
        operation: 操作名称
        duration: 执行时间
        **kwargs: 额外信息
    """
    logger_manager.log_performance_info(operation, duration, kwargs)


def log_error(
    error: Exception,
    context: str,
    **kwargs
) -> None:
    """记录错误信息的便捷函数
    
    Args:
        error: 异常对象
        context: 错误上下文
        **kwargs: 额外信息
    """
    logger_manager.log_error_with_context(error, context, kwargs)