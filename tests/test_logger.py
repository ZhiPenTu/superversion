"""
日志系统测试模块

测试日志系统的各种功能，包括不同级别的日志输出、文件保存和配置管理。
"""

import unittest
import tempfile
import os
import logging
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.logger import (
    LoggerManager, LogLevel, get_logger, configure_logging,
    log_system_startup, log_performance, log_error
)


class TestLoggerManager(unittest.TestCase):
    """日志管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.log_file_path = os.path.join(self.temp_dir, "test.log")
        
        # 重置日志管理器单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # 重置日志管理器
        if LoggerManager._instance:
            LoggerManager._instance.shutdown()
            LoggerManager._instance = None
            LoggerManager._initialized = False
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        manager1 = LoggerManager()
        manager2 = LoggerManager()
        
        self.assertIs(manager1, manager2)
        self.assertTrue(LoggerManager._initialized)
    
    def test_default_configuration(self):
        """测试默认配置"""
        manager = LoggerManager()
        
        self.assertEqual(manager._log_level, LogLevel.INFO)
        self.assertIsNone(manager._log_file_path)
        self.assertIsNotNone(manager._formatter)
    
    def test_configure_log_level(self):
        """测试日志级别配置"""
        manager = LoggerManager()
        
        # 测试有效的日志级别
        manager.configure(log_level="DEBUG")
        self.assertEqual(manager._log_level, LogLevel.DEBUG)
        
        manager.configure(log_level="ERROR")
        self.assertEqual(manager._log_level, LogLevel.ERROR)
        
        # 测试无效的日志级别（应该使用默认值）
        with patch('builtins.print') as mock_print:
            manager.configure(log_level="INVALID")
            self.assertEqual(manager._log_level, LogLevel.INFO)
            mock_print.assert_called_once()
    
    def test_configure_file_logging(self):
        """测试文件日志配置"""
        manager = LoggerManager()
        
        # 配置文件日志
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        self.assertEqual(manager._log_file_path, self.log_file_path)
        self.assertIsNotNone(manager._file_handler)
        self.assertIsNone(manager._console_handler)
    
    def test_configure_console_logging(self):
        """测试控制台日志配置"""
        manager = LoggerManager()
        
        # 配置控制台日志
        manager.configure(
            enable_console=True,
            log_file_path=None
        )
        
        self.assertIsNotNone(manager._console_handler)
        self.assertIsNone(manager._file_handler)
    
    def test_get_logger(self):
        """测试获取日志记录器"""
        manager = LoggerManager()
        manager.configure(log_level="DEBUG")
        
        logger1 = manager.get_logger("test_module")
        logger2 = manager.get_logger("test_module")
        logger3 = manager.get_logger("another_module")
        
        # 同名日志记录器应该是同一个实例
        self.assertIs(logger1, logger2)
        
        # 不同名的日志记录器应该是不同实例
        self.assertIsNot(logger1, logger3)
        
        # 检查日志记录器配置
        self.assertEqual(logger1.level, logging.DEBUG)
        self.assertFalse(logger1.propagate)
    
    def test_log_system_info(self):
        """测试系统信息记录"""
        manager = LoggerManager()
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        # 记录系统信息
        manager.log_system_info()
        
        # 检查日志文件内容
        self.assertTrue(os.path.exists(self.log_file_path))
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("视频物体检测系统启动", content)
            self.assertIn("启动时间", content)
            self.assertIn("日志级别", content)
    
    def test_log_config_info(self):
        """测试配置信息记录"""
        manager = LoggerManager()
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        # 测试配置
        test_config = {
            "video_path": "/path/to/video.mp4",
            "confidence_threshold": 0.5,
            "nested_config": {
                "key1": "value1",
                "key2": 123
            }
        }
        
        # 记录配置信息
        manager.log_config_info(test_config)
        
        # 检查日志文件内容
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("系统配置信息", content)
            self.assertIn("video_path", content)
            self.assertIn("confidence_threshold", content)
    
    def test_log_performance_info(self):
        """测试性能信息记录"""
        manager = LoggerManager()
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        # 记录性能信息
        manager.log_performance_info(
            operation="视频处理",
            duration=1.234,
            additional_info={
                "帧数": 100,
                "分辨率": "1920x1080"
            }
        )
        
        # 检查日志文件内容
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("操作: 视频处理", content)
            self.assertIn("耗时: 1.234秒", content)
            self.assertIn("帧数: 100", content)
            self.assertIn("分辨率: 1920x1080", content)
    
    def test_log_error_with_context(self):
        """测试错误信息记录"""
        manager = LoggerManager()
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=False,
            log_level="ERROR"
        )
        
        # 创建测试异常
        try:
            raise ValueError("测试错误")
        except ValueError as e:
            # 记录错误信息
            manager.log_error_with_context(
                error=e,
                context="测试上下文",
                additional_info={
                    "文件": "test.py",
                    "行号": 123
                }
            )
        
        # 检查日志文件内容
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("错误类型: ValueError", content)
            self.assertIn("错误信息: 测试错误", content)
            self.assertIn("上下文: 测试上下文", content)
            self.assertIn("文件: test.py", content)
    
    def test_file_rotation(self):
        """测试日志文件轮转"""
        manager = LoggerManager()
        
        # 配置小的文件大小以触发轮转
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=False,
            max_file_size=1024,  # 1KB
            backup_count=2
        )
        
        logger = manager.get_logger("test")
        
        # 写入大量日志以触发轮转
        for i in range(100):
            logger.info(f"这是一条测试日志消息 {i} " + "x" * 50)
        
        # 检查是否创建了备份文件
        backup_files = [
            f for f in os.listdir(self.temp_dir)
            if f.startswith("test.log.")
        ]
        
        # 应该有备份文件被创建
        self.assertTrue(len(backup_files) > 0)
    
    def test_cleanup_handlers(self):
        """测试处理器清理"""
        manager = LoggerManager()
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=True
        )
        
        # 确保处理器已创建
        self.assertIsNotNone(manager._console_handler)
        self.assertIsNotNone(manager._file_handler)
        
        # 清理处理器
        manager._cleanup_handlers()
        
        # 检查处理器是否被清理
        self.assertIsNone(manager._console_handler)
        self.assertIsNone(manager._file_handler)
    
    def test_shutdown(self):
        """测试日志系统关闭"""
        manager = LoggerManager()
        manager.configure(
            log_file_path=self.log_file_path,
            enable_console=True
        )
        
        # 获取一些日志记录器
        logger1 = manager.get_logger("test1")
        logger2 = manager.get_logger("test2")
        
        # 确保日志记录器有处理器
        self.assertTrue(len(logger1.handlers) > 0)
        self.assertTrue(len(logger2.handlers) > 0)
        
        # 关闭日志系统
        manager.shutdown()
        
        # 检查处理器是否被清理
        self.assertEqual(len(logger1.handlers), 0)
        self.assertEqual(len(logger2.handlers), 0)
        self.assertEqual(len(manager._loggers), 0)


class TestConvenienceFunctions(unittest.TestCase):
    """便捷函数测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file_path = os.path.join(self.temp_dir, "test.log")
        
        # 重置日志管理器单例
        LoggerManager._instance = None
        LoggerManager._initialized = False
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        if LoggerManager._instance:
            LoggerManager._instance.shutdown()
            LoggerManager._instance = None
            LoggerManager._initialized = False
    
    def test_get_logger_function(self):
        """测试 get_logger 便捷函数"""
        configure_logging(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        logger = get_logger("test_module")
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_module")
    
    def test_configure_logging_function(self):
        """测试 configure_logging 便捷函数"""
        configure_logging(
            log_level="DEBUG",
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        # Import the global logger_manager to check its state
        from utils.logger import logger_manager
        self.assertEqual(logger_manager._log_level, LogLevel.DEBUG)
        self.assertEqual(logger_manager._log_file_path, self.log_file_path)
    
    def test_log_system_startup_function(self):
        """测试 log_system_startup 便捷函数"""
        configure_logging(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        log_system_startup()
        
        # 检查日志文件内容
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("视频物体检测系统启动", content)
    
    def test_log_performance_function(self):
        """测试 log_performance 便捷函数"""
        configure_logging(
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        log_performance(
            operation="测试操作",
            duration=2.5,
            帧数=50,
            状态="成功"
        )
        
        # 检查日志文件内容
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("操作: 测试操作", content)
            self.assertIn("耗时: 2.500秒", content)
            self.assertIn("帧数: 50", content)
            self.assertIn("状态: 成功", content)
    
    def test_log_error_function(self):
        """测试 log_error 便捷函数"""
        configure_logging(
            log_level="ERROR",
            log_file_path=self.log_file_path,
            enable_console=False
        )
        
        try:
            raise RuntimeError("测试运行时错误")
        except RuntimeError as e:
            log_error(
                error=e,
                context="便捷函数测试",
                模块="test_logger",
                函数="test_log_error_function"
            )
        
        # 检查日志文件内容
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("错误类型: RuntimeError", content)
            self.assertIn("错误信息: 测试运行时错误", content)
            self.assertIn("上下文: 便捷函数测试", content)
            self.assertIn("模块: test_logger", content)


class TestLogLevelEnum(unittest.TestCase):
    """日志级别枚举测试类"""
    
    def test_log_level_values(self):
        """测试日志级别枚举值"""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
        self.assertEqual(LogLevel.INFO.value, "INFO")
        self.assertEqual(LogLevel.WARNING.value, "WARNING")
        self.assertEqual(LogLevel.ERROR.value, "ERROR")
        self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")
    
    def test_log_level_creation(self):
        """测试日志级别创建"""
        level = LogLevel("INFO")
        self.assertEqual(level, LogLevel.INFO)
        
        with self.assertRaises(ValueError):
            LogLevel("INVALID")


if __name__ == '__main__':
    unittest.main()