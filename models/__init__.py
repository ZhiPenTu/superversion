"""
数据模型模块

包含系统中使用的核心数据类和模型定义。
"""

from .data_models import Config, VideoInfo, DetectionResult, FrameDetections
from .interfaces import IConfigManager, IVideoProcessor, IObjectDetector, IVisualizer
from .exceptions import (
    VideoObjectDetectionError,
    VideoProcessingError,
    FileNotFoundError,
    UnsupportedFormatError,
    CorruptedFileError,
    ModelLoadError,
    ConfigurationError,
    DetectionError,
    VisualizationError
)

__all__ = [
    # 数据模型
    'Config',
    'VideoInfo', 
    'DetectionResult',
    'FrameDetections',
    
    # 接口定义
    'IConfigManager',
    'IVideoProcessor',
    'IObjectDetector',
    'IVisualizer',
    
    # 异常类
    'VideoObjectDetectionError',
    'VideoProcessingError',
    'FileNotFoundError',
    'UnsupportedFormatError',
    'CorruptedFileError',
    'ModelLoadError',
    'ConfigurationError',
    'DetectionError',
    'VisualizationError'
]