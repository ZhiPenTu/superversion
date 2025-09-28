"""
核心数据模型定义

包含系统中使用的核心数据类和配置模型。
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Config:
    """系统配置数据类
    
    包含所有系统配置参数，支持从配置文件和命令行参数加载。
    """
    video_path: str
    output_path: Optional[str] = None
    confidence_threshold: float = 0.5
    target_classes: Optional[List[str]] = None
    model_path: Optional[str] = None
    display_video: bool = True
    save_video: bool = False
    max_fps: int = 30
    log_level: str = "INFO"
    log_file_path: Optional[str] = None
    enable_console_log: bool = True


@dataclass
class VideoInfo:
    """视频信息数据类
    
    包含视频文件的基本信息，如分辨率、帧率、时长等。
    """
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


@dataclass
class DetectionResult:
    """检测结果数据类
    
    包含单个物体检测的结果信息。
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    
    
@dataclass
class FrameDetections:
    """帧检测结果数据类
    
    包含单帧中所有物体检测的结果。
    """
    frame_number: int
    timestamp: float
    detections: List[DetectionResult]
    frame_shape: tuple  # (height, width, channels)