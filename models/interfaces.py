"""
核心接口定义

定义系统中各个模块的抽象接口，确保模块间的解耦和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Dict, Any
import numpy as np
from .data_models import Config, VideoInfo, FrameDetections, DetectionResult


class IConfigManager(ABC):
    """配置管理器接口
    
    定义配置管理的标准接口，支持配置文件和命令行参数处理。
    """
    
    @abstractmethod
    def load_config(self) -> Config:
        """加载配置信息
        
        Returns:
            Config: 配置对象
        """
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """获取置信度阈值
        
        Returns:
            float: 置信度阈值
        """
        pass
    
    @abstractmethod
    def get_target_classes(self) -> Optional[List[str]]:
        """获取目标检测类别
        
        Returns:
            Optional[List[str]]: 目标类别列表，None表示检测所有类别
        """
        pass


class IVideoProcessor(ABC):
    """视频处理器接口
    
    定义视频文件处理的标准接口，包括读取、验证和写入功能。
    """
    
    @abstractmethod
    def validate_video_file(self) -> bool:
        """验证视频文件有效性
        
        Returns:
            bool: 文件有效返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def get_video_info(self) -> VideoInfo:
        """获取视频信息
        
        Returns:
            VideoInfo: 视频信息对象
        """
        pass
    
    @abstractmethod
    def read_frames(self) -> Iterator[np.ndarray]:
        """读取视频帧
        
        Yields:
            np.ndarray: 视频帧数据
        """
        pass
    
    @abstractmethod
    def release_resources(self) -> None:
        """释放视频资源"""
        pass


class IObjectDetector(ABC):
    """物体检测器接口
    
    定义物体检测的标准接口，支持模型加载和检测功能。
    """
    
    @abstractmethod
    def load_model(self) -> None:
        """加载检测模型"""
        pass
    
    @abstractmethod
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """检测视频帧中的物体
        
        Args:
            frame: 输入视频帧
            
        Returns:
            FrameDetections: 检测结果
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """获取模型支持的类别名称
        
        Returns:
            List[str]: 类别名称列表
        """
        pass
    
    @abstractmethod
    def filter_detections(self, detections: List[DetectionResult], 
                         confidence_threshold: Optional[float] = None,
                         target_classes: Optional[List[str]] = None,
                         apply_nms: bool = True,
                         nms_iou_threshold: float = 0.5) -> List[DetectionResult]:
        """综合过滤检测结果
        
        Args:
            detections: 原始检测结果列表
            confidence_threshold: 置信度阈值，None时使用实例默认值
            target_classes: 目标类别名称列表，None时不过滤
            apply_nms: 是否应用非极大值抑制
            nms_iou_threshold: NMS的IoU阈值
            
        Returns:
            List[DetectionResult]: 过滤后的检测结果
        """
        pass


class IVisualizer(ABC):
    """可视化器接口
    
    定义视频标注和可视化的标准接口。
    """
    
    @abstractmethod
    def annotate_frame(self, frame: np.ndarray, detections: FrameDetections) -> np.ndarray:
        """在视频帧上标注检测结果
        
        Args:
            frame: 输入视频帧
            detections: 检测结果
            
        Returns:
            np.ndarray: 标注后的视频帧
        """
        pass
    
    @abstractmethod
    def create_labels(self, detections: FrameDetections) -> List[str]:
        """创建标签文本
        
        Args:
            detections: 检测结果
            
        Returns:
            List[str]: 标签文本列表
        """
        pass