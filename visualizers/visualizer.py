"""
可视化器模块

实现视频帧的物体检测结果标注和可视化功能。
使用 OpenCV 和 supervision 库进行边界框绘制和标签显示。
"""

import cv2
import numpy as np
import supervision as sv
from typing import List, Tuple, Dict, Optional
from models.interfaces import IVisualizer
from models.data_models import FrameDetections, DetectionResult


class Visualizer(IVisualizer):
    """可视化器类
    
    负责在视频帧上绘制物体检测结果，包括边界框和标签。
    支持自定义颜色方案和标注样式。
    """
    
    def __init__(self, class_names: List[str], 
                 box_thickness: int = 2,
                 text_thickness: int = 1,
                 text_scale: float = 0.6,
                 text_padding: int = 10):
        """初始化可视化器
        
        Args:
            class_names: 物体类别名称列表
            box_thickness: 边界框线条粗细
            text_thickness: 文本线条粗细
            text_scale: 文本缩放比例
            text_padding: 文本边距
        """
        self.class_names = class_names
        self.box_thickness = box_thickness
        self.text_thickness = text_thickness
        self.text_scale = text_scale
        self.text_padding = text_padding
        
        # 设置标注器
        self.setup_annotators()
        
        # 定义颜色方案
        self.setup_color_scheme()
    
    def setup_annotators(self) -> None:
        """设置边界框和标签标注器
        
        使用 supervision 库创建标注器实例，配置标注样式。
        """
        # 创建边界框标注器
        self.box_annotator = sv.BoxAnnotator(
            thickness=self.box_thickness
        )
        
        # 创建标签标注器
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=self.text_thickness,
            text_scale=self.text_scale,
            text_padding=self.text_padding
        )
    
    def setup_color_scheme(self) -> None:
        """定义标注样式和颜色方案
        
        为不同类别的物体分配不同的颜色，确保视觉区分度。
        """
        # 预定义颜色列表 (BGR格式)
        self.default_colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 128),  # 青绿色
            (128, 128, 0),  # 橄榄色
        ]
        
        # 为每个类别分配颜色
        self.class_colors: Dict[str, Tuple[int, int, int]] = {}
        for i, class_name in enumerate(self.class_names):
            color_index = i % len(self.default_colors)
            self.class_colors[class_name] = self.default_colors[color_index]
        
        # 默认颜色（用于未知类别）
        self.default_color = (128, 128, 128)  # 灰色
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """获取指定类别的颜色
        
        Args:
            class_name: 类别名称
            
        Returns:
            Tuple[int, int, int]: BGR颜色值
        """
        return self.class_colors.get(class_name, self.default_color)
    
    def create_labels(self, detections: FrameDetections) -> List[str]:
        """创建标签文本
        
        为每个检测结果生成包含类别名称和置信度的标签文本。
        
        Args:
            detections: 检测结果
            
        Returns:
            List[str]: 标签文本列表
        """
        labels = []
        for detection in detections.detections:
            # 格式化标签文本：类别名称 + 置信度百分比
            confidence_percent = int(detection.confidence * 100)
            label = f"{detection.class_name} {confidence_percent}%"
            labels.append(label)
        
        return labels
    
    def convert_to_supervision_detections(self, detections: FrameDetections) -> sv.Detections:
        """将检测结果转换为 supervision 格式
        
        Args:
            detections: 自定义格式的检测结果
            
        Returns:
            sv.Detections: supervision 格式的检测结果
        """
        if not detections.detections:
            # 返回空的检测结果
            return sv.Detections.empty()
        
        # 提取边界框坐标
        xyxy = np.array([det.bbox for det in detections.detections])
        
        # 提取置信度
        confidence = np.array([det.confidence for det in detections.detections])
        
        # 提取类别ID
        class_id = np.array([det.class_id for det in detections.detections])
        
        # 创建 supervision Detections 对象
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        return sv_detections
    
    def annotate_frame(self, frame: np.ndarray, detections: FrameDetections) -> np.ndarray:
        """在视频帧上标注检测结果
        
        在检测物体周围绘制边界框，并显示类别名称和置信度分数。
        确保标注清晰可读。
        
        Args:
            frame: 输入视频帧
            detections: 检测结果
            
        Returns:
            np.ndarray: 标注后的视频帧
        """
        # 复制帧以避免修改原始数据
        annotated_frame = frame.copy()
        
        # 如果没有检测结果，直接返回原帧
        if not detections.detections:
            return annotated_frame
        
        # 转换为 supervision 格式
        sv_detections = self.convert_to_supervision_detections(detections)
        
        # 创建标签
        labels = self.create_labels(detections)
        
        # 使用 supervision 进行标注
        # 首先绘制边界框
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections
        )
        
        # 然后添加标签
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            labels=labels
        )
        
        return annotated_frame
    
    def update_color_scheme(self, new_colors: Dict[str, Tuple[int, int, int]]) -> None:
        """更新颜色方案
        
        Args:
            new_colors: 新的颜色映射字典
        """
        self.class_colors.update(new_colors)
    
    def set_annotation_style(self, box_thickness: Optional[int] = None,
                           text_thickness: Optional[int] = None,
                           text_scale: Optional[float] = None,
                           text_padding: Optional[int] = None) -> None:
        """设置标注样式
        
        Args:
            box_thickness: 边界框线条粗细
            text_thickness: 文本线条粗细
            text_scale: 文本缩放比例
            text_padding: 文本边距
        """
        if box_thickness is not None:
            self.box_thickness = box_thickness
        if text_thickness is not None:
            self.text_thickness = text_thickness
        if text_scale is not None:
            self.text_scale = text_scale
        if text_padding is not None:
            self.text_padding = text_padding
        
        # 重新设置标注器
        self.setup_annotators()