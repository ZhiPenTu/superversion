"""
物体检测器实现

基于 ultralytics YOLO 模型的物体检测器，支持模型加载、GPU/CPU自动选择和错误处理。
"""

import os
import logging
from typing import List, Optional, Union
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请安装 ultralytics 库: pip install ultralytics")

from models.interfaces import IObjectDetector
from models.data_models import FrameDetections, DetectionResult
from models.exceptions import ModelLoadError


class ObjectDetector(IObjectDetector):
    """物体检测器类
    
    基于 YOLO 模型的物体检测器，支持自动设备选择和模型验证。
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """初始化物体检测器
        
        Args:
            model_path: 模型文件路径，None时使用默认模型
            confidence_threshold: 置信度阈值
            
        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path or "yolov8n.pt"  # 默认使用 YOLOv8 nano 模型
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self.device = self._select_device()
        self.class_names: List[str] = []
        
    def _select_device(self) -> str:
        """自动选择最佳设备（GPU/CPU）
        
        Returns:
            str: 设备名称 ('cuda' 或 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            self.logger.info(f"检测到 CUDA 设备，使用 GPU 加速")
        else:
            device = 'cpu'
            self.logger.info("未检测到 CUDA 设备，使用 CPU")
            
        return device
    
    def _validate_model_path(self, model_path: str) -> bool:
        """验证模型文件路径
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 路径有效返回 True
        """
        # 如果是预训练模型名称（如 yolov8n.pt），不需要检查文件存在性
        if model_path in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
            return True
            
        # 检查自定义模型文件是否存在
        if not os.path.exists(model_path):
            return False
            
        # 检查文件扩展名
        if not model_path.lower().endswith(('.pt', '.onnx', '.engine')):
            return False
            
        return True
    
    def load_model(self) -> None:
        """加载 YOLO 检测模型
        
        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        try:
            # 验证模型路径
            if not self._validate_model_path(self.model_path):
                raise ModelLoadError(f"无效的模型路径: {self.model_path}")
            
            self.logger.info(f"正在加载模型: {self.model_path}")
            
            # 加载模型
            self.model = YOLO(self.model_path)
            
            # 将模型移动到指定设备
            if hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)
            
            # 获取类别名称
            if hasattr(self.model.model, 'names'):
                self.class_names = list(self.model.model.names.values())
            else:
                # 使用 COCO 数据集的默认类别名称
                self.class_names = self._get_default_class_names()
            
            self.logger.info(f"模型加载成功，支持 {len(self.class_names)} 个类别")
            self.logger.info(f"使用设备: {self.device}")
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def _get_default_class_names(self) -> List[str]:
        """获取 COCO 数据集的默认类别名称
        
        Returns:
            List[str]: 类别名称列表
        """
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect_objects(self, frame: np.ndarray) -> FrameDetections:
        """检测视频帧中的物体
        
        Args:
            frame: 输入视频帧 (H, W, C)
            
        Returns:
            FrameDetections: 检测结果
            
        Raises:
            ModelLoadError: 模型未加载时抛出
        """
        if self.model is None:
            raise ModelLoadError("模型未加载，请先调用 load_model() 方法")
        
        try:
            # 执行检测
            results = self.model(frame, conf=self.confidence_threshold, device=self.device)
            
            # 解析检测结果
            detections = self._parse_detection_results(results, frame.shape)
            
            # 创建帧检测结果
            frame_detections = FrameDetections(
                frame_number=0,  # 这里暂时设为0，实际使用时由调用者设置
                timestamp=0.0,   # 这里暂时设为0，实际使用时由调用者设置
                detections=detections,
                frame_shape=frame.shape
            )
            
            return frame_detections
            
        except Exception as e:
            self.logger.error(f"物体检测失败: {str(e)}")
            # 返回空的检测结果而不是抛出异常，保证程序继续运行
            return FrameDetections(
                frame_number=0,
                timestamp=0.0,
                detections=[],
                frame_shape=frame.shape
            )
    
    def detect_objects_batch(self, frames: List[np.ndarray]) -> List[FrameDetections]:
        """批量检测多个视频帧中的物体（提高性能）
        
        Args:
            frames: 输入视频帧列表
            
        Returns:
            List[FrameDetections]: 检测结果列表
            
        Raises:
            ModelLoadError: 模型未加载时抛出
        """
        if self.model is None:
            raise ModelLoadError("模型未加载，请先调用 load_model() 方法")
        
        if not frames:
            return []
        
        try:
            # 批量执行检测
            results = self.model(frames, conf=self.confidence_threshold, device=self.device)
            
            # 解析每帧的检测结果
            batch_detections = []
            for i, (result, frame) in enumerate(zip(results, frames)):
                detections = self._parse_detection_results([result], frame.shape)
                
                frame_detections = FrameDetections(
                    frame_number=i,  # 使用批次中的索引作为帧号
                    timestamp=0.0,   # 这里暂时设为0，实际使用时由调用者设置
                    detections=detections,
                    frame_shape=frame.shape
                )
                batch_detections.append(frame_detections)
            
            self.logger.info(f"批量检测完成，处理了 {len(frames)} 帧")
            return batch_detections
            
        except Exception as e:
            self.logger.error(f"批量物体检测失败: {str(e)}")
            # 返回空的检测结果列表
            return [FrameDetections(
                frame_number=i,
                timestamp=0.0,
                detections=[],
                frame_shape=frame.shape
            ) for i, frame in enumerate(frames)]
    
    def _parse_detection_results(self, results, frame_shape: tuple) -> List[DetectionResult]:
        """解析检测结果
        
        Args:
            results: YOLO 检测结果
            frame_shape: 帧的形状 (H, W, C)
            
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # 提取检测信息
            if len(boxes) > 0:
                # 获取边界框坐标 (xyxy 格式)
                xyxy = boxes.xyxy.cpu().numpy()
                # 获取置信度
                confidences = boxes.conf.cpu().numpy()
                # 获取类别ID
                class_ids = boxes.cls.cpu().numpy().astype(int)
                
                # 创建检测结果
                for i in range(len(xyxy)):
                    class_id = class_ids[i]
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # 确保边界框坐标在有效范围内
                    x1, y1, x2, y2 = xyxy[i]
                    x1 = max(0, min(x1, frame_shape[1]))
                    y1 = max(0, min(y1, frame_shape[0]))
                    x2 = max(0, min(x2, frame_shape[1]))
                    y2 = max(0, min(y2, frame_shape[0]))
                    
                    detection = DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidences[i]),
                        bbox=[float(x1), float(y1), float(x2), float(y2)]  # [x1, y1, x2, y2]
                    )
                    detections.append(detection)
        
        return detections
    
    def get_class_names(self) -> List[str]:
        """获取模型支持的类别名称
        
        Returns:
            List[str]: 类别名称列表
        """
        return self.class_names.copy()
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """设置置信度阈值
        
        Args:
            threshold: 新的置信度阈值 (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("置信度阈值必须在 0.0 到 1.0 之间")
        
        self.confidence_threshold = threshold
        self.logger.info(f"置信度阈值已更新为: {threshold}")
    
    def filter_detections_by_confidence(self, detections: List[DetectionResult], 
                                       confidence_threshold: Optional[float] = None) -> List[DetectionResult]:
        """基于置信度阈值过滤检测结果
        
        Args:
            detections: 原始检测结果列表
            confidence_threshold: 置信度阈值，None时使用实例默认值
            
        Returns:
            List[DetectionResult]: 过滤后的检测结果
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        filtered_detections = []
        for detection in detections:
            if detection.confidence >= confidence_threshold:
                filtered_detections.append(detection)
        
        self.logger.debug(f"置信度过滤: {len(detections)} -> {len(filtered_detections)} (阈值: {confidence_threshold})")
        return filtered_detections
    
    def filter_detections_by_classes(self, detections: List[DetectionResult], 
                                   target_classes: Optional[List[str]] = None) -> List[DetectionResult]:
        """基于目标类别过滤检测结果
        
        Args:
            detections: 原始检测结果列表
            target_classes: 目标类别名称列表，None时不过滤
            
        Returns:
            List[DetectionResult]: 过滤后的检测结果
        """
        if target_classes is None or len(target_classes) == 0:
            return detections
        
        # 将目标类别转换为小写以便不区分大小写匹配
        target_classes_lower = [cls.lower() for cls in target_classes]
        
        filtered_detections = []
        for detection in detections:
            if detection.class_name.lower() in target_classes_lower:
                filtered_detections.append(detection)
        
        self.logger.debug(f"类别过滤: {len(detections)} -> {len(filtered_detections)} (目标类别: {target_classes})")
        return filtered_detections
    
    def apply_non_maximum_suppression(self, detections: List[DetectionResult], 
                                    iou_threshold: float = 0.5) -> List[DetectionResult]:
        """应用非极大值抑制(NMS)处理重叠检测
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值，超过此值的重叠框将被抑制
            
        Returns:
            List[DetectionResult]: NMS处理后的检测结果
        """
        if len(detections) <= 1:
            return detections
        
        # 按置信度降序排序
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        # 计算所有边界框的面积
        areas = []
        for detection in sorted_detections:
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        # NMS算法
        keep_indices = []
        suppressed = [False] * len(sorted_detections)
        
        for i in range(len(sorted_detections)):
            if suppressed[i]:
                continue
                
            keep_indices.append(i)
            
            # 计算当前框与后续框的IoU
            for j in range(i + 1, len(sorted_detections)):
                if suppressed[j]:
                    continue
                
                iou = self._calculate_iou(
                    sorted_detections[i].bbox, 
                    sorted_detections[j].bbox,
                    areas[i], 
                    areas[j]
                )
                
                # 如果IoU超过阈值且是同一类别，则抑制置信度较低的框
                if iou > iou_threshold and sorted_detections[i].class_id == sorted_detections[j].class_id:
                    suppressed[j] = True
        
        # 返回保留的检测结果
        nms_detections = [sorted_detections[i] for i in keep_indices]
        
        self.logger.debug(f"NMS处理: {len(detections)} -> {len(nms_detections)} (IoU阈值: {iou_threshold})")
        return nms_detections
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float], 
                      area1: float, area2: float) -> float:
        """计算两个边界框的IoU (Intersection over Union)
        
        Args:
            bbox1: 第一个边界框 [x1, y1, x2, y2]
            bbox2: 第二个边界框 [x1, y1, x2, y2]
            area1: 第一个边界框的面积
            area2: 第二个边界框的面积
            
        Returns:
            float: IoU值 (0.0 - 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集区域的坐标
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # 检查是否有交集
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        # 计算交集面积
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集面积
        union_area = area1 + area2 - intersection_area
        
        # 计算IoU
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def filter_detections(self, detections: List[DetectionResult], 
                         confidence_threshold: Optional[float] = None,
                         target_classes: Optional[List[str]] = None,
                         apply_nms: bool = True,
                         nms_iou_threshold: float = 0.5) -> List[DetectionResult]:
        """综合过滤检测结果
        
        按顺序应用置信度过滤、类别过滤和NMS处理。
        
        Args:
            detections: 原始检测结果列表
            confidence_threshold: 置信度阈值，None时使用实例默认值
            target_classes: 目标类别名称列表，None时不过滤
            apply_nms: 是否应用非极大值抑制
            nms_iou_threshold: NMS的IoU阈值
            
        Returns:
            List[DetectionResult]: 过滤后的检测结果
        """
        if not detections:
            return []
        
        filtered_detections = detections
        
        # 1. 置信度过滤
        filtered_detections = self.filter_detections_by_confidence(
            filtered_detections, confidence_threshold
        )
        
        # 2. 类别过滤
        filtered_detections = self.filter_detections_by_classes(
            filtered_detections, target_classes
        )
        
        # 3. 非极大值抑制
        if apply_nms and len(filtered_detections) > 1:
            filtered_detections = self.apply_non_maximum_suppression(
                filtered_detections, nms_iou_threshold
            )
        
        self.logger.info(f"检测结果过滤完成: {len(detections)} -> {len(filtered_detections)}")
        return filtered_detections
    
    def get_model_info(self) -> dict:
        """获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        if self.model is None:
            return {"status": "未加载"}
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "num_classes": len(self.class_names),
            "class_names": self.class_names[:10],  # 只显示前10个类别
            "status": "已加载"
        }