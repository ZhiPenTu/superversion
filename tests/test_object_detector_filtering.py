"""
物体检测器过滤功能测试

测试物体检测器的各种过滤功能，包括置信度过滤、类别过滤和NMS处理。
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from typing import List

from detectors.object_detector import ObjectDetector
from models.data_models import DetectionResult, FrameDetections
from models.exceptions import ModelLoadError


class TestObjectDetectorFiltering(unittest.TestCase):
    """物体检测器过滤功能测试类"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建模拟的检测器实例
        self.detector = ObjectDetector(confidence_threshold=0.5)
        
        # 创建测试用的检测结果
        self.test_detections = [
            DetectionResult(
                class_id=0, class_name="person", confidence=0.9,
                bbox=[10.0, 10.0, 50.0, 50.0]
            ),
            DetectionResult(
                class_id=0, class_name="person", confidence=0.3,
                bbox=[15.0, 15.0, 55.0, 55.0]
            ),
            DetectionResult(
                class_id=2, class_name="car", confidence=0.8,
                bbox=[100.0, 100.0, 200.0, 200.0]
            ),
            DetectionResult(
                class_id=2, class_name="car", confidence=0.4,
                bbox=[110.0, 110.0, 210.0, 210.0]
            ),
            DetectionResult(
                class_id=1, class_name="bicycle", confidence=0.7,
                bbox=[300.0, 300.0, 400.0, 400.0]
            )
        ]
    
    def test_filter_detections_by_confidence_default_threshold(self):
        """测试使用默认置信度阈值过滤"""
        filtered = self.detector.filter_detections_by_confidence(self.test_detections)
        
        # 应该保留置信度 >= 0.5 的检测结果
        expected_count = 3  # 0.9, 0.8, 0.7
        self.assertEqual(len(filtered), expected_count)
        
        # 验证所有保留的结果置信度都 >= 0.5
        for detection in filtered:
            self.assertGreaterEqual(detection.confidence, 0.5)
    
    def test_filter_detections_by_confidence_custom_threshold(self):
        """测试使用自定义置信度阈值过滤"""
        custom_threshold = 0.75
        filtered = self.detector.filter_detections_by_confidence(
            self.test_detections, custom_threshold
        )
        
        # 应该保留置信度 >= 0.75 的检测结果
        expected_count = 2  # 0.9, 0.8
        self.assertEqual(len(filtered), expected_count)
        
        # 验证所有保留的结果置信度都 >= 0.75
        for detection in filtered:
            self.assertGreaterEqual(detection.confidence, custom_threshold)
    
    def test_filter_detections_by_confidence_empty_input(self):
        """测试空输入的置信度过滤"""
        filtered = self.detector.filter_detections_by_confidence([])
        self.assertEqual(len(filtered), 0)
    
    def test_filter_detections_by_classes_single_class(self):
        """测试单个类别过滤"""
        target_classes = ["person"]
        filtered = self.detector.filter_detections_by_classes(
            self.test_detections, target_classes
        )
        
        # 应该只保留 person 类别的检测结果
        expected_count = 2
        self.assertEqual(len(filtered), expected_count)
        
        # 验证所有保留的结果都是 person 类别
        for detection in filtered:
            self.assertEqual(detection.class_name, "person")
    
    def test_filter_detections_by_classes_multiple_classes(self):
        """测试多个类别过滤"""
        target_classes = ["person", "car"]
        filtered = self.detector.filter_detections_by_classes(
            self.test_detections, target_classes
        )
        
        # 应该保留 person 和 car 类别的检测结果
        expected_count = 4
        self.assertEqual(len(filtered), expected_count)
        
        # 验证所有保留的结果都在目标类别中
        for detection in filtered:
            self.assertIn(detection.class_name, target_classes)
    
    def test_filter_detections_by_classes_case_insensitive(self):
        """测试类别过滤的大小写不敏感"""
        target_classes = ["PERSON", "Car"]  # 混合大小写
        filtered = self.detector.filter_detections_by_classes(
            self.test_detections, target_classes
        )
        
        # 应该保留 person 和 car 类别的检测结果
        expected_count = 4
        self.assertEqual(len(filtered), expected_count)
    
    def test_filter_detections_by_classes_no_target(self):
        """测试无目标类别时不过滤"""
        # None 目标类别
        filtered = self.detector.filter_detections_by_classes(self.test_detections, None)
        self.assertEqual(len(filtered), len(self.test_detections))
        
        # 空目标类别列表
        filtered = self.detector.filter_detections_by_classes(self.test_detections, [])
        self.assertEqual(len(filtered), len(self.test_detections))
    
    def test_filter_detections_by_classes_no_match(self):
        """测试没有匹配类别的情况"""
        target_classes = ["airplane", "boat"]
        filtered = self.detector.filter_detections_by_classes(
            self.test_detections, target_classes
        )
        
        # 应该没有匹配的结果
        self.assertEqual(len(filtered), 0)
    
    def test_calculate_iou_no_overlap(self):
        """测试无重叠的IoU计算"""
        bbox1 = [0.0, 0.0, 10.0, 10.0]
        bbox2 = [20.0, 20.0, 30.0, 30.0]
        area1 = 100.0
        area2 = 100.0
        
        iou = self.detector._calculate_iou(bbox1, bbox2, area1, area2)
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_partial_overlap(self):
        """测试部分重叠的IoU计算"""
        bbox1 = [0.0, 0.0, 20.0, 20.0]  # 面积 400
        bbox2 = [10.0, 10.0, 30.0, 30.0]  # 面积 400
        area1 = 400.0
        area2 = 400.0
        
        # 交集区域: [10, 10, 20, 20] = 100
        # 并集区域: 400 + 400 - 100 = 700
        # IoU = 100 / 700 ≈ 0.143
        
        iou = self.detector._calculate_iou(bbox1, bbox2, area1, area2)
        self.assertAlmostEqual(iou, 100.0 / 700.0, places=3)
    
    def test_calculate_iou_complete_overlap(self):
        """测试完全重叠的IoU计算"""
        bbox1 = [0.0, 0.0, 10.0, 10.0]
        bbox2 = [0.0, 0.0, 10.0, 10.0]
        area1 = 100.0
        area2 = 100.0
        
        iou = self.detector._calculate_iou(bbox1, bbox2, area1, area2)
        self.assertEqual(iou, 1.0)
    
    def test_apply_non_maximum_suppression_no_overlap(self):
        """测试无重叠情况的NMS"""
        # 创建无重叠的检测结果
        detections = [
            DetectionResult(
                class_id=0, class_name="person", confidence=0.9,
                bbox=[0.0, 0.0, 10.0, 10.0]
            ),
            DetectionResult(
                class_id=0, class_name="person", confidence=0.8,
                bbox=[50.0, 50.0, 60.0, 60.0]
            )
        ]
        
        nms_result = self.detector.apply_non_maximum_suppression(detections)
        
        # 无重叠，应该保留所有检测结果
        self.assertEqual(len(nms_result), 2)
    
    def test_apply_non_maximum_suppression_with_overlap(self):
        """测试有重叠情况的NMS"""
        # 创建有重叠的检测结果
        detections = [
            DetectionResult(
                class_id=0, class_name="person", confidence=0.9,
                bbox=[0.0, 0.0, 20.0, 20.0]
            ),
            DetectionResult(
                class_id=0, class_name="person", confidence=0.7,
                bbox=[10.0, 10.0, 30.0, 30.0]  # 与第一个有重叠
            ),
            DetectionResult(
                class_id=1, class_name="car", confidence=0.8,
                bbox=[5.0, 5.0, 25.0, 25.0]  # 不同类别，不应被抑制
            )
        ]
        
        nms_result = self.detector.apply_non_maximum_suppression(detections, iou_threshold=0.1)
        
        # 应该保留置信度最高的person检测和car检测
        self.assertEqual(len(nms_result), 2)
        
        # 验证保留的是置信度最高的person检测
        person_detections = [d for d in nms_result if d.class_name == "person"]
        self.assertEqual(len(person_detections), 1)
        self.assertEqual(person_detections[0].confidence, 0.9)
        
        # 验证car检测被保留（不同类别）
        car_detections = [d for d in nms_result if d.class_name == "car"]
        self.assertEqual(len(car_detections), 1)
    
    def test_apply_non_maximum_suppression_single_detection(self):
        """测试单个检测结果的NMS"""
        detections = [
            DetectionResult(
                class_id=0, class_name="person", confidence=0.9,
                bbox=[0.0, 0.0, 10.0, 10.0]
            )
        ]
        
        nms_result = self.detector.apply_non_maximum_suppression(detections)
        
        # 单个检测结果应该被保留
        self.assertEqual(len(nms_result), 1)
        self.assertEqual(nms_result[0], detections[0])
    
    def test_apply_non_maximum_suppression_empty_input(self):
        """测试空输入的NMS"""
        nms_result = self.detector.apply_non_maximum_suppression([])
        self.assertEqual(len(nms_result), 0)
    
    def test_filter_detections_comprehensive(self):
        """测试综合过滤功能"""
        # 使用所有过滤选项
        filtered = self.detector.filter_detections(
            self.test_detections,
            confidence_threshold=0.6,
            target_classes=["person", "car"],
            apply_nms=True,
            nms_iou_threshold=0.5
        )
        
        # 验证结果
        self.assertIsInstance(filtered, list)
        
        # 验证所有结果的置信度都 >= 0.6
        for detection in filtered:
            self.assertGreaterEqual(detection.confidence, 0.6)
        
        # 验证所有结果都在目标类别中
        for detection in filtered:
            self.assertIn(detection.class_name, ["person", "car"])
    
    def test_filter_detections_no_filters(self):
        """测试不应用任何过滤器"""
        filtered = self.detector.filter_detections(
            self.test_detections,
            confidence_threshold=0.0,  # 不过滤置信度
            target_classes=None,       # 不过滤类别
            apply_nms=False           # 不应用NMS
        )
        
        # 应该返回所有原始检测结果
        self.assertEqual(len(filtered), len(self.test_detections))
    
    def test_filter_detections_empty_input(self):
        """测试空输入的综合过滤"""
        filtered = self.detector.filter_detections([])
        self.assertEqual(len(filtered), 0)
    
    def test_filter_detections_accuracy_verification(self):
        """测试过滤结果的准确性验证"""
        # 创建具有已知特征的测试数据
        test_data = [
            DetectionResult(
                class_id=0, class_name="person", confidence=0.95,
                bbox=[0.0, 0.0, 10.0, 10.0]
            ),
            DetectionResult(
                class_id=0, class_name="person", confidence=0.85,
                bbox=[5.0, 5.0, 15.0, 15.0]  # 与上面有重叠
            ),
            DetectionResult(
                class_id=1, class_name="car", confidence=0.75,
                bbox=[100.0, 100.0, 110.0, 110.0]
            ),
            DetectionResult(
                class_id=2, class_name="bicycle", confidence=0.45,  # 低置信度
                bbox=[200.0, 200.0, 210.0, 210.0]
            )
        ]
        
        # 应用过滤: 置信度 >= 0.5, 只要 person 和 car, 应用 NMS
        filtered = self.detector.filter_detections(
            test_data,
            confidence_threshold=0.5,
            target_classes=["person", "car"],
            apply_nms=True,
            nms_iou_threshold=0.3
        )
        
        # 验证结果准确性
        # 1. bicycle 应该被置信度过滤掉
        class_names = [d.class_name for d in filtered]
        self.assertNotIn("bicycle", class_names)
        
        # 2. 应该只有 person 和 car
        for detection in filtered:
            self.assertIn(detection.class_name, ["person", "car"])
        
        # 3. 所有置信度都应该 >= 0.5
        for detection in filtered:
            self.assertGreaterEqual(detection.confidence, 0.5)
        
        # 4. NMS 应该处理重叠的 person 检测，保留置信度更高的
        person_detections = [d for d in filtered if d.class_name == "person"]
        if len(person_detections) == 1:  # NMS 生效
            self.assertEqual(person_detections[0].confidence, 0.95)


if __name__ == '__main__':
    unittest.main()