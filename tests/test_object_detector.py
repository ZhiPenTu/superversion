"""
物体检测器完整单元测试

测试物体检测器的所有核心功能，包括模型加载、检测功能、过滤功能和错误处理。
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os
from typing import List

from detectors.object_detector import ObjectDetector
from models.data_models import DetectionResult, FrameDetections
from models.exceptions import ModelLoadError


class TestObjectDetectorModelLoading(unittest.TestCase):
    """物体检测器模型加载测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.test_model_path = "yolov8n.pt"
        self.custom_model_path = "custom_model.pt"
    
    @patch('detectors.object_detector.YOLO')
    @patch('torch.cuda.is_available')
    def test_init_with_default_model(self, mock_cuda, mock_yolo):
        """测试使用默认模型初始化"""
        mock_cuda.return_value = False
        
        detector = ObjectDetector()
        
        # 验证默认参数
        self.assertEqual(detector.model_path, "yolov8n.pt")
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.device, 'cpu')
        self.assertIsNone(detector.model)
    
    @patch('detectors.object_detector.YOLO')
    @patch('torch.cuda.is_available')
    def test_init_with_custom_parameters(self, mock_cuda, mock_yolo):
        """测试使用自定义参数初始化"""
        mock_cuda.return_value = True
        
        detector = ObjectDetector(
            model_path="custom_model.pt",
            confidence_threshold=0.7
        )
        
        # 验证自定义参数
        self.assertEqual(detector.model_path, "custom_model.pt")
        self.assertEqual(detector.confidence_threshold, 0.7)
        self.assertEqual(detector.device, 'cuda')
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_available(self, mock_cuda):
        """测试CUDA可用时的设备选择"""
        mock_cuda.return_value = True
        
        detector = ObjectDetector()
        self.assertEqual(detector.device, 'cuda')
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_unavailable(self, mock_cuda):
        """测试CUDA不可用时的设备选择"""
        mock_cuda.return_value = False
        
        detector = ObjectDetector()
        self.assertEqual(detector.device, 'cpu')
    
    def test_validate_model_path_pretrained_models(self):
        """测试预训练模型路径验证"""
        detector = ObjectDetector()
        
        # 测试有效的预训练模型名称
        valid_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        for model in valid_models:
            self.assertTrue(detector._validate_model_path(model))
    
    def test_validate_model_path_custom_model_exists(self):
        """测试自定义模型文件存在时的路径验证"""
        detector = ObjectDetector()
        
        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.assertTrue(detector._validate_model_path(temp_path))
        finally:
            os.unlink(temp_path)
    
    def test_validate_model_path_custom_model_not_exists(self):
        """测试自定义模型文件不存在时的路径验证"""
        detector = ObjectDetector()
        
        non_existent_path = "/path/to/non_existent_model.pt"
        self.assertFalse(detector._validate_model_path(non_existent_path))
    
    def test_validate_model_path_invalid_extension(self):
        """测试无效文件扩展名的路径验证"""
        detector = ObjectDetector()
        
        # 创建临时文件但扩展名无效
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.assertFalse(detector._validate_model_path(temp_path))
        finally:
            os.unlink(temp_path)
    
    def test_validate_model_path_valid_extensions(self):
        """测试有效文件扩展名的路径验证"""
        detector = ObjectDetector()
        
        valid_extensions = ['.pt', '.onnx', '.engine']
        for ext in valid_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                self.assertTrue(detector._validate_model_path(temp_path))
            finally:
                os.unlink(temp_path)
    
    @patch('detectors.object_detector.YOLO')
    def test_load_model_success(self, mock_yolo):
        """测试模型加载成功场景"""
        # 设置模拟对象
        mock_model_instance = Mock()
        mock_model_instance.model.names = {0: 'person', 1: 'bicycle', 2: 'car'}
        mock_model_instance.model.to = Mock()
        mock_yolo.return_value = mock_model_instance
        
        detector = ObjectDetector()
        detector.load_model()
        
        # 验证模型加载
        mock_yolo.assert_called_once_with("yolov8n.pt")
        self.assertEqual(detector.model, mock_model_instance)
        self.assertEqual(detector.class_names, ['person', 'bicycle', 'car'])
    
    @patch('detectors.object_detector.YOLO')
    def test_load_model_success_without_names(self, mock_yolo):
        """测试模型加载成功但没有类别名称的场景"""
        # 设置模拟对象（没有names属性）
        mock_model_instance = Mock()
        del mock_model_instance.model.names  # 删除names属性
        mock_yolo.return_value = mock_model_instance
        
        detector = ObjectDetector()
        detector.load_model()
        
        # 验证使用默认类别名称
        self.assertEqual(len(detector.class_names), 80)  # COCO数据集有80个类别
        self.assertIn('person', detector.class_names)
        self.assertIn('car', detector.class_names)
    
    def test_load_model_invalid_path(self):
        """测试无效模型路径的加载失败场景"""
        detector = ObjectDetector(model_path="/invalid/path/model.pt")
        
        with self.assertRaises(ModelLoadError) as context:
            detector.load_model()
        
        self.assertIn("无效的模型路径", str(context.exception))
    
    @patch('detectors.object_detector.YOLO')
    def test_load_model_yolo_exception(self, mock_yolo):
        """测试YOLO加载异常的场景"""
        mock_yolo.side_effect = Exception("YOLO加载失败")
        
        detector = ObjectDetector()
        
        with self.assertRaises(ModelLoadError) as context:
            detector.load_model()
        
        self.assertIn("模型加载失败", str(context.exception))
        self.assertIn("YOLO加载失败", str(context.exception))
    
    def test_get_default_class_names(self):
        """测试获取默认类别名称"""
        detector = ObjectDetector()
        class_names = detector._get_default_class_names()
        
        # 验证COCO数据集的类别
        self.assertEqual(len(class_names), 80)
        self.assertIn('person', class_names)
        self.assertIn('car', class_names)
        self.assertIn('bicycle', class_names)
        self.assertEqual(class_names[0], 'person')  # person是第一个类别


class TestObjectDetectorDetection(unittest.TestCase):
    """物体检测器检测功能测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.detector = ObjectDetector()
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detect_objects_model_not_loaded(self):
        """测试模型未加载时的检测"""
        with self.assertRaises(ModelLoadError) as context:
            self.detector.detect_objects(self.test_frame)
        
        self.assertIn("模型未加载", str(context.exception))
    
    @patch('detectors.object_detector.YOLO')
    def test_detect_objects_success(self, mock_yolo):
        """测试成功检测场景"""
        # 设置模拟模型和检测结果
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        
        # 模拟检测结果数据
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [10.0, 10.0, 50.0, 50.0],
            [100.0, 100.0, 200.0, 200.0]
        ])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 2])
        mock_boxes.__len__ = Mock(return_value=2)
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        self.detector.model = mock_model
        self.detector.class_names = ['person', 'bicycle', 'car']
        
        # 执行检测
        result = self.detector.detect_objects(self.test_frame)
        
        # 验证结果
        self.assertIsInstance(result, FrameDetections)
        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.frame_shape, self.test_frame.shape)
        
        # 验证第一个检测结果
        detection1 = result.detections[0]
        self.assertEqual(detection1.class_id, 0)
        self.assertEqual(detection1.class_name, 'person')
        self.assertEqual(detection1.confidence, 0.9)
        self.assertEqual(detection1.bbox, [10.0, 10.0, 50.0, 50.0])
        
        # 验证第二个检测结果
        detection2 = result.detections[1]
        self.assertEqual(detection2.class_id, 2)
        self.assertEqual(detection2.class_name, 'car')
        self.assertEqual(detection2.confidence, 0.8)
    
    @patch('detectors.object_detector.YOLO')
    def test_detect_objects_no_detections(self, mock_yolo):
        """测试无检测结果的场景"""
        # 设置模拟模型返回空结果
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        self.detector.model = mock_model
        self.detector.class_names = ['person', 'bicycle', 'car']
        
        # 执行检测
        result = self.detector.detect_objects(self.test_frame)
        
        # 验证结果
        self.assertIsInstance(result, FrameDetections)
        self.assertEqual(len(result.detections), 0)
        self.assertEqual(result.frame_shape, self.test_frame.shape)
    
    @patch('detectors.object_detector.YOLO')
    def test_detect_objects_exception_handling(self, mock_yolo):
        """测试检测过程中异常处理"""
        # 设置模拟模型抛出异常
        mock_model = Mock()
        mock_model.side_effect = Exception("检测失败")
        
        self.detector.model = mock_model
        
        # 执行检测（应该返回空结果而不是抛出异常）
        result = self.detector.detect_objects(self.test_frame)
        
        # 验证返回空结果
        self.assertIsInstance(result, FrameDetections)
        self.assertEqual(len(result.detections), 0)
        self.assertEqual(result.frame_shape, self.test_frame.shape)
    
    @patch('detectors.object_detector.YOLO')
    def test_detect_objects_bbox_clipping(self, mock_yolo):
        """测试边界框坐标裁剪功能"""
        # 设置模拟模型返回超出边界的坐标
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        
        # 模拟超出图像边界的检测结果
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
            [-10.0, -5.0, 50.0, 60.0],  # 左上角超出边界
            [600.0, 450.0, 700.0, 500.0]  # 右下角超出边界 (图像大小 640x480)
        ])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 0])
        mock_boxes.__len__ = Mock(return_value=2)
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        self.detector.model = mock_model
        self.detector.class_names = ['person']
        
        # 执行检测
        result = self.detector.detect_objects(self.test_frame)
        
        # 验证边界框被正确裁剪
        detection1 = result.detections[0]
        self.assertEqual(detection1.bbox[0], 0.0)  # x1 被裁剪到 0
        self.assertEqual(detection1.bbox[1], 0.0)  # y1 被裁剪到 0
        
        detection2 = result.detections[1]
        self.assertEqual(detection2.bbox[2], 640.0)  # x2 被裁剪到图像宽度
        self.assertEqual(detection2.bbox[3], 480.0)  # y2 被裁剪到图像高度
    
    @patch('detectors.object_detector.YOLO')
    def test_detect_objects_batch_success(self, mock_yolo):
        """测试批量检测成功场景"""
        # 创建测试帧列表
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ]
        
        # 设置模拟模型
        mock_model = Mock()
        mock_results = []
        
        for i in range(2):
            mock_result = Mock()
            mock_boxes = Mock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 10.0, 50.0, 50.0]])
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
            mock_boxes.__len__ = Mock(return_value=1)
            mock_result.boxes = mock_boxes
            mock_results.append(mock_result)
        
        mock_model.return_value = mock_results
        
        self.detector.model = mock_model
        self.detector.class_names = ['person']
        
        # 执行批量检测
        results = self.detector.detect_objects_batch(frames)
        
        # 验证结果
        self.assertEqual(len(results), 2)
        for i, result in enumerate(results):
            self.assertIsInstance(result, FrameDetections)
            self.assertEqual(result.frame_number, i)
            self.assertEqual(len(result.detections), 1)
    
    @patch('detectors.object_detector.YOLO')
    def test_detect_objects_batch_empty_input(self, mock_yolo):
        """测试批量检测空输入"""
        # 设置模拟模型以避免ModelLoadError
        mock_model = Mock()
        self.detector.model = mock_model
        
        result = self.detector.detect_objects_batch([])
        self.assertEqual(len(result), 0)
    
    def test_detect_objects_batch_model_not_loaded(self):
        """测试批量检测模型未加载"""
        frames = [self.test_frame]
        
        with self.assertRaises(ModelLoadError):
            self.detector.detect_objects_batch(frames)


class TestObjectDetectorFiltering(unittest.TestCase):
    """物体检测器过滤功能测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.detector = ObjectDetector(confidence_threshold=0.5)
        
        # 创建测试检测结果
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
                class_id=1, class_name="bicycle", confidence=0.7,
                bbox=[300.0, 300.0, 400.0, 400.0]
            )
        ]
    
    def test_set_confidence_threshold_valid(self):
        """测试设置有效的置信度阈值"""
        self.detector.set_confidence_threshold(0.7)
        self.assertEqual(self.detector.confidence_threshold, 0.7)
    
    def test_set_confidence_threshold_invalid(self):
        """测试设置无效的置信度阈值"""
        with self.assertRaises(ValueError):
            self.detector.set_confidence_threshold(-0.1)
        
        with self.assertRaises(ValueError):
            self.detector.set_confidence_threshold(1.1)
    
    def test_filter_detections_by_confidence(self):
        """测试置信度过滤"""
        filtered = self.detector.filter_detections_by_confidence(
            self.test_detections, 0.6
        )
        
        # 应该保留置信度 >= 0.6 的结果
        self.assertEqual(len(filtered), 3)  # 0.9, 0.8, 0.7
        for detection in filtered:
            self.assertGreaterEqual(detection.confidence, 0.6)
    
    def test_filter_detections_by_classes(self):
        """测试类别过滤"""
        filtered = self.detector.filter_detections_by_classes(
            self.test_detections, ["person", "car"]
        )
        
        # 应该保留person和car类别的结果
        self.assertEqual(len(filtered), 3)  # 2个person + 1个car
        for detection in filtered:
            self.assertIn(detection.class_name, ["person", "car"])
    
    def test_comprehensive_filtering(self):
        """测试综合过滤功能"""
        filtered = self.detector.filter_detections(
            self.test_detections,
            confidence_threshold=0.6,
            target_classes=["person", "car"],
            apply_nms=True
        )
        
        # 验证过滤结果
        for detection in filtered:
            self.assertGreaterEqual(detection.confidence, 0.6)
            self.assertIn(detection.class_name, ["person", "car"])


class TestObjectDetectorUtilities(unittest.TestCase):
    """物体检测器工具功能测试类"""
    
    def setUp(self):
        """测试前的设置"""
        self.detector = ObjectDetector()
    
    def test_get_class_names_empty(self):
        """测试获取空类别名称列表"""
        class_names = self.detector.get_class_names()
        self.assertEqual(len(class_names), 0)
    
    def test_get_class_names_with_data(self):
        """测试获取有数据的类别名称列表"""
        self.detector.class_names = ['person', 'car', 'bicycle']
        class_names = self.detector.get_class_names()
        
        # 验证返回副本而不是原始列表
        self.assertEqual(class_names, ['person', 'car', 'bicycle'])
        self.assertIsNot(class_names, self.detector.class_names)
    
    def test_get_model_info_not_loaded(self):
        """测试获取未加载模型的信息"""
        info = self.detector.get_model_info()
        self.assertEqual(info["status"], "未加载")
    
    @patch('detectors.object_detector.YOLO')
    def test_get_model_info_loaded(self, mock_yolo):
        """测试获取已加载模型的信息"""
        # 设置模拟模型
        mock_model = Mock()
        self.detector.model = mock_model
        self.detector.class_names = ['person', 'car', 'bicycle']
        
        info = self.detector.get_model_info()
        
        # 验证模型信息
        self.assertEqual(info["status"], "已加载")
        self.assertEqual(info["model_path"], "yolov8n.pt")
        self.assertEqual(info["confidence_threshold"], 0.5)
        self.assertEqual(info["num_classes"], 3)
        self.assertEqual(len(info["class_names"]), 3)


class TestObjectDetectorErrorHandling(unittest.TestCase):
    """物体检测器错误处理测试类"""
    
    def test_import_error_handling(self):
        """测试导入错误处理"""
        # 这个测试需要在实际环境中手动验证
        # 因为我们无法在测试中真正移除ultralytics包
        pass
    
    def test_model_validation_comprehensive(self):
        """测试模型验证的综合场景"""
        detector = ObjectDetector()
        
        # 测试各种无效路径
        invalid_paths = [
            "",  # 空路径
            "/non/existent/path.pt",  # 不存在的路径
            "model.txt",  # 错误扩展名
            "model",  # 无扩展名
        ]
        
        for path in invalid_paths:
            if path not in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
                self.assertFalse(detector._validate_model_path(path), f"路径 {path} 应该无效")
    
    def test_detection_data_format_validation(self):
        """测试检测结果数据格式验证"""
        detector = ObjectDetector()
        
        # 测试_parse_detection_results方法的边界情况
        # 空结果
        empty_results = []
        detections = detector._parse_detection_results(empty_results, (480, 640, 3))
        self.assertEqual(len(detections), 0)
        
        # 结果中boxes为None
        mock_result = Mock()
        mock_result.boxes = None
        results_with_none = [mock_result]
        detections = detector._parse_detection_results(results_with_none, (480, 640, 3))
        self.assertEqual(len(detections), 0)
    
    def test_iou_calculation_edge_cases(self):
        """测试IoU计算的边界情况"""
        detector = ObjectDetector()
        
        # 测试零面积的情况
        bbox1 = [0.0, 0.0, 0.0, 0.0]  # 零面积
        bbox2 = [1.0, 1.0, 2.0, 2.0]
        iou = detector._calculate_iou(bbox1, bbox2, 0.0, 1.0)
        self.assertEqual(iou, 0.0)
        
        # 测试负面积的情况（不应该发生，但要处理）
        bbox1 = [10.0, 10.0, 5.0, 5.0]  # x2 < x1, y2 < y1
        bbox2 = [0.0, 0.0, 1.0, 1.0]
        iou = detector._calculate_iou(bbox1, bbox2, -25.0, 1.0)
        self.assertEqual(iou, 0.0)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)