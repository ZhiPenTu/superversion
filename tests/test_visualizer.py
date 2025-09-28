"""
可视化器单元测试

测试 Visualizer 类的标注功能、标签生成和不同检测结果的处理。
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from visualizers.visualizer import Visualizer
from models.data_models import FrameDetections, DetectionResult


class TestVisualizer:
    """可视化器测试类"""
    
    @pytest.fixture
    def sample_class_names(self):
        """测试用类别名称"""
        return ["person", "car", "bicycle", "dog"]
    
    @pytest.fixture
    def visualizer(self, sample_class_names):
        """创建测试用可视化器实例"""
        return Visualizer(sample_class_names)
    
    @pytest.fixture
    def sample_frame(self):
        """创建测试用视频帧"""
        # 创建一个 640x480 的彩色图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 填充为蓝色背景
        frame[:, :] = (100, 100, 100)
        return frame
    
    @pytest.fixture
    def sample_detections(self):
        """创建测试用检测结果"""
        detections = [
            DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.85,
                bbox=[100, 100, 200, 300]  # x1, y1, x2, y2
            ),
            DetectionResult(
                class_id=1,
                class_name="car",
                confidence=0.92,
                bbox=[300, 150, 500, 350]
            ),
            DetectionResult(
                class_id=2,
                class_name="bicycle",
                confidence=0.67,
                bbox=[50, 200, 150, 400]
            )
        ]
        
        return FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=detections,
            frame_shape=(480, 640, 3)
        )
    
    @pytest.fixture
    def empty_detections(self):
        """创建空检测结果"""
        return FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=[],
            frame_shape=(480, 640, 3)
        )
    
    def test_visualizer_initialization(self, sample_class_names):
        """测试可视化器初始化"""
        visualizer = Visualizer(sample_class_names)
        
        # 验证基本属性
        assert visualizer.class_names == sample_class_names
        assert visualizer.box_thickness == 2
        assert visualizer.text_thickness == 1
        assert visualizer.text_scale == 0.6
        assert visualizer.text_padding == 10
        
        # 验证标注器已创建
        assert visualizer.box_annotator is not None
        assert visualizer.label_annotator is not None
        
        # 验证颜色方案已设置
        assert len(visualizer.class_colors) == len(sample_class_names)
        for class_name in sample_class_names:
            assert class_name in visualizer.class_colors
    
    def test_custom_initialization_parameters(self, sample_class_names):
        """测试自定义初始化参数"""
        visualizer = Visualizer(
            sample_class_names,
            box_thickness=3,
            text_thickness=2,
            text_scale=0.8,
            text_padding=15
        )
        
        assert visualizer.box_thickness == 3
        assert visualizer.text_thickness == 2
        assert visualizer.text_scale == 0.8
        assert visualizer.text_padding == 15
    
    def test_color_scheme_setup(self, visualizer, sample_class_names):
        """测试颜色方案设置"""
        # 验证每个类别都有颜色
        for class_name in sample_class_names:
            color = visualizer.get_class_color(class_name)
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
        
        # 测试未知类别返回默认颜色
        unknown_color = visualizer.get_class_color("unknown_class")
        assert unknown_color == visualizer.default_color
    
    def test_create_labels_with_detections(self, visualizer, sample_detections):
        """测试标签生成功能 - 有检测结果"""
        labels = visualizer.create_labels(sample_detections)
        
        # 验证标签数量
        assert len(labels) == len(sample_detections.detections)
        
        # 验证标签格式
        expected_labels = [
            "person 85%",
            "car 92%",
            "bicycle 67%"
        ]
        
        assert labels == expected_labels
    
    def test_create_labels_empty_detections(self, visualizer, empty_detections):
        """测试标签生成功能 - 空检测结果"""
        labels = visualizer.create_labels(empty_detections)
        assert labels == []
    
    def test_create_labels_edge_cases(self, visualizer):
        """测试标签生成的边界情况"""
        # 测试极低置信度
        low_confidence_detection = FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=[
                DetectionResult(
                    class_id=0,
                    class_name="person",
                    confidence=0.01,  # 1%
                    bbox=[100, 100, 200, 300]
                )
            ],
            frame_shape=(480, 640, 3)
        )
        
        labels = visualizer.create_labels(low_confidence_detection)
        assert labels == ["person 1%"]
        
        # 测试极高置信度
        high_confidence_detection = FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=[
                DetectionResult(
                    class_id=0,
                    class_name="person",
                    confidence=0.999,  # 99%
                    bbox=[100, 100, 200, 300]
                )
            ],
            frame_shape=(480, 640, 3)
        )
        
        labels = visualizer.create_labels(high_confidence_detection)
        assert labels == ["person 99%"]
    
    def test_convert_to_supervision_detections(self, visualizer, sample_detections):
        """测试转换为 supervision 格式"""
        sv_detections = visualizer.convert_to_supervision_detections(sample_detections)
        
        # 验证转换结果
        assert len(sv_detections) == len(sample_detections.detections)
        
        # 验证边界框
        expected_xyxy = np.array([
            [100, 100, 200, 300],
            [300, 150, 500, 350],
            [50, 200, 150, 400]
        ])
        np.testing.assert_array_equal(sv_detections.xyxy, expected_xyxy)
        
        # 验证置信度
        expected_confidence = np.array([0.85, 0.92, 0.67])
        np.testing.assert_array_almost_equal(sv_detections.confidence, expected_confidence)
        
        # 验证类别ID
        expected_class_id = np.array([0, 1, 2])
        np.testing.assert_array_equal(sv_detections.class_id, expected_class_id)
    
    def test_convert_empty_detections(self, visualizer, empty_detections):
        """测试转换空检测结果"""
        sv_detections = visualizer.convert_to_supervision_detections(empty_detections)
        assert len(sv_detections) == 0
    
    @patch('supervision.LabelAnnotator.annotate')
    @patch('supervision.BoxAnnotator.annotate')
    def test_annotate_frame_with_detections(self, mock_box_annotate, mock_label_annotate, visualizer, sample_frame, sample_detections):
        """测试帧标注功能 - 有检测结果"""
        # 设置 mock 返回值
        mock_box_annotate.return_value = sample_frame.copy()
        mock_label_annotate.return_value = sample_frame.copy()
        
        # 执行标注
        result_frame = visualizer.annotate_frame(sample_frame, sample_detections)
        
        # 验证结果
        assert result_frame is not None
        assert result_frame.shape == sample_frame.shape
        
        # 验证 BoxAnnotator 被调用
        mock_box_annotate.assert_called_once()
        
        # 验证 BoxAnnotator 调用参数
        box_call_args = mock_box_annotate.call_args
        assert 'scene' in box_call_args.kwargs
        assert 'detections' in box_call_args.kwargs
        assert 'labels' not in box_call_args.kwargs  # BoxAnnotator 不应该有 labels 参数
        
        # 验证 LabelAnnotator 被调用
        mock_label_annotate.assert_called_once()
        
        # 验证 LabelAnnotator 调用参数
        label_call_args = mock_label_annotate.call_args
        assert 'scene' in label_call_args.kwargs
        assert 'detections' in label_call_args.kwargs
        assert 'labels' in label_call_args.kwargs
        
        # 验证标签参数
        labels = label_call_args.kwargs['labels']
        expected_labels = ["person 85%", "car 92%", "bicycle 67%"]
        assert labels == expected_labels
    
    def test_annotate_frame_empty_detections(self, visualizer, sample_frame, empty_detections):
        """测试帧标注功能 - 空检测结果"""
        result_frame = visualizer.annotate_frame(sample_frame, empty_detections)
        
        # 验证返回原帧
        np.testing.assert_array_equal(result_frame, sample_frame)
    
    def test_annotate_frame_preserves_original(self, visualizer, sample_frame, sample_detections):
        """测试标注不修改原始帧"""
        original_frame = sample_frame.copy()
        
        # 执行标注
        visualizer.annotate_frame(sample_frame, sample_detections)
        
        # 验证原始帧未被修改
        np.testing.assert_array_equal(sample_frame, original_frame)
    
    def test_update_color_scheme(self, visualizer):
        """测试更新颜色方案"""
        new_colors = {
            "person": (255, 0, 0),  # 红色
            "car": (0, 255, 0),     # 绿色
        }
        
        visualizer.update_color_scheme(new_colors)
        
        # 验证颜色已更新
        assert visualizer.get_class_color("person") == (255, 0, 0)
        assert visualizer.get_class_color("car") == (0, 255, 0)
        
        # 验证其他颜色未受影响
        bicycle_color = visualizer.get_class_color("bicycle")
        assert bicycle_color != (255, 0, 0) and bicycle_color != (0, 255, 0)
    
    def test_set_annotation_style(self, visualizer):
        """测试设置标注样式"""
        # 更新样式参数
        visualizer.set_annotation_style(
            box_thickness=5,
            text_thickness=3,
            text_scale=1.0,
            text_padding=20
        )
        
        # 验证参数已更新
        assert visualizer.box_thickness == 5
        assert visualizer.text_thickness == 3
        assert visualizer.text_scale == 1.0
        assert visualizer.text_padding == 20
    
    def test_set_annotation_style_partial(self, visualizer):
        """测试部分更新标注样式"""
        original_text_scale = visualizer.text_scale
        
        # 只更新部分参数
        visualizer.set_annotation_style(box_thickness=4)
        
        # 验证指定参数已更新
        assert visualizer.box_thickness == 4
        
        # 验证其他参数未变
        assert visualizer.text_scale == original_text_scale
    
    def test_different_detection_results_handling(self, visualizer, sample_frame):
        """测试不同检测结果的处理"""
        # 测试单个检测结果
        single_detection = FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=[
                DetectionResult(
                    class_id=0,
                    class_name="person",
                    confidence=0.75,
                    bbox=[100, 100, 200, 300]
                )
            ],
            frame_shape=(480, 640, 3)
        )
        
        result = visualizer.annotate_frame(sample_frame, single_detection)
        assert result is not None
        assert result.shape == sample_frame.shape
        
        # 测试多个重叠检测结果
        overlapping_detections = FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=[
                DetectionResult(
                    class_id=0,
                    class_name="person",
                    confidence=0.85,
                    bbox=[100, 100, 200, 300]
                ),
                DetectionResult(
                    class_id=0,
                    class_name="person",
                    confidence=0.75,
                    bbox=[110, 110, 210, 310]  # 重叠区域
                )
            ],
            frame_shape=(480, 640, 3)
        )
        
        result = visualizer.annotate_frame(sample_frame, overlapping_detections)
        assert result is not None
        assert result.shape == sample_frame.shape
    
    def test_edge_case_bbox_coordinates(self, visualizer, sample_frame):
        """测试边界框坐标的边界情况"""
        # 测试边界框在图像边缘
        edge_detection = FrameDetections(
            frame_number=1,
            timestamp=0.033,
            detections=[
                DetectionResult(
                    class_id=0,
                    class_name="person",
                    confidence=0.85,
                    bbox=[0, 0, 50, 50]  # 左上角
                ),
                DetectionResult(
                    class_id=1,
                    class_name="car",
                    confidence=0.75,
                    bbox=[590, 430, 640, 480]  # 右下角
                )
            ],
            frame_shape=(480, 640, 3)
        )
        
        result = visualizer.annotate_frame(sample_frame, edge_detection)
        assert result is not None
        assert result.shape == sample_frame.shape
    
    def test_class_names_consistency(self, visualizer):
        """测试类别名称一致性"""
        # 验证类别名称列表
        assert isinstance(visualizer.class_names, list)
        assert len(visualizer.class_names) > 0
        
        # 验证每个类别都有对应的颜色
        for class_name in visualizer.class_names:
            color = visualizer.get_class_color(class_name)
            assert color is not None
            assert isinstance(color, tuple)
            assert len(color) == 3


if __name__ == "__main__":
    pytest.main([__file__])