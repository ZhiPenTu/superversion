"""
视频处理器单元测试

测试 VideoProcessor 类的各种功能，包括文件验证、信息提取和错误处理。
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import cv2
import numpy as np
from pathlib import Path

from processors.video_processor import VideoProcessor
from models.exceptions import (
    FileNotFoundError,
    UnsupportedFormatError,
    CorruptedFileError,
    VideoProcessingError
)
from models.data_models import VideoInfo


class TestVideoProcessor(unittest.TestCase):
    """视频处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self.invalid_path = os.path.join(self.temp_dir, "nonexistent.mp4")
        self.unsupported_path = os.path.join(self.temp_dir, "test.txt")
        
        # 创建一个假的文本文件用于格式测试
        with open(self.unsupported_path, 'w') as f:
            f.write("not a video file")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_file_exists_valid_file(self):
        """测试文件存在性检查 - 有效文件"""
        # 创建一个空文件
        with open(self.valid_video_path, 'w') as f:
            f.write("")
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertTrue(processor.check_file_exists())
    
    def test_check_file_exists_invalid_file(self):
        """测试文件存在性检查 - 不存在的文件"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.invalid_path)
            self.assertFalse(processor.check_file_exists())
    
    def test_check_file_format_supported(self):
        """测试文件格式检查 - 支持的格式"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertTrue(processor.check_file_format())
    
    def test_check_file_format_unsupported(self):
        """测试文件格式检查 - 不支持的格式"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.unsupported_path)
            self.assertFalse(processor.check_file_format())
    
    @patch('cv2.VideoCapture')
    def test_check_file_integrity_valid(self, mock_video_capture):
        """测试文件完整性检查 - 有效文件"""
        # 模拟有效的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertTrue(processor.check_file_integrity())
            mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_check_file_integrity_invalid(self, mock_video_capture):
        """测试文件完整性检查 - 无效文件"""
        # 模拟无法打开的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertFalse(processor.check_file_integrity())
    
    @patch('cv2.VideoCapture')
    def test_check_file_integrity_invalid_properties(self, mock_video_capture):
        """测试文件完整性检查 - 无效属性"""
        # 模拟视频文件属性无效
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 0,  # 无效帧数
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertFalse(processor.check_file_integrity())
            mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_check_file_integrity_read_failure(self, mock_video_capture):
        """测试文件完整性检查 - 读取第一帧失败"""
        # 模拟无法读取第一帧
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)  # 读取失败
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertFalse(processor.check_file_integrity())
            mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_check_file_integrity_exception(self, mock_video_capture):
        """测试文件完整性检查 - 异常处理"""
        # 模拟检查过程中抛出异常
        mock_video_capture.side_effect = Exception("OpenCV异常")
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            self.assertFalse(processor.check_file_integrity())
    
    def test_validate_video_file_not_found(self):
        """测试视频文件验证 - 文件不存在"""
        with self.assertRaises(FileNotFoundError) as context:
            VideoProcessor(self.invalid_path)
        
        self.assertIn("视频文件不存在", str(context.exception))
        self.assertEqual(context.exception.error_code, "FILE_NOT_FOUND")
    
    def test_validate_video_file_unsupported_format(self):
        """测试视频文件验证 - 不支持的格式"""
        with self.assertRaises(UnsupportedFormatError) as context:
            VideoProcessor(self.unsupported_path)
        
        self.assertIn("不支持的视频格式", str(context.exception))
        self.assertEqual(context.exception.error_code, "UNSUPPORTED_FORMAT")
    
    @patch('cv2.VideoCapture')
    def test_validate_video_file_corrupted(self, mock_video_capture):
        """测试视频文件验证 - 文件损坏"""
        # 创建文件
        with open(self.valid_video_path, 'w') as f:
            f.write("fake video content")
        
        # 模拟损坏的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with self.assertRaises(CorruptedFileError) as context:
            VideoProcessor(self.valid_video_path)
        
        self.assertIn("视频文件损坏", str(context.exception))
        self.assertEqual(context.exception.error_code, "CORRUPTED_FILE")
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_success(self, mock_video_capture):
        """测试获取视频信息 - 成功"""
        # 模拟有效的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            video_info = processor.get_video_info()
            
            self.assertIsInstance(video_info, VideoInfo)
            self.assertEqual(video_info.width, 1920)
            self.assertEqual(video_info.height, 1080)
            self.assertEqual(video_info.fps, 30.0)
            self.assertEqual(video_info.frame_count, 300)
            self.assertEqual(video_info.duration, 10.0)  # 300 frames / 30 fps
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_caching(self, mock_video_capture):
        """测试视频信息缓存功能"""
        # 模拟有效的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            # 第一次调用
            video_info1 = processor.get_video_info()
            # 第二次调用应该返回缓存的结果
            video_info2 = processor.get_video_info()
            
            # 验证返回的是同一个对象（缓存）
            self.assertIs(video_info1, video_info2)
            self.assertEqual(video_info1.width, 640)
            self.assertEqual(video_info1.height, 480)
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_zero_fps(self, mock_video_capture):
        """测试获取视频信息 - 零帧率处理"""
        # 模拟零帧率的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 0.0,  # 零帧率
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            video_info = processor.get_video_info()
            
            # 验证零帧率时时长为0
            self.assertEqual(video_info.fps, 0.0)
            self.assertEqual(video_info.duration, 0.0)
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_failure(self, mock_video_capture):
        """测试获取视频信息 - 失败"""
        # 模拟无法打开的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            
            with self.assertRaises(VideoProcessingError) as context:
                processor.get_video_info()
            
            self.assertIn("获取视频信息失败", str(context.exception))
            self.assertEqual(context.exception.error_code, "VIDEO_INFO_FAILED")
    
    @patch('cv2.VideoCapture')
    def test_read_frames_success(self, mock_video_capture):
        """测试读取视频帧 - 成功"""
        # 模拟视频帧读取
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        # 模拟读取3帧然后结束
        frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8)),
            (True, np.full((480, 640, 3), 128, dtype=np.uint8)),
            (False, None)  # 结束
        ]
        mock_cap.read.side_effect = frames
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            frame_count = 0
            for ret, frame in processor.read_frames():
                if not ret:
                    break
                frame_count += 1
                self.assertTrue(ret)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape, (480, 640, 3))
            
            self.assertEqual(frame_count, 3)
    
    @patch('cv2.VideoCapture')
    def test_read_frames_failure(self, mock_video_capture):
        """测试读取视频帧 - 失败"""
        # 模拟无法打开的视频文件
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            
            with self.assertRaises(VideoProcessingError) as context:
                list(processor.read_frames())
            
            self.assertIn("读取视频帧失败", str(context.exception))
            self.assertEqual(context.exception.error_code, "FRAME_READ_FAILED")
    
    @patch('cv2.VideoCapture')
    def test_read_frames_exception_during_read(self, mock_video_capture):
        """测试读取视频帧 - 读取过程中异常"""
        # 模拟读取过程中抛出异常
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = Exception("读取异常")
        mock_video_capture.return_value = mock_cap
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            with self.assertRaises(VideoProcessingError) as context:
                list(processor.read_frames())
            
            self.assertIn("读取视频帧失败", str(context.exception))
            self.assertIn("帧 0", str(context.exception))  # 应该显示帧号   
 
    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_create_video_writer_success(self, mock_video_capture, mock_video_writer):
        """测试创建视频写入器 - 成功"""
        # 模拟视频信息
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # 模拟视频写入器
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            output_path = os.path.join(self.temp_dir, "output.mp4")
            writer = processor.create_video_writer(output_path)
            
            self.assertEqual(writer, mock_writer)
            mock_video_writer.assert_called_once()
    
    @patch('os.makedirs')
    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_create_video_writer_with_directory_creation(self, mock_video_capture, mock_video_writer, mock_makedirs):
        """测试创建视频写入器 - 自动创建目录"""
        # 模拟视频信息
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # 模拟视频写入器
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            # 使用不存在的目录路径
            output_path = os.path.join(self.temp_dir, "new_dir", "output.mp4")
            
            with patch('os.path.exists', return_value=False):
                writer = processor.create_video_writer(output_path)
            
            # 验证目录创建被调用
            mock_makedirs.assert_called_once()
            self.assertEqual(writer, mock_writer)
    
    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_create_video_writer_custom_params(self, mock_video_capture, mock_video_writer):
        """测试创建视频写入器 - 自定义参数"""
        # 模拟视频信息
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # 模拟视频写入器
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            output_path = os.path.join(self.temp_dir, "output.mp4")
            custom_fps = 25.0
            custom_size = (640, 480)
            
            writer = processor.create_video_writer(output_path, fps=custom_fps, frame_size=custom_size)
            
            # 验证使用了自定义参数
            args, kwargs = mock_video_writer.call_args
            self.assertEqual(args[2], custom_fps)  # fps参数
            self.assertEqual(args[3], custom_size)  # frame_size参数
    
    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_create_video_writer_failure(self, mock_video_capture, mock_video_writer):
        """测试创建视频写入器 - 失败"""
        # 模拟视频信息
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # 模拟写入器创建失败
        mock_writer = Mock()
        mock_writer.isOpened.return_value = False
        mock_video_writer.return_value = mock_writer
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            processor.cap = mock_cap
            
            output_path = os.path.join(self.temp_dir, "output.mp4")
            
            with self.assertRaises(VideoProcessingError) as context:
                processor.create_video_writer(output_path)
            
            self.assertIn("无法创建视频写入器", str(context.exception))
            self.assertEqual(context.exception.error_code, "WRITER_CREATE_FAILED")
    
    @patch('platform.system')
    def test_get_platform_codec_windows(self, mock_system):
        """测试平台编解码器选择 - Windows"""
        mock_system.return_value = "Windows"
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            codec = processor._get_platform_codec()
            
            expected = cv2.VideoWriter_fourcc(*'mp4v')
            self.assertEqual(codec, expected)
    
    @patch('platform.system')
    def test_get_platform_codec_macos(self, mock_system):
        """测试平台编解码器选择 - macOS"""
        mock_system.return_value = "Darwin"
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            codec = processor._get_platform_codec()
            
            expected = cv2.VideoWriter_fourcc(*'mp4v')
            self.assertEqual(codec, expected)
    
    @patch('platform.system')
    def test_get_platform_codec_linux(self, mock_system):
        """测试平台编解码器选择 - Linux"""
        mock_system.return_value = "Linux"
        
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            codec = processor._get_platform_codec()
            
            expected = cv2.VideoWriter_fourcc(*'XVID')
            self.assertEqual(codec, expected)
    
    def test_write_frame_success(self):
        """测试写入视频帧 - 成功"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            
            # 模拟写入器
            mock_writer = Mock()
            processor.writer = mock_writer
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = processor.write_frame(frame)
            
            self.assertTrue(result)
            mock_writer.write.assert_called_once_with(frame)
    
    def test_write_frame_no_writer(self):
        """测试写入视频帧 - 写入器未初始化"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            with self.assertRaises(VideoProcessingError) as context:
                processor.write_frame(frame)
            
            self.assertIn("视频写入器未初始化", str(context.exception))
            self.assertEqual(context.exception.error_code, "WRITER_NOT_INITIALIZED")
    
    def test_write_frame_failure(self):
        """测试写入视频帧 - 写入失败"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            
            # 模拟写入器抛出异常
            mock_writer = Mock()
            mock_writer.write.side_effect = Exception("写入失败")
            processor.writer = mock_writer
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            with self.assertRaises(VideoProcessingError) as context:
                processor.write_frame(frame)
            
            self.assertIn("写入视频帧失败", str(context.exception))
            self.assertEqual(context.exception.error_code, "FRAME_WRITE_FAILED")
    
    def test_release_resources(self):
        """测试资源释放"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            processor = VideoProcessor(self.valid_video_path)
            
            # 模拟资源
            mock_cap = Mock()
            mock_writer = Mock()
            processor.cap = mock_cap
            processor.writer = mock_writer
            
            processor.release_resources()
            
            mock_cap.release.assert_called_once()
            mock_writer.release.assert_called_once()
            self.assertIsNone(processor.cap)
            self.assertIsNone(processor.writer)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            with patch.object(VideoProcessor, 'release_resources') as mock_release:
                with VideoProcessor(self.valid_video_path) as processor:
                    self.assertIsInstance(processor, VideoProcessor)
                
                # 上下文管理器退出时应该调用 release_resources
                # 注意：可能会被调用多次（上下文管理器 + 析构函数）
                self.assertTrue(mock_release.called)
    
    def test_destructor(self):
        """测试析构函数"""
        with patch.object(VideoProcessor, 'validate_video_file'):
            with patch.object(VideoProcessor, 'release_resources') as mock_release:
                processor = VideoProcessor(self.valid_video_path)
                del processor
                
                mock_release.assert_called_once()


class TestVideoProcessorIntegration(unittest.TestCase):
    """视频处理器集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_message_formatting(self):
        """测试错误信息格式化"""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.mp4")
        
        try:
            VideoProcessor(nonexistent_path)
        except FileNotFoundError as e:
            error_str = str(e)
            self.assertIn("错误:", error_str)
            self.assertIn("错误代码:", error_str)
            self.assertIn("建议:", error_str)
            self.assertIn("FILE_NOT_FOUND", error_str)
    
    def test_supported_formats(self):
        """测试支持的视频格式"""
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
        for fmt in supported_formats:
            test_path = os.path.join(self.temp_dir, f"test{fmt}")
            with patch.object(VideoProcessor, 'validate_video_file'):
                processor = VideoProcessor(test_path)
                self.assertTrue(processor.check_file_format())


if __name__ == '__main__':
    unittest.main()