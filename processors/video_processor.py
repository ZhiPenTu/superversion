"""
视频处理器模块

负责视频文件的读取、验证、信息提取和写入操作。
支持 .mp4 格式的视频文件处理，提供详细的错误信息和资源管理。
集成性能优化功能，包括帧缓冲和内存管理。
"""

import os
import platform
from typing import Iterator, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

from models.data_models import VideoInfo
from models.exceptions import (
    VideoProcessingError,
    FileNotFoundError,
    UnsupportedFormatError,
    CorruptedFileError
)
from utils.performance import FrameBuffer, MemoryManager


class VideoProcessor:
    """视频处理器类
    
    负责处理视频文件的读取、验证和写入操作。
    支持 .mp4 格式的视频文件处理。
    """
    
    # 支持的视频格式
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
    
    def __init__(self, video_path: str) -> None:
        """初始化视频处理器
        
        Args:
            video_path: 视频文件路径
            
        Raises:
            VideoProcessingError: 当视频文件无效时抛出
        """
        self.video_path = str(Path(video_path).resolve())
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self._video_info: Optional[VideoInfo] = None
        
        # 验证视频文件
        self.validate_video_file()
    
    def validate_video_file(self) -> bool:
        """验证视频文件的有效性
        
        检查文件是否存在、格式是否正确、文件是否完整。
        
        Returns:
            bool: 文件有效返回 True，否则抛出异常
            
        Raises:
            FileNotFoundError: 文件不存在
            UnsupportedFormatError: 文件格式不支持
            CorruptedFileError: 文件损坏或无法读取
        """
        # 检查文件是否存在
        if not self.check_file_exists():
            raise FileNotFoundError(self.video_path)
        
        # 检查文件格式
        if not self.check_file_format():
            file_ext = Path(self.video_path).suffix.lower()
            raise UnsupportedFormatError(
                self.video_path, 
                f"当前格式: {file_ext}, 支持格式: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # 检查文件完整性
        if not self.check_file_integrity():
            raise CorruptedFileError(self.video_path)
        
        return True
    
    def check_file_exists(self) -> bool:
        """检查文件是否存在
        
        Returns:
            bool: 文件存在返回 True，否则返回 False
        """
        return os.path.isfile(self.video_path)
    
    def check_file_format(self) -> bool:
        """检查文件格式是否支持
        
        Returns:
            bool: 格式支持返回 True，否则返回 False
        """
        file_ext = Path(self.video_path).suffix.lower()
        return file_ext in self.SUPPORTED_FORMATS
    
    def check_file_integrity(self) -> bool:
        """检查文件完整性
        
        尝试打开视频文件并读取基本信息来验证文件完整性。
        
        Returns:
            bool: 文件完整返回 True，否则返回 False
        """
        try:
            # 尝试打开视频文件
            test_cap = cv2.VideoCapture(self.video_path)
            if not test_cap.isOpened():
                return False
            
            # 尝试读取基本属性
            frame_count = test_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            width = test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # 检查基本属性是否有效
            if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
                test_cap.release()
                return False
            
            # 尝试读取第一帧
            ret, frame = test_cap.read()
            test_cap.release()
            
            return ret and frame is not None
            
        except Exception:
            return False
    
    def get_video_info(self) -> VideoInfo:
        """获取视频信息
        
        提取视频的基本信息，包括分辨率、帧率、时长等。
        
        Returns:
            VideoInfo: 包含视频信息的数据类
            
        Raises:
            VideoProcessingError: 当无法获取视频信息时抛出
        """
        if self._video_info is not None:
            return self._video_info
        
        try:
            # 如果还没有打开视频，先打开
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise VideoProcessingError(
                        f"无法打开视频文件: {self.video_path}",
                        "VIDEO_OPEN_FAILED",
                        "请检查文件是否被其他程序占用"
                    )
            
            # 获取视频属性
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算时长
            duration = frame_count / fps if fps > 0 else 0
            
            # 获取编解码器信息
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            self._video_info = VideoInfo(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec
            )
            
            return self._video_info
            
        except Exception as e:
            raise VideoProcessingError(
                f"获取视频信息失败: {str(e)}",
                "VIDEO_INFO_FAILED",
                "请检查视频文件是否完整"
            )
    
    def read_frames(self) -> Iterator[Tuple[bool, Optional[np.ndarray]]]:
        """迭代读取视频帧
        
        提供视频帧的迭代器，支持逐帧处理。
        
        Yields:
            Tuple[bool, Optional[np.ndarray]]: (是否成功读取, 帧数据)
            
        Raises:
            VideoProcessingError: 当视频读取失败时抛出
        """
        frame_number = 0
        try:
            # 如果还没有打开视频，先打开
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise VideoProcessingError(
                        f"无法打开视频文件: {self.video_path}",
                        "VIDEO_OPEN_FAILED",
                        "请检查文件是否被其他程序占用"
                    )
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_number += 1
                yield ret, frame
                
        except Exception as e:
            raise VideoProcessingError(
                f"读取视频帧失败 (帧 {frame_number}): {str(e)}",
                "FRAME_READ_FAILED",
                "请检查视频文件是否完整或尝试重新打开文件"
            )
    
    def create_video_writer(self, output_path: str, fps: Optional[float] = None, 
                          frame_size: Optional[Tuple[int, int]] = None) -> cv2.VideoWriter:
        """创建视频写入器
        
        创建用于保存处理后视频的写入器，保持原始分辨率和帧率。
        
        Args:
            output_path: 输出视频文件路径
            fps: 输出视频帧率，默认使用原视频帧率
            frame_size: 输出视频尺寸 (width, height)，默认使用原视频尺寸
            
        Returns:
            cv2.VideoWriter: 视频写入器对象
            
        Raises:
            VideoProcessingError: 当无法创建写入器时抛出
        """
        try:
            # 获取视频信息
            video_info = self.get_video_info()
            
            # 使用提供的参数或默认值
            output_fps = fps if fps is not None else video_info.fps
            output_size = frame_size if frame_size is not None else (video_info.width, video_info.height)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 根据平台选择编解码器
            fourcc = self._get_platform_codec()
            
            # 创建视频写入器
            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                output_fps,
                output_size
            )
            
            if not self.writer.isOpened():
                raise VideoProcessingError(
                    f"无法创建视频写入器: {output_path}",
                    "WRITER_CREATE_FAILED",
                    "请检查输出路径是否有写入权限，或尝试不同的输出格式"
                )
            
            return self.writer
            
        except Exception as e:
            if isinstance(e, VideoProcessingError):
                raise
            raise VideoProcessingError(
                f"创建视频写入器失败: {str(e)}",
                "WRITER_CREATE_FAILED",
                "请检查输出路径和参数是否正确"
            )
    
    def _get_platform_codec(self) -> int:
        """根据平台返回最佳视频编解码器
        
        Returns:
            int: OpenCV 编解码器 fourcc 代码
        """
        system = platform.system()
        
        if system == "Windows":
            # Windows 平台推荐使用 mp4v
            return cv2.VideoWriter_fourcc(*'mp4v')
        elif system == "Darwin":  # macOS
            # macOS 平台推荐使用 mp4v
            return cv2.VideoWriter_fourcc(*'mp4v')
        else:  # Linux
            # Linux 平台推荐使用 XVID
            return cv2.VideoWriter_fourcc(*'XVID')
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """写入单帧到输出视频
        
        Args:
            frame: 要写入的帧数据
            
        Returns:
            bool: 写入成功返回 True，否则返回 False
            
        Raises:
            VideoProcessingError: 当写入器未初始化时抛出
        """
        if self.writer is None:
            raise VideoProcessingError(
                "视频写入器未初始化",
                "WRITER_NOT_INITIALIZED",
                "请先调用 create_video_writer 方法"
            )
        
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            raise VideoProcessingError(
                f"写入视频帧失败: {str(e)}",
                "FRAME_WRITE_FAILED",
                "请检查磁盘空间是否充足"
            )
    
    def release_resources(self) -> None:
        """释放视频资源
        
        释放视频读取器和写入器占用的资源。
        """
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                
        except Exception:
            # 忽略释放资源时的异常
            pass
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动释放资源"""
        self.release_resources()
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.release_resources()