#!/usr/bin/env python3
"""
视频物体检测系统主程序

基于 YOLO 模型的本地视频物体识别系统，支持实时显示和视频保存功能。
集成配置管理、视频处理、物体检测和可视化模块。
"""

import sys
import os
import time
import logging
from typing import Optional
import cv2

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_manager import ConfigManager
from processors.video_processor import VideoProcessor
from detectors.object_detector import ObjectDetector
from visualizers.visualizer import Visualizer
from utils.logger import (
    configure_logging, get_logger, log_system_startup, 
    log_performance, log_error, logger_manager
)
from models.exceptions import (
    VideoObjectDetectionError,
    VideoProcessingError,
    ModelLoadError,
    ConfigurationError
)


class VideoObjectDetectionApp:
    """视频物体检测应用主类
    
    协调各个模块，处理用户输入和程序流程控制。
    实现完整的视频处理流程：配置加载 -> 模型初始化 -> 视频处理 -> 结果显示/保存。
    """
    
    def __init__(self):
        """初始化应用"""
        self.config_manager: Optional[ConfigManager] = None
        self.video_processor: Optional[VideoProcessor] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.visualizer: Optional[Visualizer] = None
        
        # 设置日志
        self.setup_logging()
        self.logger = get_logger(__name__)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detection_count': 0,
            'start_time': None,
            'end_time': None
        }
    
    def setup_logging(self) -> None:
        """设置日志系统
        
        配置结构化日志记录，支持不同级别的日志输出和文件保存。
        """
        # 配置日志系统
        log_file_path = os.path.join("logs", "video_detection.log")
        
        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 使用我们的结构化日志系统
        configure_logging(
            log_level="INFO",
            log_file_path=log_file_path,
            enable_console=True
        )
        
        # 记录系统启动信息
        log_system_startup()
        
        # 设置第三方库的日志级别
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        logging.getLogger('cv2').setLevel(logging.WARNING)
    
    def initialize_components(self, config_path: Optional[str] = None, 
                            args: Optional[list] = None) -> bool:
        """初始化所有组件
        
        按顺序初始化配置管理器、视频处理器、物体检测器和可视化器。
        
        Args:
            config_path: 配置文件路径
            args: 命令行参数列表
            
        Returns:
            bool: 初始化成功返回 True
            
        Raises:
            ConfigurationError: 配置加载失败
            ModelLoadError: 模型加载失败
            VideoProcessingError: 视频处理器初始化失败
        """
        try:
            init_start_time = time.time()
            self.logger.info("正在初始化视频物体检测系统...")
            
            # 1. 初始化配置管理器
            self.logger.info("加载配置...")
            config_start_time = time.time()
            self.config_manager = ConfigManager(config_path, args)
            config = self.config_manager.load_config()
            
            # 记录配置信息
            logger_manager.log_config_info(config.__dict__)
            log_performance("配置加载", time.time() - config_start_time)
            
            # 2. 初始化视频处理器
            self.logger.info("初始化视频处理器...")
            video_start_time = time.time()
            self.video_processor = VideoProcessor(config.video_path)
            video_info = self.video_processor.get_video_info()
            log_performance("视频处理器初始化", time.time() - video_start_time,
                          分辨率=f"{video_info.width}x{video_info.height}",
                          帧率=f"{video_info.fps:.1f}fps",
                          时长=f"{video_info.duration:.1f}s")
            
            # 3. 初始化物体检测器
            self.logger.info("加载物体检测模型...")
            model_start_time = time.time()
            self.object_detector = ObjectDetector(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold
            )
            self.object_detector.load_model()
            model_info = self.object_detector.get_model_info()
            log_performance("模型加载", time.time() - model_start_time,
                          类别数=model_info['num_classes'],
                          模型路径=config.model_path or "默认模型")
            
            # 4. 初始化可视化器
            self.logger.info("初始化可视化器...")
            viz_start_time = time.time()
            class_names = self.object_detector.get_class_names()
            self.visualizer = Visualizer(class_names)
            log_performance("可视化器初始化", time.time() - viz_start_time,
                          支持类别数=len(class_names))
            
            # 更新统计信息
            self.stats['total_frames'] = video_info.frame_count
            
            # 记录总初始化时间
            total_init_time = time.time() - init_start_time
            log_performance("系统初始化", total_init_time,
                          总帧数=self.stats['total_frames'])
            
            self.logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            log_error(e, "系统初始化", 
                     配置文件=config_path or "无",
                     视频路径=getattr(config, 'video_path', '未知') if 'config' in locals() else '未知')
            raise
    
    def process_video(self) -> bool:
        """处理视频的主要流程
        
        执行完整的视频处理流程：读取帧 -> 物体检测 -> 可视化标注 -> 显示/保存。
        
        Returns:
            bool: 处理成功返回 True
        """
        if not all([self.config_manager, self.video_processor, 
                   self.object_detector, self.visualizer]):
            raise RuntimeError("系统组件未完全初始化")
        
        config = self.config_manager.get_config()
        self.stats['start_time'] = time.time()
        
        # 记录视频处理开始
        video_info = self.video_processor.get_video_info()
        log_performance("视频处理开始", 0,
                      总帧数=self.stats['total_frames'],
                      分辨率=f"{video_info.width}x{video_info.height}",
                      原始帧率=f"{video_info.fps:.1f}fps")
        
        # 初始化视频写入器（如果需要保存）
        video_writer = None
        if config.save_video and config.output_path:
            try:
                writer_start_time = time.time()
                video_writer = self.video_processor.create_video_writer(
                    config.output_path,
                    fps=min(video_info.fps, config.max_fps)
                )
                log_performance("视频写入器创建", time.time() - writer_start_time,
                              输出路径=config.output_path,
                              输出帧率=f"{min(video_info.fps, config.max_fps):.1f}fps")
            except Exception as e:
                log_error(e, "创建视频写入器", 
                         输出路径=config.output_path,
                         原始帧率=video_info.fps,
                         目标帧率=config.max_fps)
                return False
        
        # 初始化显示窗口（如果需要显示）
        window_name = "视频物体检测"
        if config.display_video:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
            # 设置窗口大小自适应
            video_info = self.video_processor.get_video_info()
            self._setup_display_window(window_name, video_info)
            self.logger.info("实时视频显示已启动")
            self.logger.info("按 ESC 键退出程序")
        
        try:
            frame_count = 0
            fps_calculator = FPSCalculator()
            save_success = True  # 跟踪保存状态
            last_progress_report = 0  # 上次进度报告的时间
            
            # 逐帧处理视频
            for ret, frame in self.video_processor.read_frames():
                if not ret or frame is None:
                    break
                
                frame_count += 1
                current_fps = fps_calculator.update()
                
                # 物体检测
                try:
                    detections = self.object_detector.detect_objects(frame)
                    
                    # 应用过滤
                    filtered_detections = self.object_detector.filter_detections(
                        detections.detections,
                        confidence_threshold=config.confidence_threshold,
                        target_classes=config.target_classes,
                        apply_nms=True
                    )
                    
                    # 更新检测结果
                    detections.detections = filtered_detections
                    detections.frame_number = frame_count
                    detections.timestamp = frame_count / self.video_processor.get_video_info().fps
                    
                    # 更新统计信息
                    self.stats['detection_count'] += len(filtered_detections)
                    
                except Exception as e:
                    self.logger.warning(f"第 {frame_count} 帧检测失败: {str(e)}")
                    # 创建空的检测结果继续处理
                    from models.data_models import FrameDetections
                    detections = FrameDetections(
                        frame_number=frame_count,
                        timestamp=frame_count / self.video_processor.get_video_info().fps,
                        detections=[],
                        frame_shape=frame.shape
                    )
                
                # 可视化标注
                try:
                    annotated_frame = self.visualizer.annotate_frame(frame, detections)
                    
                    # 添加FPS信息到帧上
                    if config.display_video:
                        annotated_frame = self._add_info_overlay(
                            annotated_frame, frame_count, current_fps, len(detections.detections)
                        )
                    
                except Exception as e:
                    self.logger.warning(f"第 {frame_count} 帧标注失败: {str(e)}")
                    annotated_frame = frame  # 使用原始帧
                
                # 显示视频（如果启用）
                if config.display_video:
                    cv2.imshow(window_name, annotated_frame)
                    
                    # 检查用户输入
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC键
                        self.logger.info("用户按下 ESC 键，退出程序")
                        break
                
                # 保存视频帧（如果启用）
                if video_writer is not None:
                    try:
                        video_writer.write(annotated_frame)
                    except Exception as e:
                        self.logger.error(f"保存第 {frame_count} 帧失败: {str(e)}")
                        save_success = False
                
                # 更新处理进度
                self.stats['processed_frames'] = frame_count
                
                # 显示保存进度（如果正在保存视频）
                current_time = time.time()
                if (video_writer is not None and 
                    current_time - last_progress_report >= 2.0):  # 每2秒报告一次进度
                    progress_percent = (frame_count / self.stats['total_frames']) * 100
                    self.logger.info(f"保存进度: {progress_percent:.1f}% "
                                   f"({frame_count}/{self.stats['total_frames']} 帧)")
                    last_progress_report = current_time
                
                # 控制帧率，确保不低于15fps的播放体验
                if config.display_video:
                    # 确保显示帧率不低于15fps
                    display_fps = max(15.0, min(current_fps, config.max_fps))
                    fps_calculator.limit_fps(display_fps)
                
                # 定期输出进度信息
                if frame_count % 100 == 0:
                    progress = (frame_count / self.stats['total_frames']) * 100
                    self.logger.info(f"处理进度: {frame_count}/{self.stats['total_frames']} "
                                   f"({progress:.1f}%), 当前FPS: {current_fps:.1f}")
            
            self.stats['end_time'] = time.time()
            
            # 记录视频处理完成的性能信息
            total_duration = self.stats['end_time'] - self.stats['start_time']
            avg_fps = frame_count / total_duration if total_duration > 0 else 0
            
            log_performance("视频处理完成", total_duration,
                          处理帧数=frame_count,
                          总检测数=self.stats['detection_count'],
                          平均FPS=f"{avg_fps:.2f}",
                          完成率=f"{(frame_count/self.stats['total_frames']*100):.1f}%")
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("用户中断程序")
            return False
        except Exception as e:
            self.logger.error(f"视频处理过程中发生错误: {str(e)}")
            return False
        finally:
            # 清理资源
            if config.display_video:
                cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()
                
                # 提供保存成功/失败的用户反馈
                if config.save_video and config.output_path:
                    if save_success and os.path.exists(config.output_path):
                        file_size = os.path.getsize(config.output_path) / (1024 * 1024)  # MB
                        self.logger.info(f"✓ 视频保存成功!")
                        self.logger.info(f"  输出文件: {config.output_path}")
                        self.logger.info(f"  文件大小: {file_size:.1f} MB")
                        self.logger.info(f"  保存帧数: {self.stats['processed_frames']}")
                    else:
                        self.logger.error(f"✗ 视频保存失败!")
                        self.logger.error(f"  目标路径: {config.output_path}")
                        if not os.path.exists(config.output_path):
                            self.logger.error(f"  错误原因: 输出文件未创建")
                        else:
                            self.logger.error(f"  错误原因: 保存过程中发生错误")
                        self.logger.error(f"  建议: 检查输出路径权限或磁盘空间")
    
    def _setup_display_window(self, window_name: str, video_info) -> None:
        """设置显示窗口大小自适应
        
        Args:
            window_name: 窗口名称
            video_info: 视频信息对象
        """
        # 获取屏幕分辨率（简单估算）
        screen_width = 1920  # 默认屏幕宽度
        screen_height = 1080  # 默认屏幕高度
        
        # 计算合适的窗口大小
        video_width = video_info.width
        video_height = video_info.height
        
        # 如果视频尺寸超过屏幕的80%，则缩放
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        if video_width > max_width or video_height > max_height:
            # 计算缩放比例，保持宽高比
            scale_w = max_width / video_width
            scale_h = max_height / video_height
            scale = min(scale_w, scale_h)
            
            new_width = int(video_width * scale)
            new_height = int(video_height * scale)
        else:
            new_width = video_width
            new_height = video_height
        
        # 设置窗口大小
        cv2.resizeWindow(window_name, new_width, new_height)
        
        # 将窗口移动到屏幕中央
        cv2.moveWindow(window_name, 
                      (screen_width - new_width) // 2, 
                      (screen_height - new_height) // 2)
        
        self.logger.info(f"显示窗口设置: {new_width}x{new_height} "
                        f"(原始: {video_width}x{video_height})")
    
    def _add_info_overlay(self, frame, frame_number: int, fps: float, 
                         detection_count: int) -> cv2.Mat:
        """在帧上添加信息覆盖层
        
        Args:
            frame: 输入帧
            frame_number: 帧号
            fps: 当前FPS
            detection_count: 检测到的物体数量
            
        Returns:
            cv2.Mat: 添加信息后的帧
        """
        # 创建信息文本
        info_lines = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Objects: {detection_count}"
        ]
        
        # 在左上角绘制信息
        y_offset = 30
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * 25
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        return frame
    
    def print_statistics(self) -> None:
        """打印处理统计信息"""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            avg_fps = self.stats['processed_frames'] / duration if duration > 0 else 0
            
            print("\n" + "="*50)
            print("处理统计信息")
            print("="*50)
            print(f"总帧数: {self.stats['total_frames']}")
            print(f"处理帧数: {self.stats['processed_frames']}")
            print(f"检测物体总数: {self.stats['detection_count']}")
            print(f"处理时间: {duration:.2f} 秒")
            print(f"平均FPS: {avg_fps:.2f}")
            print(f"完成率: {(self.stats['processed_frames']/self.stats['total_frames']*100):.1f}%")
            print("="*50)
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.video_processor:
                self.video_processor.release_resources()
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.warning(f"清理资源时发生错误: {str(e)}")
    
    def run(self, config_path: Optional[str] = None, args: Optional[list] = None) -> int:
        """运行应用程序
        
        Args:
            config_path: 配置文件路径
            args: 命令行参数列表
            
        Returns:
            int: 退出代码 (0表示成功，非0表示失败)
        """
        try:
            # 初始化组件
            if not self.initialize_components(config_path, args):
                return 1
            
            # 处理视频
            if not self.process_video():
                return 1
            
            # 打印统计信息
            self.print_statistics()
            
            return 0
            
        except ConfigurationError as e:
            self.logger.error(f"配置错误: {str(e)}")
            return 2
        except ModelLoadError as e:
            self.logger.error(f"模型加载错误: {str(e)}")
            return 3
        except VideoProcessingError as e:
            self.logger.error(f"视频处理错误: {str(e)}")
            return 4
        except KeyboardInterrupt:
            self.logger.info("程序被用户中断")
            return 130
        except Exception as e:
            self.logger.error(f"未知错误: {str(e)}")
            return 1
        finally:
            self.cleanup()


class FPSCalculator:
    """FPS计算器
    
    用于计算和控制视频处理的帧率。
    """
    
    def __init__(self, window_size: int = 30):
        """初始化FPS计算器
        
        Args:
            window_size: 计算FPS的时间窗口大小
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """更新FPS计算
        
        Returns:
            float: 当前FPS
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # 保持窗口大小
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # 计算FPS
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                fps = (len(self.frame_times) - 1) / time_diff
                return fps
        
        return 0.0
    
    def limit_fps(self, target_fps: float) -> None:
        """限制FPS到目标值
        
        Args:
            target_fps: 目标FPS
        """
        if target_fps <= 0:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_time
        target_interval = 1.0 / target_fps
        
        if elapsed < target_interval:
            sleep_time = target_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_time = time.time()


def print_usage_examples():
    """打印使用示例"""
    print("\n视频物体检测系统 - 使用示例:")
    print("="*60)
    print("基本用法:")
    print("  python main.py --video input.mp4")
    print()
    print("设置置信度阈值:")
    print("  python main.py --video input.mp4 --confidence 0.6")
    print()
    print("保存标注后的视频:")
    print("  python main.py --video input.mp4 --save --output output.mp4")
    print()
    print("只检测特定类别:")
    print("  python main.py --video input.mp4 --classes person car bicycle")
    print()
    print("使用自定义模型:")
    print("  python main.py --video input.mp4 --model custom_model.pt")
    print()
    print("使用配置文件:")
    print("  python main.py --config config.yaml --video input.mp4")
    print()
    print("不显示实时视频窗口:")
    print("  python main.py --video input.mp4 --no-display --save")
    print()
    print("完整示例:")
    print("  python main.py --video traffic.mp4 --confidence 0.7 \\")
    print("                 --classes person car truck --save \\")
    print("                 --output annotated_traffic.mp4 --max-fps 25")
    print("="*60)


def handle_command_line_errors():
    """处理命令行参数错误"""
    if len(sys.argv) == 1:
        print("错误: 缺少必需的参数")
        print_usage_examples()
        return False
    
    # 检查是否请求帮助
    if '--help' in sys.argv or '-h' in sys.argv:
        return True  # 让 argparse 处理帮助信息
    
    # 检查是否提供了视频文件
    if '--video' not in sys.argv and '-v' not in sys.argv:
        print("错误: 必须指定视频文件路径")
        print("使用 --video 或 -v 参数指定输入视频文件")
        print_usage_examples()
        return False
    
    return True


def main():
    """主函数入口"""
    # 处理命令行参数错误
    if not handle_command_line_errors():
        sys.exit(1)
    
    # 创建应用实例并运行
    app = VideoObjectDetectionApp()
    
    try:
        exit_code = app.run()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n程序运行时发生未知错误: {str(e)}")
        print("请检查输入参数和文件路径是否正确")
        sys.exit(1)


if __name__ == "__main__":
    main()