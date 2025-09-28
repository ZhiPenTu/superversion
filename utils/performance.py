"""
性能优化工具模块

提供帧缓冲、内存管理和GPU加速等性能优化功能。
"""

import gc
import time
import threading
import queue
from typing import List, Optional, Tuple, Any, Callable
from collections import deque
import numpy as np
import psutil
import logging

from models.data_models import FrameDetections


class FrameBuffer:
    """帧缓冲器
    
    实现多线程帧缓冲机制，提高视频处理效率。
    支持预读取帧数据，减少I/O等待时间。
    """
    
    def __init__(self, max_size: int = 10, prefetch_size: int = 5):
        """初始化帧缓冲器
        
        Args:
            max_size: 缓冲区最大大小
            prefetch_size: 预取帧数量
        """
        self.max_size = max_size
        self.prefetch_size = prefetch_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.prefetch_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_source: Optional[Any] = None
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            'frames_buffered': 0,
            'buffer_hits': 0,
            'buffer_misses': 0,
            'total_frames_read': 0
        }
    
    def start_buffering(self, frame_source: Any) -> None:
        """开始帧缓冲
        
        Args:
            frame_source: 帧数据源（支持迭代器接口）
        """
        self.frame_source = frame_source
        self.stop_event.clear()
        
        # 启动预取线程
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_frames,
            daemon=True
        )
        self.prefetch_thread.start()
        
        self.logger.info(f"帧缓冲已启动，缓冲区大小: {self.max_size}")
    
    def _prefetch_frames(self) -> None:
        """预取帧数据的后台线程"""
        try:
            for ret, frame in self.frame_source:
                if self.stop_event.is_set():
                    break
                
                if not ret or frame is None:
                    # 添加结束标记
                    self.buffer.put((False, None), timeout=1.0)
                    break
                
                try:
                    # 将帧添加到缓冲区
                    self.buffer.put((ret, frame), timeout=1.0)
                    self.stats['frames_buffered'] += 1
                except queue.Full:
                    # 缓冲区满，跳过当前帧
                    self.logger.warning("帧缓冲区已满，跳过帧")
                    continue
                    
        except Exception as e:
            self.logger.error(f"帧预取线程错误: {str(e)}")
        finally:
            self.logger.debug("帧预取线程结束")
    
    def get_frame(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """从缓冲区获取帧
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 帧数据)
        """
        try:
            ret, frame = self.buffer.get(timeout=timeout)
            self.stats['total_frames_read'] += 1
            
            if ret:
                self.stats['buffer_hits'] += 1
            else:
                self.stats['buffer_misses'] += 1
            
            return ret, frame
            
        except queue.Empty:
            self.stats['buffer_misses'] += 1
            return False, None
    
    def stop_buffering(self) -> None:
        """停止帧缓冲"""
        self.stop_event.set()
        
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
        
        # 清空缓冲区
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("帧缓冲已停止")
    
    def get_buffer_status(self) -> dict:
        """获取缓冲区状态
        
        Returns:
            dict: 缓冲区状态信息
        """
        return {
            'buffer_size': self.buffer.qsize(),
            'max_size': self.max_size,
            'is_active': not self.stop_event.is_set(),
            'stats': self.stats.copy()
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_buffering()


class MemoryManager:
    """内存管理器
    
    监控和优化内存使用，防止内存泄漏和过度使用。
    """
    
    def __init__(self, max_memory_mb: Optional[int] = None, 
                 gc_threshold: float = 0.8):
        """初始化内存管理器
        
        Args:
            max_memory_mb: 最大内存使用量（MB），None表示不限制
            gc_threshold: 触发垃圾回收的内存使用阈值（0.0-1.0）
        """
        self.max_memory_mb = max_memory_mb
        self.gc_threshold = gc_threshold
        self.logger = logging.getLogger(__name__)
        
        # 获取系统内存信息
        self.system_memory = psutil.virtual_memory()
        self.process = psutil.Process()
        
        # 统计信息
        self.stats = {
            'gc_count': 0,
            'peak_memory_mb': 0,
            'memory_warnings': 0,
            'last_gc_time': 0
        }
        
        self.logger.info(f"内存管理器初始化，系统内存: {self.system_memory.total / (1024**3):.1f} GB")
    
    def get_memory_usage(self) -> dict:
        """获取当前内存使用情况
        
        Returns:
            dict: 内存使用信息
        """
        try:
            # 进程内存使用
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # 系统内存使用
            system_memory = psutil.virtual_memory()
            
            # 更新峰值内存
            if memory_mb > self.stats['peak_memory_mb']:
                self.stats['peak_memory_mb'] = memory_mb
            
            return {
                'process_memory_mb': memory_mb,
                'system_memory_percent': system_memory.percent,
                'system_available_mb': system_memory.available / (1024 * 1024),
                'peak_memory_mb': self.stats['peak_memory_mb']
            }
            
        except Exception as e:
            self.logger.error(f"获取内存使用信息失败: {str(e)}")
            return {}
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况并执行必要的清理
        
        Returns:
            bool: 内存使用正常返回True，否则返回False
        """
        memory_info = self.get_memory_usage()
        
        if not memory_info:
            return True
        
        process_memory_mb = memory_info['process_memory_mb']
        system_memory_percent = memory_info['system_memory_percent']
        
        # 检查进程内存限制
        if self.max_memory_mb and process_memory_mb > self.max_memory_mb:
            self.logger.warning(f"进程内存使用超过限制: {process_memory_mb:.1f} MB > {self.max_memory_mb} MB")
            self.stats['memory_warnings'] += 1
            self._force_garbage_collection()
            return False
        
        # 检查系统内存使用
        if system_memory_percent > self.gc_threshold * 100:
            self.logger.warning(f"系统内存使用过高: {system_memory_percent:.1f}%")
            self.stats['memory_warnings'] += 1
            self._force_garbage_collection()
        
        # 定期垃圾回收
        current_time = time.time()
        if current_time - self.stats['last_gc_time'] > 30:  # 每30秒检查一次
            if process_memory_mb > 500:  # 进程内存超过500MB时执行GC
                self._force_garbage_collection()
        
        return True
    
    def _force_garbage_collection(self) -> None:
        """强制执行垃圾回收"""
        try:
            # 执行垃圾回收
            collected = gc.collect()
            
            self.stats['gc_count'] += 1
            self.stats['last_gc_time'] = time.time()
            
            # 获取回收后的内存使用
            memory_info = self.get_memory_usage()
            current_memory = memory_info.get('process_memory_mb', 0)
            
            self.logger.info(f"垃圾回收完成，回收对象: {collected}, "
                           f"当前内存: {current_memory:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"垃圾回收失败: {str(e)}")
    
    def optimize_memory_for_frame_processing(self, frame_shape: tuple) -> dict:
        """为帧处理优化内存设置
        
        Args:
            frame_shape: 帧的形状 (H, W, C)
            
        Returns:
            dict: 优化建议
        """
        # 计算单帧内存使用
        frame_size_mb = np.prod(frame_shape) * 4 / (1024 * 1024)  # 假设float32
        
        # 获取当前内存状态
        memory_info = self.get_memory_usage()
        available_mb = memory_info.get('system_available_mb', 1000)
        
        # 计算建议的缓冲区大小
        max_buffer_frames = max(1, int(available_mb * 0.1 / frame_size_mb))
        max_buffer_frames = min(max_buffer_frames, 20)  # 最多20帧
        
        # 计算建议的批处理大小
        batch_size = max(1, int(available_mb * 0.05 / frame_size_mb))
        batch_size = min(batch_size, 8)  # 最多8帧批处理
        
        recommendations = {
            'frame_size_mb': frame_size_mb,
            'recommended_buffer_size': max_buffer_frames,
            'recommended_batch_size': batch_size,
            'available_memory_mb': available_mb,
            'memory_pressure': memory_info.get('system_memory_percent', 0) > 80
        }
        
        self.logger.info(f"内存优化建议: 缓冲区大小={max_buffer_frames}, "
                        f"批处理大小={batch_size}, 单帧大小={frame_size_mb:.2f}MB")
        
        return recommendations
    
    def get_memory_stats(self) -> dict:
        """获取内存管理统计信息
        
        Returns:
            dict: 统计信息
        """
        current_memory = self.get_memory_usage()
        
        return {
            **self.stats,
            **current_memory,
            'max_memory_limit_mb': self.max_memory_mb,
            'gc_threshold': self.gc_threshold
        }


class GPUAccelerator:
    """GPU加速器
    
    管理GPU资源和加速计算，支持CUDA和其他GPU后端。
    """
    
    def __init__(self):
        """初始化GPU加速器"""
        self.logger = logging.getLogger(__name__)
        self.device_info = self._detect_gpu_devices()
        self.current_device = None
        self.memory_pool = None
        
    def _detect_gpu_devices(self) -> dict:
        """检测可用的GPU设备
        
        Returns:
            dict: GPU设备信息
        """
        device_info = {
            'cuda_available': False,
            'cuda_devices': [],
            'current_device': None,
            'total_memory_mb': 0,
            'available_memory_mb': 0
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                device_info['cuda_available'] = True
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    device_props = torch.cuda.get_device_properties(i)
                    device_memory = torch.cuda.get_device_properties(i).total_memory
                    
                    device_info['cuda_devices'].append({
                        'id': i,
                        'name': device_props.name,
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                        'total_memory_mb': device_memory / (1024 * 1024),
                        'multiprocessor_count': device_props.multi_processor_count
                    })
                
                # 选择最佳设备（通常是第一个）
                if device_count > 0:
                    self.current_device = 0
                    device_info['current_device'] = 0
                    
                    # 获取当前设备内存信息
                    torch.cuda.set_device(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    device_info['total_memory_mb'] = total_memory / (1024 * 1024)
                    
                    # 获取可用内存
                    if hasattr(torch.cuda, 'mem_get_info'):
                        free_memory, _ = torch.cuda.mem_get_info()
                        device_info['available_memory_mb'] = free_memory / (1024 * 1024)
                
                self.logger.info(f"检测到 {device_count} 个CUDA设备")
                for device in device_info['cuda_devices']:
                    self.logger.info(f"  设备 {device['id']}: {device['name']} "
                                   f"({device['total_memory_mb']:.0f} MB)")
            else:
                self.logger.info("未检测到CUDA设备，将使用CPU")
                
        except ImportError:
            self.logger.warning("PyTorch未安装，无法使用GPU加速")
        except Exception as e:
            self.logger.error(f"GPU设备检测失败: {str(e)}")
        
        return device_info
    
    def is_gpu_available(self) -> bool:
        """检查GPU是否可用
        
        Returns:
            bool: GPU可用返回True
        """
        return self.device_info['cuda_available']
    
    def get_optimal_device(self) -> str:
        """获取最优设备
        
        Returns:
            str: 设备名称 ('cuda' 或 'cpu')
        """
        if self.is_gpu_available():
            return 'cuda'
        return 'cpu'
    
    def get_gpu_memory_info(self) -> dict:
        """获取GPU内存信息
        
        Returns:
            dict: GPU内存信息
        """
        if not self.is_gpu_available():
            return {'error': 'GPU不可用'}
        
        try:
            import torch
            
            if self.current_device is not None:
                torch.cuda.set_device(self.current_device)
                
                # 获取内存信息
                if hasattr(torch.cuda, 'mem_get_info'):
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    used_memory = total_memory - free_memory
                    
                    return {
                        'device_id': self.current_device,
                        'total_memory_mb': total_memory / (1024 * 1024),
                        'used_memory_mb': used_memory / (1024 * 1024),
                        'free_memory_mb': free_memory / (1024 * 1024),
                        'memory_usage_percent': (used_memory / total_memory) * 100
                    }
            
            return {'error': '无法获取GPU内存信息'}
            
        except Exception as e:
            return {'error': f'获取GPU内存信息失败: {str(e)}'}
    
    def optimize_gpu_memory(self) -> bool:
        """优化GPU内存使用
        
        Returns:
            bool: 优化成功返回True
        """
        if not self.is_gpu_available():
            return False
        
        try:
            import torch
            
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # 设置内存分配策略
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    # 限制GPU内存使用为80%
                    torch.cuda.set_per_process_memory_fraction(0.8)
                
                self.logger.info("GPU内存优化完成")
                return True
                
        except Exception as e:
            self.logger.error(f"GPU内存优化失败: {str(e)}")
        
        return False
    
    def get_recommended_batch_size(self, frame_shape: tuple, 
                                 model_memory_mb: float = 500) -> int:
        """根据GPU内存推荐批处理大小
        
        Args:
            frame_shape: 帧的形状 (H, W, C)
            model_memory_mb: 模型占用的内存（MB）
            
        Returns:
            int: 推荐的批处理大小
        """
        if not self.is_gpu_available():
            return 1
        
        gpu_memory = self.get_gpu_memory_info()
        if 'error' in gpu_memory:
            return 1
        
        # 计算单帧内存使用（包括输入和中间结果）
        frame_size_mb = np.prod(frame_shape) * 4 / (1024 * 1024)  # float32
        estimated_frame_memory = frame_size_mb * 3  # 考虑中间结果
        
        # 计算可用内存
        available_memory = gpu_memory['free_memory_mb'] - model_memory_mb
        available_memory = max(0, available_memory * 0.8)  # 保留20%缓冲
        
        # 计算批处理大小
        if estimated_frame_memory > 0:
            batch_size = int(available_memory / estimated_frame_memory)
            batch_size = max(1, min(batch_size, 16))  # 限制在1-16之间
        else:
            batch_size = 1
        
        self.logger.info(f"GPU批处理大小推荐: {batch_size} "
                        f"(可用内存: {available_memory:.1f}MB, "
                        f"单帧估算: {estimated_frame_memory:.1f}MB)")
        
        return batch_size
    
    def get_device_info(self) -> dict:
        """获取设备信息
        
        Returns:
            dict: 设备信息
        """
        return self.device_info.copy()


class PerformanceOptimizer:
    """性能优化器
    
    集成帧缓冲、内存管理和GPU加速的综合性能优化器。
    """
    
    def __init__(self, config: Optional[dict] = None):
        """初始化性能优化器
        
        Args:
            config: 优化配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        self.frame_buffer: Optional[FrameBuffer] = None
        self.memory_manager = MemoryManager(
            max_memory_mb=self.config.get('max_memory_mb'),
            gc_threshold=self.config.get('gc_threshold', 0.8)
        )
        self.gpu_accelerator = GPUAccelerator()
        
        # 性能统计
        self.performance_stats = {
            'optimization_start_time': time.time(),
            'frames_processed': 0,
            'total_processing_time': 0,
            'average_fps': 0,
            'memory_optimizations': 0,
            'gpu_optimizations': 0
        }
        
        self.logger.info("性能优化器初始化完成")
    
    def optimize_for_video_processing(self, video_info: dict) -> dict:
        """为视频处理优化性能设置
        
        Args:
            video_info: 视频信息字典
            
        Returns:
            dict: 优化建议和设置
        """
        frame_shape = (video_info.get('height', 720), 
                      video_info.get('width', 1280), 3)
        
        # 内存优化建议
        memory_recommendations = self.memory_manager.optimize_memory_for_frame_processing(frame_shape)
        
        # GPU优化建议
        gpu_recommendations = {}
        if self.gpu_accelerator.is_gpu_available():
            gpu_recommendations = {
                'use_gpu': True,
                'device': self.gpu_accelerator.get_optimal_device(),
                'batch_size': self.gpu_accelerator.get_recommended_batch_size(frame_shape),
                'gpu_memory_info': self.gpu_accelerator.get_gpu_memory_info()
            }
            
            # 优化GPU内存
            if self.gpu_accelerator.optimize_gpu_memory():
                self.performance_stats['gpu_optimizations'] += 1
        else:
            gpu_recommendations = {
                'use_gpu': False,
                'device': 'cpu',
                'batch_size': 1
            }
        
        # 帧缓冲设置
        buffer_size = memory_recommendations.get('recommended_buffer_size', 5)
        buffer_size = min(buffer_size, self.config.get('max_buffer_size', 10))
        
        optimization_settings = {
            'frame_buffer_size': buffer_size,
            'memory_settings': memory_recommendations,
            'gpu_settings': gpu_recommendations,
            'performance_mode': self._determine_performance_mode(memory_recommendations, gpu_recommendations)
        }
        
        self.logger.info(f"视频处理优化完成，性能模式: {optimization_settings['performance_mode']}")
        
        return optimization_settings
    
    def _determine_performance_mode(self, memory_rec: dict, gpu_rec: dict) -> str:
        """确定性能模式
        
        Args:
            memory_rec: 内存建议
            gpu_rec: GPU建议
            
        Returns:
            str: 性能模式 ('high', 'medium', 'low')
        """
        if gpu_rec.get('use_gpu') and not memory_rec.get('memory_pressure', False):
            return 'high'
        elif gpu_rec.get('use_gpu') or memory_rec.get('available_memory_mb', 0) > 2000:
            return 'medium'
        else:
            return 'low'
    
    def create_optimized_frame_buffer(self, frame_source: Any, 
                                    buffer_size: Optional[int] = None) -> FrameBuffer:
        """创建优化的帧缓冲器
        
        Args:
            frame_source: 帧数据源
            buffer_size: 缓冲区大小，None时使用推荐值
            
        Returns:
            FrameBuffer: 帧缓冲器实例
        """
        if buffer_size is None:
            buffer_size = self.config.get('frame_buffer_size', 5)
        
        self.frame_buffer = FrameBuffer(
            max_size=buffer_size,
            prefetch_size=max(1, buffer_size // 2)
        )
        
        self.frame_buffer.start_buffering(frame_source)
        
        self.logger.info(f"优化帧缓冲器已创建，缓冲区大小: {buffer_size}")
        
        return self.frame_buffer
    
    def monitor_performance(self, processing_time: float) -> dict:
        """监控处理性能
        
        Args:
            processing_time: 单帧处理时间
            
        Returns:
            dict: 性能监控信息
        """
        self.performance_stats['frames_processed'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        # 计算平均FPS
        if self.performance_stats['total_processing_time'] > 0:
            self.performance_stats['average_fps'] = (
                self.performance_stats['frames_processed'] / 
                self.performance_stats['total_processing_time']
            )
        
        # 检查内存使用
        memory_ok = self.memory_manager.check_memory_usage()
        if not memory_ok:
            self.performance_stats['memory_optimizations'] += 1
        
        # 获取当前状态
        current_status = {
            'current_fps': 1.0 / processing_time if processing_time > 0 else 0,
            'average_fps': self.performance_stats['average_fps'],
            'memory_usage': self.memory_manager.get_memory_usage(),
            'frames_processed': self.performance_stats['frames_processed'],
            'memory_ok': memory_ok
        }
        
        # 如果有GPU，获取GPU状态
        if self.gpu_accelerator.is_gpu_available():
            current_status['gpu_memory'] = self.gpu_accelerator.get_gpu_memory_info()
        
        return current_status
    
    def get_optimization_report(self) -> dict:
        """获取优化报告
        
        Returns:
            dict: 优化报告
        """
        total_time = time.time() - self.performance_stats['optimization_start_time']
        
        report = {
            'performance_stats': self.performance_stats.copy(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'gpu_info': self.gpu_accelerator.get_device_info(),
            'total_optimization_time': total_time,
            'optimization_efficiency': self._calculate_efficiency()
        }
        
        if self.frame_buffer:
            report['buffer_stats'] = self.frame_buffer.get_buffer_status()
        
        return report
    
    def _calculate_efficiency(self) -> dict:
        """计算优化效率
        
        Returns:
            dict: 效率指标
        """
        stats = self.performance_stats
        
        if stats['frames_processed'] == 0:
            return {'efficiency_score': 0, 'status': 'no_data'}
        
        # 计算效率分数（基于FPS和资源使用）
        avg_fps = stats['average_fps']
        memory_stats = self.memory_manager.get_memory_stats()
        
        # FPS效率（假设目标是30fps）
        fps_efficiency = min(1.0, avg_fps / 30.0)
        
        # 内存效率（基于GC次数）
        memory_efficiency = max(0.0, 1.0 - (memory_stats['gc_count'] / max(1, stats['frames_processed'])))
        
        # 综合效率分数
        efficiency_score = (fps_efficiency * 0.7 + memory_efficiency * 0.3)
        
        return {
            'efficiency_score': efficiency_score,
            'fps_efficiency': fps_efficiency,
            'memory_efficiency': memory_efficiency,
            'status': 'good' if efficiency_score > 0.8 else 'moderate' if efficiency_score > 0.6 else 'poor'
        }
    
    def cleanup(self) -> None:
        """清理优化器资源"""
        if self.frame_buffer:
            self.frame_buffer.stop_buffering()
        
        # 强制执行最后的内存清理
        self.memory_manager._force_garbage_collection()
        
        # 清理GPU缓存
        if self.gpu_accelerator.is_gpu_available():
            self.gpu_accelerator.optimize_gpu_memory()
        
        self.logger.info("性能优化器清理完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()