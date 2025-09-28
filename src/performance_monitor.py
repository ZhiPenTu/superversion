import time
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    fps: float
    memory_usage: float
    gpu_usage: float
    processing_time: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = None
        self.frame_count = 0
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            time.sleep(1)  # Collect metrics every second
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        memory_usage = psutil.virtual_memory().percent
        
        # GPU usage (if available)
        gpu_usage = 0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = gpu_info.gpu
        except:
            pass
            
        return PerformanceMetrics(
            fps=fps,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            processing_time=elapsed
        )
    
    def record_frame(self):
        """Record that a frame was processed"""
        self.frame_count += 1
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        
    def get_average_metrics(self) -> Dict:
        """Get average performance metrics"""
        if not self.metrics_history:
            return {}
            
        avg_fps = sum(m.fps for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history)
        avg_gpu = sum(m.gpu_usage for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            'average_fps': avg_fps,
            'average_memory_usage': avg_memory,
            'average_gpu_usage': avg_gpu,
            'total_frames': self.frame_count
        }