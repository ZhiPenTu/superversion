import unittest
import time
import numpy as np
from src.performance_monitor import PerformanceMonitor
from src.video_processor import FrameBuffer

class TestPerformanceOptimization(unittest.TestCase):
    
    def test_frame_buffer(self):
        """Test frame buffering mechanism"""
        buffer = FrameBuffer(max_size=5)
        
        # Test buffer operations
        for i in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer.put(frame)
        
        # Test retrieval
        frame = buffer.get()
        self.assertIsNotNone(frame)
        
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Simulate frame processing
        for _ in range(10):
            monitor.record_frame()
            time.sleep(0.1)
        
        monitor.stop_monitoring()
        metrics = monitor.get_average_metrics()
        
        self.assertIn('average_fps', metrics)
        self.assertGreater(metrics['total_frames'], 0)
        
    def test_memory_optimization(self):
        """Test memory management"""
        from src.video_processor import VideoProcessor
        
        config = {'memory_threshold': 50}
        processor = VideoProcessor(config)
        
        # Test memory monitoring
        memory_cleaned = processor._monitor_memory()
        self.assertIsInstance(memory_cleaned, bool)

if __name__ == '__main__':
    unittest.main()