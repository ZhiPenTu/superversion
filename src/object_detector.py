import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ObjectDetector:
    def __init__(self, config):
        self.config = config
        self.device = self._select_device()
        self.batch_size = config.get('batch_size', 4)
        self.use_gpu_acceleration = config.get('use_gpu', True) and torch.cuda.is_available()
        
    def _select_device(self):
        """Automatically select best available device"""
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            return torch.device('cuda')
        return torch.device('cpu')
    
    def detect_batch_optimized(self, frames):
        """Optimized batch detection with GPU acceleration"""
        if not frames:
            return []
            
        # Convert frames to tensor batch
        frame_tensors = []
        for frame in frames:
            tensor = torch.from_numpy(frame).float()
            if self.use_gpu_acceleration:
                tensor = tensor.cuda()
            frame_tensors.append(tensor)
        
        # Batch processing
        with torch.no_grad():
            batch_tensor = torch.stack(frame_tensors)
            results = self.model(batch_tensor)
            
        # Clear GPU cache if using CUDA
        if self.use_gpu_acceleration:
            torch.cuda.empty_cache()
            
        return results
    
    def detect_with_threading(self, frames):
        """Multi-threaded detection for CPU optimization"""
        if self.use_gpu_acceleration:
            return self.detect_batch_optimized(frames)
            
        with ThreadPoolExecutor(max_workers=self.config.get('cpu_threads', 4)) as executor:
            futures = [executor.submit(self._detect_single, frame) for frame in frames]
            results = [future.result() for future in futures]
        return results