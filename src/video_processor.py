import threading
from queue import Queue
from collections import deque
import gc
import psutil

class FrameBuffer:
    def __init__(self, max_size=30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)
    
    def get(self):
        with self.lock:
            return self.buffer.popleft() if self.buffer else None

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.frame_buffer = FrameBuffer(config.get('buffer_size', 30))
        self.memory_threshold = config.get('memory_threshold', 80)  # 80% memory usage
        
    def _monitor_memory(self):
        """Monitor memory usage and trigger cleanup if needed"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.memory_threshold:
            gc.collect()
            return True
        return False
    
    def process_video_optimized(self, video_path, detector, visualizer):
        """Optimized video processing with buffering"""
        cap = cv2.VideoCapture(video_path)
        
        # Pre-allocate frame buffer
        frame_queue = Queue(maxsize=self.config.get('queue_size', 10))
        
        def frame_reader():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_queue.put(frame)
                
                # Memory management
                if self._monitor_memory():
                    time.sleep(0.01)  # Brief pause for GC
        
        # Start frame reading thread
        reader_thread = threading.Thread(target=frame_reader)
        reader_thread.start()
        
        # Process frames with buffering
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                # Process frame...
            else:
                break
                
        reader_thread.join()
        cap.release()