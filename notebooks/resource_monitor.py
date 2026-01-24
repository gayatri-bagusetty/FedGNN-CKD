import time
import os
import psutil

class ResourceMonitor:
    def __init__(self):
        self.start_time = None
        self.process = psutil.Process(os.getpid())

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        return time.time() - self.start_time

    def memory_usage_mb(self):
        return self.process.memory_info().rss / (1024 ** 2)

    @staticmethod
    def model_size_mb(model_path):
        if not os.path.exists(model_path):
            return 0.0
        return os.path.getsize(model_path) / (1024 ** 2)