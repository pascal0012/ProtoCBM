import time
from typing import Optional

class Timer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        name = self.name or "Block"
        print(f"{name} took {self.elapsed:.4f} seconds")
        return False