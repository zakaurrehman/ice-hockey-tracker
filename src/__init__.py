# src/__init__.py

"""Hockey player tracking package"""
import torch
from .config import DEVICE, TORCH_DEVICE, CUDA_AVAILABLE, GPU_MEMORY_FRACTION, TORCH_BACKENDS_CUDNN_BENCHMARK

def initialize_gpu():
    """Initialize GPU settings if available."""
    if CUDA_AVAILABLE:
        try:
            # Configure CUDA settings
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = TORCH_BACKENDS_CUDNN_BENCHMARK
            torch.cuda.empty_cache()
            
            # Set memory fraction
            if GPU_MEMORY_FRACTION > 0:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            
            print(f"GPU initialized: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            return True
        except Exception as e:
            print(f"Error initializing GPU: {e}")
            print("Falling back to CPU")
            return False
    return False

# Initialize GPU first
gpu_initialized = initialize_gpu()

# Import other modules
from .detector import PlayerDetector
from .enhanced_tracker import EnhancedPlayerTracker
from .utils import print_gpu_utilization
from .video import extract_video_clip, get_video_info

__version__ = '1.0.0'