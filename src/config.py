"""Configuration settings for hockey player tracking"""
import torch
import numpy as np
import os

# GPU Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(DEVICE)
CUDA_AVAILABLE = torch.cuda.is_available()

# GPU Memory and Performance Settings
GPU_MEMORY_FRACTION = 0.7
TORCH_BACKENDS_CUDNN_BENCHMARK = True

# Model settings
YOLO_MODEL = 'yolov8s.pt'  # Using small model for speed/accuracy balance
YOLO_CONF = 0.15  # Detection confidence threshold
YOLO_IMAGE_SIZE = 800  # Input image size
NMS_THRESHOLD = 0.3  # Non-max suppression threshold

# Video processing settings
BATCH_SIZE = 2  # Process frames in pairs
MIN_CLIP_DURATION = 0.5  # Minimum duration for a clip in seconds
MAX_MISSED_FRAMES = 15  # Maximum frames to wait before ending a segment
PLAYER_TIMEOUT = 30  # Frames to wait before considering a player inactive

# Video Analysis Settings
MIN_PLAYERS_ACTIVE = 3  # Minimum players needed to consider play active
MAX_PLAYERS_PER_TEAM = 6  # Maximum players expected per team
PLAY_BREAK_THRESHOLD = 20  # Frames threshold to detect play stoppage

# Player Tracking Settings
TRACKING_SETTINGS = {
    'min_detection_conf': 0.15,     # Minimum confidence for player detection
    'min_tracking_conf': 0.1,       # Minimum confidence to maintain tracking
    'max_track_age': 30,           # Maximum frames to maintain tracking without detection
    'min_segment_duration': 15,     # Minimum frames for a valid tracking segment
    'max_segment_gap': 10,         # Maximum frames gap to bridge segments
    'detection_interval': 5        # Frames between full detections
}

# Jersey colors in HSV format (Expanded ranges for better detection)
JERSEY_COLORS = {
    'black-jersey': [
        ((0, 0, 0), (180, 255, 100)),      # Wide black range
        ((0, 0, 0), (180, 150, 80))       # Dark black with reflection
    ],
    'white-red-jersey': [
        ((0, 0, 150), (180, 50, 255)),    # Wide white range
        ((0, 130, 130), (15, 255, 255)),  # Wide red range
        ((165, 130, 130), (180, 255, 255)) # Wrapped red
    ],
    'white-jersey': [
        ((0, 0, 150), (180, 50, 255)),    # White base
        ((0, 0, 130), (180, 70, 255))    # Off-white range
    ],
    'black-white-numbers': [
        ((0, 0, 0), (180, 255, 80)),      # Black base
        ((0, 0, 150), (180, 80, 255))     # White numbers
    ],
    'game-red': [
        ((0, 70, 70), (20, 255, 255)),    # Wide red range
        ((160, 70, 70), (180, 255, 255))  # Wrapped red
    ],
    'game-blue': [
        ((85, 15, 15), (120, 255, 255)),  # Wide blue range
        ((90, 10, 15), (140, 255, 255))   # Additional blue
    ],
    'game-black': [
        ((0, 0, 0), (180, 140, 60)),      # Black with lighting
        ((0, 0, 0), (180, 190, 70))       # Black with reflections
    ],
    'black-light-blue': [
        ((0, 0, 0), (180, 255, 120)),     # Wide black range
        ((85, 20, 20), (120, 255, 255)),  # Wide blue range
        ((0, 0, 0), (180, 110, 110))      # Additional dark range
    ],
    'white-red-blue': [
        ((0, 0, 130), (180, 80, 255)),    # White range
        ((0, 110, 110), (25, 255, 255)),  # Red range
        ((155, 110, 110), (180, 255, 255)), # Wrapped red
        ((80, 30, 30), (155, 255, 255))   # Blue range
    ]
}

# OCR settings (Optimized for jersey numbers)
OCR_SETTINGS = {
    'batch_size': 1,
    'min_size': 8,
    'text_threshold': 0.3,
    'link_threshold': 0.2,
    'low_text': 0.2,
    'canvas_size': 1536,   # Larger for better number recognition
    'mag_ratio': 1.5,
    'gpu_mem_fraction': 0.3,
    'paragraph': False     # Disable for speed
}

# Image processing settings
IMAGE_PROCESSING = {
    'resize_factor': 1.5,       # Resize factor for number detection
    'contrast_alpha': 1.8,      # Contrast enhancement
    'contrast_beta': 35,        # Brightness adjustment
    'blur_kernel': (3, 3),      # Denoising kernel
    'threshold_block_size': 11,
    'threshold_C': 2,
    'max_dimension': 1280,      # Maximum image dimension
    'enable_sharpening': True,  # Enable image sharpening
    'enable_denoising': True,   # Enable denoising
    'denoise_strength': 7,
    'sharpen_kernel': np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
}

# Padding settings for bounding boxes
BBOX_PADDING = {
    'top': 30,
    'bottom': 30,
    'left': 20,
    'right': 20
}

# Debug settings
SAVE_DEBUG_FRAMES = True
DEBUG_FRAME_INTERVAL = 100
DEBUG_OUTPUT_DIR = "debug_frames"
DEBUG_FOLDERS = {
    'main': "debug_frames",
    'detections': "debug_frames/detections",
    'tracked': "debug_frames/tracked",
    'numbers': "debug_frames/numbers",
    'colors': "debug_frames/colors"
}

# Create debug directories if enabled
if SAVE_DEBUG_FRAMES:
    for folder in DEBUG_FOLDERS.values():
        os.makedirs(folder, exist_ok=True)

# Video output settings
VIDEO_OUTPUT = {
    'codec': 'mp4v',
    'fps_factor': 1.0,
    'min_clip_frames': 15,
    'max_clip_gap': 15,
    'max_dimension': 1920,    # Full HD
    'compression_quality': 95
}

# Debug visualization colors (BGR format)
DEBUG_COLORS = {
    'detected': (0, 255, 0),     # Green
    'color_match': (0, 165, 255),  # Orange
    'no_match': (0, 0, 255),     # Red
    'text_color': (255, 255, 255)  # White
}

# CUDA optimization settings
CUDA_SETTINGS = {
    'device_index': 0,
    'non_blocking': True,
    'async_op': True,
    'precision': 'half',     # Use half precision for speed
    'pin_memory': True,
    'num_workers': 4,
    'prefetch_factor': 2,
    'use_mixed_precision': True  # Enable for speed
}

# Memory management settings
MEMORY_SETTINGS = {
    'clear_cache_interval': 400,  # Clear cache every N frames
    'force_gc_interval': 800,    # Force garbage collection interval
    'max_batch_memory_mb': 1024, # Maximum batch memory
    'buffer_size': 8            # Frame buffer size
}

# Processing optimization flags
ENABLE_OPTIMIZATIONS = {
    'use_mixed_precision': True,
    'enable_tensor_cores': True,
    'enable_cuda_graphs': True,
    'enable_memory_pinning': True,
    'enable_async_loading': True,
    'enable_frame_prefetch': True,
    'enable_batch_processing': True
}

# Performance monitoring settings
PERFORMANCE_MONITORING = {
    'log_gpu_usage': True,
    'log_memory_usage': True,
    'log_processing_time': True,
    'profiling_enabled': True
}