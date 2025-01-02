"""Video analysis module for hockey player tracking"""
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from ultralytics import YOLO
import logging
from .config import (
    YOLO_MODEL, YOLO_CONF, YOLO_IMAGE_SIZE, NMS_THRESHOLD,
    BATCH_SIZE, MIN_PLAYERS_ACTIVE, MAX_PLAYERS_PER_TEAM,
    PLAY_BREAK_THRESHOLD, TRACKING_SETTINGS, GPU_MEMORY_FRACTION,
    VIDEO_OUTPUT, MEMORY_SETTINGS
)
from .utils import is_jersey_color, enhance_number_region, validate_bbox

class VideoSegment:
    """Class to store video segment information"""
    def __init__(self, start_frame, end_frame=None):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.players = set()
        self.frame_count = 0
        self.is_active = True

    def add_player(self, jersey_number, jersey_color):
        """Add player to segment"""
        self.players.add(f"{jersey_color}_{jersey_number}")
        
    def has_player(self, jersey_number, jersey_color):
        """Check if player is in segment"""
        return f"{jersey_color}_{jersey_number}" in self.players

    def get_duration(self):
        """Get segment duration in frames"""
        if self.end_frame is None:
            return self.frame_count
        return self.end_frame - self.start_frame + 1

class VideoAnalyzer:
    def __init__(self, video_path):
        """Initialize video analyzer"""
        self.video_path = video_path
        self.logger = self._setup_logger()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")
            
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        self.model = YOLO(YOLO_MODEL).to(self.device)
        
        # Initialize tracking variables
        self.segments = []
        self.current_segment = None
        self.inactive_frames = 0
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger('VideoAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def detect_players(self, frame):
        """Detect players in frame"""
        try:
            if frame is None:
                return []
                
            # Ensure frame size
            if frame.shape[1] > YOLO_IMAGE_SIZE or frame.shape[0] > YOLO_IMAGE_SIZE:
                scale = YOLO_IMAGE_SIZE / max(frame.shape[0], frame.shape[1])
                new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                frame = cv2.resize(frame, new_size)
            
            # Run detection
            results = self.model(frame, 
                               conf=YOLO_CONF,
                               iou=NMS_THRESHOLD,
                               imgsz=YOLO_IMAGE_SIZE)
            
            detections = []
            if results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    if conf > TRACKING_SETTINGS['min_detection_conf']:
                        valid_bbox = validate_bbox(frame, box)
                        if valid_bbox is not None:
                            detections.append((valid_bbox, conf))
                            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return []

    def analyze_frame(self, frame, frame_number):
        """Analyze a single frame"""
        try:
            detections = self.detect_players(frame)
            
            # Check for active play
            if len(detections) >= MIN_PLAYERS_ACTIVE:
                self.inactive_frames = 0
                
                # Start new segment if needed
                if self.current_segment is None:
                    self.current_segment = VideoSegment(frame_number)
                    self.segments.append(self.current_segment)
                
                # Process detections
                for bbox, conf in detections:
                    self._process_detection(frame, bbox, frame_number)
                    
                self.current_segment.frame_count += 1
                
            else:
                self.inactive_frames += 1
                if self.inactive_frames >= PLAY_BREAK_THRESHOLD:
                    self._end_current_segment(frame_number)
                    
        except Exception as e:
            self.logger.error(f"Frame analysis error: {str(e)}")

    def _process_detection(self, frame, bbox, frame_number):
        """Process a single detection"""
        try:
            for color in JERSEY_COLORS.keys():
                if is_jersey_color(frame, bbox, color):
                    jersey_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    number = self._detect_number(jersey_region)
                    if number:
                        self.current_segment.add_player(number, color)
                        break
        except Exception as e:
            self.logger.error(f"Detection processing error: {str(e)}")

    def _detect_number(self, jersey_region):
        """Detect jersey number"""
        try:
            enhanced = enhance_number_region(jersey_region)
            if enhanced is None:
                return None
                
            # OCR logic here
            return None  # Placeholder - implement actual OCR
            
        except Exception as e:
            self.logger.error(f"Number detection error: {str(e)}")
            return None

    def _end_current_segment(self, frame_number):
        """End current segment"""
        if self.current_segment:
            self.current_segment.end_frame = frame_number
            self.current_segment.is_active = False
            self.current_segment = None

    def analyze_video(self, target_jersey=None, target_color=None):
        """Analyze complete video"""
        try:
            frame_number = 0
            progress_bar = tqdm(total=self.total_frames, desc="Analyzing video")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame_number += 1
                self.analyze_frame(frame, frame_number)
                progress_bar.update(1)
                
                # Memory management
                if frame_number % MEMORY_SETTINGS['clear_cache_interval'] == 0:
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                        
            progress_bar.close()
            
            # End final segment
            if self.current_segment:
                self._end_current_segment(frame_number)
                
            # Filter segments for target player if specified
            if target_jersey and target_color:
                return self._filter_segments(target_jersey, target_color)
            
            return self.segments
            
        except Exception as e:
            self.logger.error(f"Video analysis error: {str(e)}")
            return []
        finally:
            self.cleanup()

    def _filter_segments(self, target_jersey, target_color):
        """Filter segments for target player"""
        filtered_segments = []
        for segment in self.segments:
            if segment.has_player(target_jersey, target_color):
                filtered_segments.append(segment)
        return filtered_segments

    def extract_highlights(self, output_dir, target_jersey=None, target_color=None):
        """Extract highlight clips"""
        try:
            segments = self._filter_segments(target_jersey, target_color) if target_jersey else self.segments
            
            clip_paths = []
            for idx, segment in enumerate(segments):
                if segment.get_duration() >= TRACKING_SETTINGS['min_segment_duration']:
                    output_path = os.path.join(output_dir, f"highlight_{idx+1}.mp4")
                    
                    if self._create_clip(segment, output_path):
                        clip_paths.append(output_path)
                        
            return clip_paths
            
        except Exception as e:
            self.logger.error(f"Highlight extraction error: {str(e)}")
            return []

    def _create_clip(self, segment, output_path):
        """Create video clip from segment"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return False
                
            writer = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
            
            for _ in range(segment.get_duration()):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if writer is None:
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*VIDEO_OUTPUT['codec']),
                        self.fps,
                        (self.width, self.height)
                    )
                    
                writer.write(frame)
                
            if writer:
                writer.release()
            cap.release()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Clip creation error: {str(e)}")
            return False

    def get_player_summary(self):
        """Get summary of detected players"""
        player_stats = {}
        for segment in self.segments:
            for player in segment.players:
                if player not in player_stats:
                    player_stats[player] = {
                        'segments': 0,
                        'total_frames': 0
                    }
                player_stats[player]['segments'] += 1
                player_stats[player]['total_frames'] += segment.get_duration()
                
        return player_stats

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.cap.isOpened():
                self.cap.release()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")