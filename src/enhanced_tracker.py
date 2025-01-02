"""Enhanced player tracking module with continuous player detection"""
import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np
import easyocr
from contextlib import contextmanager
import logging
from .config import (
    JERSEY_COLORS, YOLO_MODEL, YOLO_CONF, YOLO_IMAGE_SIZE, NMS_THRESHOLD,
    BATCH_SIZE, MAX_MISSED_FRAMES, MIN_CLIP_DURATION, OCR_SETTINGS,
    CUDA_SETTINGS, MEMORY_SETTINGS, ENABLE_OPTIMIZATIONS, DEBUG_FOLDERS,
    GPU_MEMORY_FRACTION
)
from .utils import is_jersey_color, enhance_number_region, save_debug_image, create_debug_visualization, validate_bbox

class PlayerInfo:
    """Class to store player detection information"""
    def __init__(self, jersey_number, jersey_color):
        self.jersey_number = jersey_number
        self.jersey_color = jersey_color
        self.last_seen_frame = 0
        self.first_seen_frame = 0
        self.detected_frames = []
        self.is_active = False

class CUDADebugMixin:
    """Mixin class to add CUDA debugging capabilities"""
    def __init__(self):
        self.debug_logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('CUDADebug')
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    @contextmanager
    def cuda_debug_context(self):
        """Context manager for CUDA debugging"""
        original_env = os.environ.get('CUDA_LAUNCH_BLOCKING')
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1' if not ENABLE_OPTIMIZATIONS['enable_async_loading'] else '0'
        
        try:
            yield
        finally:
            if original_env is None:
                del os.environ['CUDA_LAUNCH_BLOCKING']
            else:
                os.environ['CUDA_LAUNCH_BLOCKING'] = original_env
    
    def check_cuda_status(self):
        """Check CUDA device status and memory usage"""
        if not torch.cuda.is_available():
            self.debug_logger.error("CUDA is not available")
            return False
            
        device_props = torch.cuda.get_device_properties(0)
        self.debug_logger.info(f"GPU: {device_props.name}")
        self.debug_logger.info(f"Compute capability: {device_props.major}.{device_props.minor}")
        
        memory_stats = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        }
        
        self.debug_logger.info("CUDA Memory Stats (MB):")
        for key, value in memory_stats.items():
            self.debug_logger.info(f"{key}: {value:.2f}")
        
        return True

class EnhancedPlayerTracker(CUDADebugMixin):
    def __init__(self, jersey_number, jersey_color, conf_threshold=YOLO_CONF, min_clip_duration=MIN_CLIP_DURATION):
        """Initialize the EnhancedPlayerTracker"""
        super().__init__()
        
        with self.cuda_debug_context():
            try:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
                else:
                    self.device = 'cpu'
                    self.debug_logger.warning("CUDA is not available. Falling back to CPU.")

                self.debug_logger.info(f"Initializing Enhanced Player Tracker on {self.device.upper()}...")

                # Load and optimize YOLO model
                self.model = YOLO(YOLO_MODEL)
                self.model.to(self.device)
                if ENABLE_OPTIMIZATIONS['use_mixed_precision'] and self.device == 'cuda':
                    self.model = self.model.half()

                # Initialize settings
                self.jersey_number = str(jersey_number)
                self.jersey_color = jersey_color
                self.conf_threshold = conf_threshold
                self.min_clip_duration = min_clip_duration
                
                # Initialize OCR with optimized settings
                self.ocr = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
                
                # Initialize player tracking
                self.players = {}  # Dictionary to store player information
                self.active_players = set()  # Set of currently active players
                self.target_player = PlayerInfo(jersey_number, jersey_color)

                print(f"Tracking player {jersey_number} with jersey color '{jersey_color}'...")
                
            except Exception as e:
                self.debug_logger.error(f"Initialization error: {str(e)}")
                raise

    def detect_players(self, frames):
        """Detect all players in frames"""
        with self.cuda_debug_context():
            try:
                all_detections = []
                
                # Validate frames
                if isinstance(frames, list):
                    processed_frames = []
                    for frame in frames:
                        if frame is not None and frame.size > 0 and len(frame.shape) == 3:
                            processed_frames.append(frame)
                    
                    if not processed_frames:
                        return []
                        
                    batch = np.stack(processed_frames)
                else:
                    if frames is None or not hasattr(frames, 'shape'):
                        return []
                    batch = frames

                # Run detection
                with torch.cuda.amp.autocast(enabled=ENABLE_OPTIMIZATIONS['use_mixed_precision']):
                    results = self.model(batch, 
                                      conf=self.conf_threshold,
                                      iou=NMS_THRESHOLD,
                                      imgsz=YOLO_IMAGE_SIZE)

                # Process results safely
                for idx, result in enumerate(results):
                    frame_detections = []
                    if result.boxes:
                        try:
                            current_frame = batch[idx] if isinstance(batch, np.ndarray) else batch
                            if not hasattr(current_frame, 'shape'):
                                continue

                            boxes = result.boxes.xyxy.cpu().numpy()
                            confs = result.boxes.conf.cpu().numpy()
                            
                            for i in range(len(boxes)):
                                try:
                                    if confs[i] < self.conf_threshold:
                                        continue
                                        
                                    bbox = boxes[i].tolist()
                                    valid_bbox = validate_bbox(current_frame, bbox)
                                    
                                    if valid_bbox is not None:
                                        frame_detections.append((valid_bbox, float(confs[i])))
                                except:
                                    continue
                                    
                        except Exception as e:
                            self.debug_logger.error(f"Frame processing error: {str(e)}")
                            continue
                    
                    all_detections.append(frame_detections)

                return all_detections
                
            except Exception as e:
                self.debug_logger.error(f"Detection error: {str(e)}")
                return [[] for _ in range(len(frames) if isinstance(frames, list) else 1)]
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def update_player_info(self, frame_number, jersey_number, jersey_color):
        """Update player tracking information"""
        player_key = f"{jersey_color}_{jersey_number}"
        
        if player_key not in self.players:
            self.players[player_key] = PlayerInfo(jersey_number, jersey_color)
            self.players[player_key].first_seen_frame = frame_number
            
        player = self.players[player_key]
        player.last_seen_frame = frame_number
        player.detected_frames.append(frame_number)
        player.is_active = True
        self.active_players.add(player_key)

    def check_player_timeouts(self, current_frame, timeout_frames=30):
        """Check for players who haven't been seen recently"""
        players_to_remove = set()
        
        for player_key in self.active_players:
            player = self.players[player_key]
            if current_frame - player.last_seen_frame > timeout_frames:
                player.is_active = False
                players_to_remove.add(player_key)
                
        self.active_players -= players_to_remove

    def track_player(self, video_path, output_dir, progress_signal):
        """Track players with continuous detection"""
        with self.cuda_debug_context():
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.debug_logger.error("Unable to open video file.")
                    return None

                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Setup output
                output_video_path = os.path.join(output_dir, f"player_{self.jersey_number}_highlights.mp4")
                tracked_frames_dir = os.path.join(output_dir, f"player_{self.jersey_number}_tracked_frames")
                os.makedirs(tracked_frames_dir, exist_ok=True)

                writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )

                # Initialize variables
                frame_buffer = []
                original_frames = []
                timestamps = []
                frame_count = 0
                include_frame = False
                segment_start = None

                # Calculate target size
                target_height = YOLO_IMAGE_SIZE
                target_width = YOLO_IMAGE_SIZE
                aspect_ratio = width / height
                if aspect_ratio > 1:
                    target_width = int(YOLO_IMAGE_SIZE * aspect_ratio)
                else:
                    target_height = int(YOLO_IMAGE_SIZE / aspect_ratio)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Process frame
                    if frame is not None and frame.size > 0:
                        # Resize frame
                        resized = cv2.resize(frame, (target_width, target_height), 
                                           interpolation=cv2.INTER_LINEAR)
                        
                        frame_buffer.append(resized)
                        original_frames.append(frame)

                    # Process when buffer is full
                    if len(frame_buffer) >= BATCH_SIZE:
                        batch_detections = self.detect_players(frame_buffer)
                        
                        for idx, detections in enumerate(batch_detections):
                            current_frame_num = frame_count - len(frame_buffer) + idx + 1
                            players_in_frame = set()
                            
                            for bbox, conf in detections:
                                if self.is_target_player(frame_buffer[idx], bbox):
                                    self.update_player_info(current_frame_num, self.jersey_number, self.jersey_color)
                                    include_frame = True
                                    if segment_start is None:
                                        segment_start = current_frame_num
                                    
                                    # Save frame
                                    frame_path = os.path.join(tracked_frames_dir, f"frame_{current_frame_num}.jpg")
                                    cv2.imwrite(frame_path, original_frames[idx])
                                    break

                            # Check for player timeouts
                            self.check_player_timeouts(current_frame_num)
                            
                            # Check for segment end
                            if not self.target_player.is_active and segment_start is not None:
                                timestamps.append((segment_start, current_frame_num))
                                segment_start = None

                        frame_buffer = []
                        original_frames = []

                    # Update progress
                    if progress_signal:
                        progress = int((frame_count / total_frames) * 100)
                        progress_signal.emit(progress)

                    # Periodic cleanup
                    if frame_count % MEMORY_SETTINGS['clear_cache_interval'] == 0:
                        torch.cuda.empty_cache()

                # Handle final segment
                if segment_start is not None:
                    timestamps.append((segment_start, frame_count))

                # Create highlight video
                for start, end in timestamps:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    for _ in range(end - start + 1):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            writer.write(frame)

                # Print player summary
                self.print_player_summary()

                cap.release()
                writer.release()
                
                return output_video_path
                
            except Exception as e:
                self.debug_logger.error(f"Tracking error: {str(e)}")
                return None
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def print_player_summary(self):
        """Print summary of detected players"""
        print("\nPlayer Detection Summary:")
        for player_key, player in self.players.items():
            total_frames = len(player.detected_frames)
            duration_seconds = total_frames / 30  # Assuming 30 fps
            print(f"Player {player.jersey_number} ({player.jersey_color}):")
            print(f"  - First seen: Frame {player.first_seen_frame}")
            print(f"  - Last seen: Frame {player.last_seen_frame}")
            print(f"  - Total frames detected: {total_frames}")
            print(f"  - Approximate duration: {duration_seconds:.2f} seconds")

    def is_target_player(self, frame, bbox):
        """Check if detection matches target player"""
        try:
            valid_bbox = validate_bbox(frame, bbox)
            if valid_bbox is None:
                return False

            if is_jersey_color(frame, valid_bbox, self.jersey_color):
                detected_number = self.detect_jersey_number(frame[valid_bbox[1]:valid_bbox[3], 
                                                                valid_bbox[0]:valid_bbox[2]])
                if detected_number == self.jersey_number:
                    return True
            return False
            
        except Exception as e:
            self.debug_logger.error(f"Error in target player detection: {str(e)}")
            return False

    def detect_jersey_number(self, frame):
        """Detect jersey number with enhanced processing"""
        try:
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                return None

            enhanced = enhance_number_region(frame)
            if enhanced is None:
                return None

            results = self.ocr.readtext(enhanced,
                                      min_size=OCR_SETTINGS['min_size'],
                                      text_threshold=OCR_SETTINGS['text_threshold'])
            
            for _, text, conf in results:
                if text.isdigit() and conf > OCR_SETTINGS['text_threshold']:
                    return text
                    
            return None
            
        except Exception as e:
            self.debug_logger.error(f"Error in jersey number detection: {str(e)}")
            return None

    def get_active_players(self):
        """Get list of currently active players"""
        active_players = []
        for player_key in self.active_players:
            player = self.players[player_key]
            active_players.append({
                'number': player.jersey_number,
                'color': player.jersey_color,
                'frames_visible': len(player.detected_frames)
            })
        return active_players

    def get_player_timestamps(self):
        """Get timestamps for when players were active"""
        timestamps = []
        for player_key, player in self.players.items():
            if len(player.detected_frames) > 0:
                frames = sorted(player.detected_frames)
                segments = self._consolidate_segments(frames)
                timestamps.append({
                    'player': player_key,
                    'segments': segments
                })
        return timestamps

    def _consolidate_segments(self, frames, max_gap=30):
        """Consolidate frame numbers into continuous segments"""
        if not frames:
            return []
            
        segments = []
        start = frames[0]
        prev = frames[0]
        
        for frame in frames[1:]:
            if frame - prev > max_gap:
                segments.append((start, prev))
                start = frame
            prev = frame
            
        segments.append((start, prev))
        return segments

    def get_player_stats(self):
        """Get statistics for all detected players"""
        stats = {}
        for player_key, player in self.players.items():
            stats[player_key] = {
                'total_frames': len(player.detected_frames),
                'first_appearance': player.first_seen_frame,
                'last_appearance': player.last_seen_frame,
                'is_active': player.is_active,
                'segments': self._consolidate_segments(sorted(player.detected_frames))
            }
        return stats

    def cleanup(self):
        """Cleanup resources"""
        try:
            torch.cuda.empty_cache()
            self.check_cuda_status()
            self.players.clear()
            self.active_players.clear()
        except Exception as e:
            self.debug_logger.error(f"Cleanup error: {str(e)}")

    def process_segment(self, start_frame, end_frame, cap, output_path):
        """Process a specific segment of video"""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = None
            frames_written = 0
            
            for _ in range(end_frame - start_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30,  # fps
                        (width, height)
                    )
                
                writer.write(frame)
                frames_written += 1
            
            if writer is not None:
                writer.release()
                
            return frames_written > 0
            
        except Exception as e:
            self.debug_logger.error(f"Error processing segment: {str(e)}")
            return False

    def create_highlight_clips(self, video_path, output_dir):
        """Create individual highlight clips for detected segments"""
        try:
            timestamps = self.get_player_timestamps()
            if not timestamps:
                return []
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
                
            clip_paths = []
            for idx, timestamp in enumerate(timestamps):
                player = timestamp['player']
                for seg_idx, (start, end) in enumerate(timestamp['segments']):
                    clip_name = f"highlight_{player}_seg{seg_idx+1}.mp4"
                    clip_path = os.path.join(output_dir, clip_name)
                    
                    if self.process_segment(start, end, cap, clip_path):
                        clip_paths.append(clip_path)
            
            cap.release()
            return clip_paths
            
        except Exception as e:
            self.debug_logger.error(f"Error creating highlight clips: {str(e)}")
            return []

    def is_play_active(self, detections, min_players=3):
        """Determine if play is active based on number of players detected"""
        try:
            player_count = 0
            players_seen = set()
            
            for detection in detections:
                bbox, conf = detection
                if conf > self.conf_threshold:
                    player_count += 1
                    if player_count >= min_players:
                        return True
            
            return False
            
        except Exception as e:
            self.debug_logger.error(f"Error checking play active status: {str(e)}")
            return False