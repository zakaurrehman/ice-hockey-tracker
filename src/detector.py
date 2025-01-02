import torch
import cv2
import easyocr
import numpy as np
import os
import traceback
import time
from ultralytics import YOLO
from .config import *
from .utils import (print_gpu_utilization, is_jersey_color, enhance_number_region, 
                   create_debug_visualization, save_debug_image)
from .video import extract_video_clip, get_video_info
from .utils import print_gpu_utilization, is_jersey_color


class PlayerDetector:
    def __init__(self):
        """Initialize the player detector with GPU support"""
        print("\nInitializing PlayerDetector...")
        
        # Initialize YOLO model
        self.model = YOLO(YOLO_MODEL)
        if CUDA_AVAILABLE:
            self.model.to(TORCH_DEVICE)
            print(f"Model loaded on: {TORCH_DEVICE}")
        
        # Initialize OCR with GPU support
        self.ocr = easyocr.Reader(['en'], gpu=CUDA_AVAILABLE)
        print("OCR initialized")
        
        # Create debug directory if needed
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    def process_batch(self, frames):
        """Process multiple frames at once using GPU"""
        try:
            processed_frames = []
            for frame in frames:
                # Convert frame to the format expected by YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 640))
                frame_chw = frame_resized.transpose(2, 0, 1)
                frame_tensor = torch.from_numpy(frame_chw).float() / 255.0
                processed_frames.append(frame_tensor)
            
            # Stack frames and move to GPU
            batch_tensor = torch.stack(processed_frames).to(TORCH_DEVICE)
            
            # Run inference with automatic mixed precision
            with torch.amp.autocast(TORCH_DEVICE.type):
                results = self.model(batch_tensor)
            
            return results
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return None

    def _extract_player_bbox(self, frame, xyxy):
        """Extract and preprocess player bounding box"""
        try:
            y1, y2 = int(xyxy[1]), int(xyxy[3])
            x1, x2 = int(xyxy[0]), int(xyxy[2])
            
            # Add padding with more generous bounds
            height, width = frame.shape[:2]
            padding = 30  # Increased padding for better number detection
            y1 = max(0, y1 - padding)
            y2 = min(height, y2 + padding)
            x1 = max(0, x1 - padding)
            x2 = min(width, x2 + padding)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            bbox = frame[y1:y2, x1:x2]
            if bbox.size == 0:
                return None
                
            # Enhance the bbox for better number detection
            bbox_enhanced = cv2.resize(bbox, (bbox.shape[1]*2, bbox.shape[0]*2))
            bbox_enhanced = cv2.convertScaleAbs(bbox_enhanced, alpha=1.3, beta=20)
            
            return bbox_enhanced
            
        except Exception as e:
            print(f"Error extracting bbox: {str(e)}")
            return None

    def _check_player_number(self, bbox, jersey_number):
        """Check if bbox contains the target player number"""
        try:
            if bbox is None:
                return False

            # Try multiple preprocessing approaches
            preprocessed_images = [
                bbox,  # Original enhanced bbox
                enhance_number_region(bbox),  # Apply additional enhancement
                cv2.convertScaleAbs(bbox, alpha=1.5, beta=30)  # Different contrast
            ]
            
            for img in preprocessed_images:
                ocr_result = self.ocr.readtext(img, 
                                             allowlist='0123456789',
                                             batch_size=1,
                                             min_size=10)
                
                for detection in ocr_result:
                    text = detection[1]
                    conf = detection[2]
                    
                    # Debug output for high confidence detections
                    if conf > OCR_CONFIDENCE * 0.8:
                        print(f"OCR detected: {text} (conf: {conf:.2f})")
                    
                    if jersey_number in text and conf > OCR_CONFIDENCE:
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error in number detection: {str(e)}")
            return False

    def detect_player(self, video_path, output_path, jersey_color, jersey_number):
        """Main detection function"""
        # Initialize video capture and variables
        cap = cv2.VideoCapture(video_path)
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        timestamps = []
        start_frame = None
        frame_count = 0
        frames_batch = []
        missed_frames = 0
        missed_frames_threshold = 5  # Reduced threshold for more clips
        min_clip_duration = 0.3      # Reduced minimum duration
        
        print(f"\nProcessing video with {video_info['total_frames']} frames...")
        print(f"Tracking player with {jersey_color} jersey and number {jersey_number}")
        print(f"Video FPS: {fps}, Resolution: {video_info['width']}x{video_info['height']}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % SKIP_FRAMES != 0:  # Skip frames for speed
                    continue
                
                frames_batch.append(frame)
                
                # Process batch when full or at end
                if len(frames_batch) >= BATCH_SIZE or frame_count >= video_info['total_frames']:
                    if frame_count % 50 == 0:
                        print(f"Progress: {frame_count}/{video_info['total_frames']} "
                              f"frames ({(frame_count/video_info['total_frames']*100):.1f}%)")
                        print_gpu_utilization()
                    
                    results = self.process_batch(frames_batch)
                    if results is None:
                        frames_batch = []
                        continue
                    
                    # Process each frame in the batch
                    for idx, result in enumerate(results):
                        player_found = False
                        current_frame = frames_batch[idx].copy()
                        current_frame_number = frame_count - len(frames_batch) + idx + 1
                        
                        # Check each detection
                        if len(result.boxes) > 0:
                            for box in result.boxes:
                                if (box.cls[0].item() == 0 and  # person class
                                    box.conf[0].item() > PERSON_CONF_THRESHOLD):
                                    
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    
                                    # Check jersey color and number
                                    if is_jersey_color(current_frame, xyxy, jersey_color):
                                        bbox = self._extract_player_bbox(current_frame, xyxy)
                                        
                                        if bbox is not None and self._check_player_number(bbox, jersey_number):
                                            if start_frame is None:
                                                start_frame = current_frame_number
                                                print(f"\nStarted tracking player {jersey_number} "
                                                      f"at frame {current_frame_number}")
                                            
                                            missed_frames = 0
                                            player_found = True
                                            
                                            # Save debug frame if enabled
                                            if SAVE_DEBUG_FRAMES:
                                                debug_frame = create_debug_visualization(
                                                    current_frame, xyxy, True, True, jersey_number)
                                                save_debug_image(debug_frame, 
                                                               f"{DEBUG_OUTPUT_DIR}/frame_{current_frame_number}.jpg")
                                            
                                            break
                        
                        # Handle end of detection
                        if not player_found and start_frame is not None:
                            missed_frames += 1
                            if missed_frames >= missed_frames_threshold:
                                current_duration = (current_frame_number - start_frame) / fps
                                if current_duration >= min_clip_duration:
                                    timestamps.append((start_frame, current_frame_number))
                                    print(f"Found clip from frame {start_frame} to {current_frame_number} "
                                          f"(duration: {current_duration:.2f}s)")
                                start_frame = None
                                missed_frames = 0
                    
                    frames_batch = []

            # Handle final clip
            if start_frame is not None:
                end_frame = video_info['total_frames']
                final_duration = (end_frame - start_frame) / fps
                if final_duration >= min_clip_duration:
                    timestamps.append((start_frame, end_frame))
                    print(f"Found final clip from frame {start_frame} to {end_frame} "
                          f"(duration: {final_duration:.2f}s)")

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            traceback.print_exc()
        finally:
            cap.release()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()

        return self._save_clips(video_path, output_path, timestamps, jersey_color, 
                              jersey_number, fps)

    def _save_clips(self, video_path, output_path, timestamps, jersey_color, 
                   jersey_number, fps):
        """Save detected clips and create highlight video"""
        if not timestamps:
            print("\nNo clips found for the specified player")
            return []
            
        print(f"\nFound {len(timestamps)} clips")
        os.makedirs(output_path, exist_ok=True)
        
        saved_clips = []
        for i, (start, end) in enumerate(timestamps):
            try:
                clip_duration = (end - start) / fps
                output_filename = (f"{output_path}/player_{jersey_number}_"
                                 f"{jersey_color}_clip_{i}_{clip_duration:.1f}s.mp4")
                extract_video_clip(video_path, output_filename, start, end, fps)
                saved_clips.append(output_filename)
                print(f"Saved clip {i+1}/{len(timestamps)}: {output_filename}")
            except Exception as e:
                print(f"Error saving clip {i}: {str(e)}")
        
        # Create highlight video
        if saved_clips:
            self.create_highlight_video(saved_clips, output_path, jersey_number)
        
        return saved_clips

    def create_highlight_video(self, clips, output_path, jersey_number):
        """Combine all clips into a single highlight video"""
        if not clips:
            return None
            
        print("\nCreating highlight video...")
        highlight_path = f"{output_path}/player_{jersey_number}_highlights.mp4"
        
        try:
            # Get first clip properties
            cap = cv2.VideoCapture(clips[0])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Create output video
            out = cv2.VideoWriter(highlight_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (width, height))
            
            # Add each clip to highlight video
            for i, clip_path in enumerate(clips):
                print(f"Adding clip {i+1}/{len(clips)} to highlight video")
                cap = cv2.VideoCapture(clip_path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                cap.release()
                
                # Add transition frames between clips
                if i < len(clips) - 1:
                    transition_frames = int(fps)  # 1 second transition
                    for _ in range(transition_frames):
                        transition_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        out.write(transition_frame)
            
            out.release()
            print(f"Created highlight video: {highlight_path}")
            return highlight_path
            
        except Exception as e:
            print(f"Error creating highlight video: {str(e)}")
            return None

    def __del__(self):
        """Cleanup when detector is destroyed"""
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()