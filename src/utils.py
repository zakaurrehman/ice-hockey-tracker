"""Utility functions for the player tracker"""
import cv2
import numpy as np
import os
import torch
from .config import (
    JERSEY_COLORS, DEBUG_COLORS, DEBUG_OUTPUT_DIR, SAVE_DEBUG_FRAMES,
    IMAGE_PROCESSING, DEBUG_FOLDERS, BBOX_PADDING
)

def validate_bbox(frame, bbox):
    """Strictly validate and adjust bounding box coordinates"""
    try:
        if frame is None or bbox is None or len(bbox) != 4:
            return None

        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Convert to float first to handle any numeric type
        try:
            bbox = [float(x) for x in bbox]
        except (TypeError, ValueError):
            return None
        
        # Convert to int after rounding
        x1 = int(round(max(0, min(bbox[0], width - 1))))
        y1 = int(round(max(0, min(bbox[1], height - 1))))
        x2 = int(round(max(0, min(bbox[2], width - 1))))
        y2 = int(round(max(0, min(bbox[3], height - 1))))
        
        # Ensure box is valid
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Ensure minimum size
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return None
            
        # Final validation
        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            return None
            
        return [x1, y1, x2, y2]
        
    except Exception as e:
        print(f"Error validating bbox: {str(e)}")
        return None

def print_gpu_utilization():
    """Print detailed GPU memory usage and device information"""
    try:
        if torch.cuda.is_available():
            # Get detailed memory stats
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"GPU Memory Usage:")
            print(f" - Allocated: {allocated:.2f} MB")
            print(f" - Reserved: {reserved:.2f} MB")
            print(f" - Max Allocated: {max_allocated:.2f} MB")
            
            # Additional GPU info
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            print(f"GPU Device: {props.name}")
            print(f"Total Memory: {props.total_memory / 1024**2:.2f} MB")
            
            # Reset peak stats for next measurement
            torch.cuda.reset_peak_memory_stats()
            
    except Exception as e:
        print(f"Error checking GPU utilization: {str(e)}")

def is_jersey_color(frame, bbox, color_name, color_threshold=100):
    """Check jersey color with safe region extraction"""
    try:
        if color_name not in JERSEY_COLORS or frame is None:
            return False

        # Extract and validate coordinates
        x1, y1, x2, y2 = map(int, map(float, bbox))
        height, width = frame.shape[:2]
        
        # Strict boundary checking
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x2 <= x1 or y2 <= y1:
            return False

        # Safe region extraction
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return False

        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Color matching
        total_matches = 0
        for lower, upper in JERSEY_COLORS[color_name]:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            match_percentage = (np.sum(mask > 0) / region.size) * 100
            if match_percentage > 5:
                return True
            total_matches += match_percentage

        return total_matches > 8
        
    except Exception as e:
        print(f"Error in jersey color detection: {str(e)}")
        return False
def enhance_number_region(bbox):
    """Enhanced number region processing with boundary checks"""
    try:
        if bbox is None or bbox.size == 0 or len(bbox.shape) != 3:
            return None

        # Calculate new dimensions
        height, width = bbox.shape[:2]
        resize_factor = IMAGE_PROCESSING['resize_factor']
        max_dim = IMAGE_PROCESSING['max_dimension']
        
        # Calculate target size maintaining aspect ratio
        if width > height:
            new_width = min(int(width * resize_factor), max_dim)
            scale = new_width / width
            new_height = int(height * scale)
        else:
            new_height = min(int(height * resize_factor), max_dim)
            scale = new_height / height
            new_width = int(width * scale)

        # Ensure valid dimensions
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Resize image
        resized = cv2.resize(bbox, (new_width, new_height), 
                           interpolation=cv2.INTER_LINEAR)

        # Enhance contrast
        enhanced = cv2.convertScaleAbs(resized, 
                                     alpha=IMAGE_PROCESSING['contrast_alpha'],
                                     beta=IMAGE_PROCESSING['contrast_beta'])
        del resized

        # Optional denoising
        if IMAGE_PROCESSING['enable_denoising']:
            denoised = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                IMAGE_PROCESSING['denoise_strength'],
                IMAGE_PROCESSING['denoise_strength'],
                7,
                21
            )
            del enhanced
            enhanced = denoised

        # Optional sharpening
        if IMAGE_PROCESSING['enable_sharpening']:
            sharpened = cv2.filter2D(
                enhanced,
                -1,
                IMAGE_PROCESSING['sharpen_kernel']
            )
            del enhanced
            enhanced = sharpened

        return enhanced

    except Exception as e:
        print(f"Error enhancing number region: {str(e)}")
        return bbox
    finally:
        # Clean up any temporary arrays
        for name in ['resized', 'denoised', 'sharpened']:
            if name in locals():
                del locals()[name]

def save_debug_image(image, path, label=""):
    """Save debug image with memory optimization"""
    try:
        if image is None:
            return
            
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create copy to avoid modifying original
        debug_image = image.copy()
        
        if label:
            # Calculate optimal text size
            height = debug_image.shape[0]
            font_scale = height / 1000.0
            thickness = max(1, int(height / 500))
            
            # Add background for better visibility
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )[0]
            
            cv2.rectangle(
                debug_image,
                (5, 5),
                (text_size[0] + 15, text_size[1] + 15),
                DEBUG_COLORS['text_color'],
                -1
            )
            
            # Add text
            cv2.putText(
                debug_image,
                label,
                (10, text_size[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text on white background
                thickness
            )
        
        # Save with optimized compression
        cv2.imwrite(path, debug_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
    except Exception as e:
        print(f"Error saving debug image: {str(e)}")
    finally:
        if 'debug_image' in locals():
            del debug_image

def create_debug_visualization(frame, bbox, color_match, number_match, jersey_number):
    """Create debug visualization with boundary checks"""
    try:
        if frame is None:
            return None
            
        debug_frame = frame.copy()
        
        # Validate and adjust bbox
        valid_bbox = validate_bbox(frame, bbox)
        if valid_bbox is None:
            return frame
            
        x1, y1, x2, y2 = valid_bbox
        
        # Determine visualization style
        if number_match:
            color = DEBUG_COLORS['detected']
            label = f"#{jersey_number}"
            confidence = "High"
        elif color_match:
            color = DEBUG_COLORS['color_match']
            label = "Color Match"
            confidence = "Medium"
        else:
            color = DEBUG_COLORS['no_match']
            label = "No Match"
            confidence = "Low"

        # Calculate text scale
        height = debug_frame.shape[0]
        font_scale = height / 1000.0
        thickness = max(1, int(height / 500))
        
        # Draw bounding box
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add label with background
        label_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )[0]
        
        # Draw label background
        cv2.rectangle(
            debug_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )
        
        # Add text
        cv2.putText(
            debug_frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
        
        # Add confidence
        conf_pos = (x1, y1 - label_size[1] - 15)
        cv2.putText(
            debug_frame,
            confidence,
            conf_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            color,
            thickness
        )

        return debug_frame

    except Exception as e:
        print(f"Error creating debug visualization: {str(e)}")
        return frame
    finally:
        # Clean up if error occurred
        if 'debug_frame' in locals() and id(debug_frame) != id(frame):
            del debug_frame