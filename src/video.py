"""Video processing functions"""
import cv2
import os

def extract_video_clip(video_path, output_path, start_frame, end_frame, fps):
    """Extract a clip from the video between start_frame and end_frame"""
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        total_frames = end_frame - start_frame
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"Extracting clip: {frame_count}/{total_frames} frames "
                      f"({(frame_count/total_frames*100):.1f}%)")
        
        print(f"Successfully extracted clip: {output_path}")
        
    except Exception as e:
        print(f"Error extracting clip: {str(e)}")
        raise
    
    finally:
        cap.release()
        out.release()

def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration': total_frames / fps
    }