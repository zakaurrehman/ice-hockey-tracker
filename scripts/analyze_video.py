"""Script to analyze hockey video and detect players"""
import argparse
from pathlib import Path
import os
import sys
from tqdm import tqdm
from datetime import datetime

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.video_analyzer import VideoAnalyzer
from src.enhanced_tracker import EnhancedPlayerTracker
from src.config import BATCH_SIZE, OCR_SETTINGS

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze hockey video for player detection')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--chunk-duration', type=int, default=300,
                       help='Duration of each chunk in seconds (default: 300)')
    parser.add_argument('--jersey-number', type=int, help='Jersey number to track')
    parser.add_argument('--jersey-color', type=str, help='Jersey color to track')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save debug frames')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Run analysis only without tracking')
    return parser.parse_args()

def create_output_dirs(base_dir):
    """Create output directories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    
    # Create main output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'frames').mkdir(exist_ok=True)
    (output_dir / 'highlights').mkdir(exist_ok=True)
    (output_dir / 'debug').mkdir(exist_ok=True)
    
    return output_dir

def analyze_video(video_path, output_dir, jersey_number=None, jersey_color=None):
    """Run video analysis"""
    try:
        print(f"\nAnalyzing video: {video_path}")
        print("Initializing analyzer...")
        
        analyzer = VideoAnalyzer(video_path)
        segments = analyzer.analyze_video(jersey_number, jersey_color)
        
        print("\nAnalysis Results:")
        for i, segment in enumerate(segments):
            print(f"\nSegment {i+1}:")
            print(f"  Duration: {segment.get_duration()} frames")
            print(f"  Players detected: {len(segment.players)}")
            for player in sorted(segment.players):
                print(f"    - {player}")
                
        # Extract highlights if specific player was requested
        if jersey_number and jersey_color:
            print("\nExtracting highlights...")
            highlights_dir = output_dir / 'highlights'
            highlights = analyzer.extract_highlights(str(highlights_dir),
                                                  jersey_number,
                                                  jersey_color)
            if highlights:
                print(f"\nCreated {len(highlights)} highlight clips:")
                for clip in highlights:
                    print(f"  - {clip}")
            else:
                print("No highlights created.")
                
        # Generate summary
        summary_path = output_dir / 'analysis_summary.txt'
        _save_analysis_summary(summary_path, segments, analyzer.get_player_summary())
        print(f"\nSaved analysis summary to: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False

def track_player(video_path, output_dir, jersey_number, jersey_color):
    """Run player tracking"""
    try:
        print(f"\nTracking player {jersey_number} ({jersey_color}) in video: {video_path}")
        print("Initializing tracker...")
        
        tracker = EnhancedPlayerTracker(jersey_number, jersey_color)
        
        highlights_dir = output_dir / 'highlights'
        output_path = tracker.track_player(
            video_path,
            str(highlights_dir),
            lambda x: print(f"Progress: {x}%", end='\r')
        )
        
        if output_path:
            print(f"\nCreated highlight video: {output_path}")
            
            # Get and save player stats
            stats = tracker.get_player_stats()
            stats_path = output_dir / 'tracking_stats.txt'
            _save_tracking_stats(stats_path, stats)
            print(f"Saved tracking statistics to: {stats_path}")
            
            return True
        else:
            print("Failed to create highlight video.")
            return False
            
    except Exception as e:
        print(f"Error during tracking: {e}")
        return False
    finally:
        if 'tracker' in locals():
            tracker.cleanup()

def _save_analysis_summary(path, segments, player_stats):
    """Save analysis summary to file"""
    with open(path, 'w') as f:
        f.write("VIDEO ANALYSIS SUMMARY\n")
        f.write("=====================\n\n")
        
        f.write("Segments Found\n")
        f.write("-------------\n")
        for i, segment in enumerate(segments):
            f.write(f"\nSegment {i+1}:\n")
            f.write(f"  Duration: {segment.get_duration()} frames\n")
            f.write(f"  Players detected: {len(segment.players)}\n")
            for player in sorted(segment.players):
                f.write(f"    - {player}\n")
        
        f.write("\nPlayer Statistics\n")
        f.write("----------------\n")
        for player, stats in player_stats.items():
            f.write(f"\n{player}:\n")
            f.write(f"  Total Frames: {stats['total_frames']}\n")
            f.write(f"  Segments: {stats['segments']}\n")

def _save_tracking_stats(path, stats):
    """Save tracking statistics to file"""
    with open(path, 'w') as f:
        f.write("PLAYER TRACKING STATISTICS\n")
        f.write("========================\n\n")
        
        for player_key, player_stats in stats.items():
            f.write(f"Player: {player_key}\n")
            f.write(f"Total Frames: {player_stats['total_frames']}\n")
            f.write(f"Segments: {len(player_stats['segments'])}\n")
            
            f.write("\nSegments:\n")
            for start, end in player_stats['segments']:
                duration = end - start
                f.write(f"  {start} - {end} (duration: {duration} frames)\n")
            f.write("\n")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate video file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
        
    # Create output directory
    output_dir = create_output_dirs(args.output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    success = False
    if args.analyze_only:
        success = analyze_video(
            str(video_path),
            output_dir,
            args.jersey_number,
            args.jersey_color
        )
    else:
        if not args.jersey_number or not args.jersey_color:
            print("Error: Jersey number and color required for tracking mode")
            return 1
            
        success = track_player(
            str(video_path),
            output_dir,
            args.jersey_number,
            args.jersey_color
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())