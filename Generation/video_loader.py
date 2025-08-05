#!/usr/bin/env python3
"""
Video and summary data loader
"""
import os
import json
import pandas as pd
from typing import Optional, Dict, Any
try:
    from .utils import log_message
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import log_message


class VideoLoader:
    """Handles loading video files and associated summary data"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
    
    def load_video_summary(self, video_path: str) -> Optional[str]:
        """Load video summary from JSON file"""
        try:
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            summary_path = os.path.join(video_dir, f"{video_name}_summary.json")
            
            if not os.path.exists(summary_path):
                return None
            
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # Convert summary data to text format
            segments_text = []
            for segment in summary_data:
                start_time = segment['start_time']
                end_time = segment['end_time']
                description = segment['description']
                segments_text.append(f"{start_time:.1f}s-{end_time:.1f}s: {description}")
            
            return "\n".join(segments_text)
            
        except Exception as e:
            log_message(f"Failed to load summary file: {str(e)}")
            return None
    
    def load_gaze_data(self, video_path: str) -> Optional[pd.DataFrame]:
        """Load gaze tracking data from CSV file"""
        try:
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
            
            if not os.path.exists(csv_path):
                return None
            
            return pd.read_csv(csv_path)
            
        except Exception:
            return None
    
    def get_video_path(self, sequence_id: str) -> str:
        """Get full video path from sequence ID"""
        video_dir = os.path.join(self.dataset_path, sequence_id)
        return os.path.join(video_dir, f"{sequence_id}.mp4")
    
    def validate_video_exists(self, video_path: str) -> bool:
        """Check if video file exists"""
        return os.path.exists(video_path)


def main():
    """Test video loader functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test video loader')
    parser.add_argument('--dataset-path', required=True, help='Dataset directory path')
    parser.add_argument('--sequence-id', required=True, help='Video sequence ID to test')
    
    args = parser.parse_args()
    
    loader = VideoLoader(args.dataset_path)
    video_path = loader.get_video_path(args.sequence_id)
    
    log_message(f"Testing video loader for sequence: {args.sequence_id}")
    log_message(f"Video path: {video_path}")
    log_message(f"Video exists: {loader.validate_video_exists(video_path)}")
    
    # Test summary loading
    summary = loader.load_video_summary(video_path)
    if summary:
        log_message("Summary loaded successfully")
        print("Summary content:")
        print(summary)
    else:
        log_message("No summary file found")
    
    # Test gaze data loading
    gaze_data = loader.load_gaze_data(video_path)
    if gaze_data is not None:
        log_message(f"Gaze data loaded: {len(gaze_data)} frames")
    else:
        log_message("No gaze data found")


if __name__ == "__main__":
    main() 