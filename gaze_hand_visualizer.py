#!/usr/bin/env python3
"""
Gaze and Hand Tracking Visualizer
Add gaze points and hand tracking to MP4 videos
"""

import cv2
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple


class GazeHandVisualizer:
    """Visualize gaze and hand tracking on videos"""
    
    def __init__(self):
        # Visualization colors (BGR format)
        self.gaze_color = (0, 255, 0)      # Green for gaze
        self.left_hand_color = (255, 0, 0)  # Blue for left hand
        self.right_hand_color = (0, 0, 255) # Red for right hand
        
        # Point sizes
        self.gaze_radius = 8
        self.hand_radius = 6
    
    def draw_tracking_points(self, frame, gaze_point, left_hand, right_hand):
        """Draw gaze and hand tracking points"""
        height, width = frame.shape[:2]
        
        # Draw current points with rings
        if gaze_point and 0 <= gaze_point[0] < width and 0 <= gaze_point[1] < height:
            cv2.circle(frame, gaze_point, self.gaze_radius, self.gaze_color, 2)
            cv2.circle(frame, gaze_point, self.gaze_radius + 3, self.gaze_color, 1)
        
        if left_hand and 0 <= left_hand[0] < width and 0 <= left_hand[1] < height:
            cv2.circle(frame, left_hand, self.hand_radius, self.left_hand_color, 2)
            cv2.circle(frame, left_hand, self.hand_radius + 2, self.left_hand_color, 1)
            
        if right_hand and 0 <= right_hand[0] < width and 0 <= right_hand[1] < height:
            cv2.circle(frame, right_hand, self.hand_radius, self.right_hand_color, 2)
            cv2.circle(frame, right_hand, self.hand_radius + 2, self.right_hand_color, 1)
    
    def add_legend(self, frame):
        """Add legend to the frame"""
        height, width = frame.shape[:2]
        
        # Legend background
        legend_height = 80
        legend_width = 200
        legend_x = width - legend_width - 10
        legend_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Legend text and symbols
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Gaze
        cv2.circle(frame, (legend_x + 15, legend_y + 20), 5, self.gaze_color, -1)
        cv2.putText(frame, "Gaze", (legend_x + 30, legend_y + 25), font, font_scale, (255, 255, 255), 1)
        
        # Left hand
        cv2.circle(frame, (legend_x + 15, legend_y + 40), 5, self.left_hand_color, -1)
        cv2.putText(frame, "Left Hand", (legend_x + 30, legend_y + 45), font, font_scale, (255, 255, 255), 1)
        
        # Right hand
        cv2.circle(frame, (legend_x + 15, legend_y + 60), 5, self.right_hand_color, -1)
        cv2.putText(frame, "Right Hand", (legend_x + 30, legend_y + 65), font, font_scale, (255, 255, 255), 1)
    
    def visualize_sequence(self, video_path: str, csv_path: str, output_path: str):
        """Create visualization for a sequence"""
        video_path = Path(video_path)
        csv_path = Path(csv_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return False
            
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}")
            return False
        
        # Load tracking data
        print(f"Loading tracking data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Open video
        print(f"Processing video {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process each frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get tracking data for this frame
            if frame_idx < len(df):
                row = df.iloc[frame_idx]
                
                # Extract coordinates
                gaze_point = None
                if pd.notna(row['gaze_x']) and pd.notna(row['gaze_y']):
                    gaze_point = (int(row['gaze_x']), int(row['gaze_y']))
                
                left_hand = None
                if pd.notna(row['left_hand_x']) and pd.notna(row['left_hand_y']):
                    left_hand = (int(row['left_hand_x']), int(row['left_hand_y']))
                
                right_hand = None
                if pd.notna(row['right_hand_x']) and pd.notna(row['right_hand_y']):
                    right_hand = (int(row['right_hand_x']), int(row['right_hand_y']))
                
                # Draw tracking points
                self.draw_tracking_points(frame, gaze_point, left_hand, right_hand)
            
            # Add legend
            self.add_legend(frame)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Visualization saved to: {output_path}")
        return True
    
    def process_directory(self, data_dir: str, output_suffix: str = "_with_tracking"):
        """Process all sequences in a directory"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Directory not found: {data_path}")
            return
        
        # Find all sequence directories
        sequences = []
        for item in data_path.iterdir():
            if item.is_dir() and (item / f"{item.name}.mp4").exists() and (item / f"{item.name}_tracking.csv").exists():
                sequences.append(item.name)
        
        if not sequences:
            print("No valid sequences found")
            return
        
        print(f"Found {len(sequences)} sequences to process")
        
        for i, seq_name in enumerate(sequences):
            print(f"\nProcessing sequence {i+1}/{len(sequences)}: {seq_name}")
            
            seq_dir = data_path / seq_name
            video_path = seq_dir / f"{seq_name}.mp4"
            csv_path = seq_dir / f"{seq_name}_tracking.csv"
            output_path = seq_dir / f"{seq_name}{output_suffix}.mp4"
            
            success = self.visualize_sequence(str(video_path), str(csv_path), str(output_path))
            if success:
                print(f"✓ Completed: {seq_name}")
            else:
                print(f"✗ Failed: {seq_name}")
        
        print(f"\nProcessed {len(sequences)} sequences")


def main():
    parser = argparse.ArgumentParser(description="Visualize gaze and hand tracking on videos")
    parser.add_argument("--input", help="Input video file or directory")
    parser.add_argument("--csv", help="CSV tracking file (required if input is single file)")
    parser.add_argument("--output", help="Output video file (required if input is single file)")
    parser.add_argument("--data_dir", default="Data", help="Data directory to process all sequences")
    parser.add_argument("--suffix", default="_with_tracking", help="Output file suffix")
    
    args = parser.parse_args()
    
    visualizer = GazeHandVisualizer()
    
    if args.input:
        # Process single file
        if not args.csv or not args.output:
            print("Error: --csv and --output are required when processing single file")
            return
        
        visualizer.visualize_sequence(args.input, args.csv, args.output)
    else:
        # Process directory
        visualizer.process_directory(args.data_dir, args.suffix)


if __name__ == "__main__":
    main() 