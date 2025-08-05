#!/usr/bin/env python3
"""
Frame extraction and video clustering functionality
"""
import cv2
import random
import torch
import numpy as np
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    from .utils import log_message
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import log_message


class FrameExtractor:
    """Handles frame extraction from videos"""
    
    def __init__(self):
        pass
    
    def extract_random_frame(self, video_path: str, temp_dir: str) -> Tuple[str, float]:
        """Extract a random frame from video and return path and timestamp"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Select random frame
        frame_num = random.randint(0, total_frames - 1)
        timestamp = frame_num / fps
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise Exception("Cannot read frame")
        
        # Save frame to temp directory
        output_path = os.path.join(temp_dir, f"keyframe_{timestamp:.1f}s.jpg")
        cv2.imwrite(output_path, frame)
        cap.release()
        
        return output_path, timestamp
    
    def extract_frame_at_timestamp(self, video_path: str, timestamp: float, temp_dir: str) -> str:
        """Extract frame at specific timestamp"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(timestamp * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = max(0, min(frame_num, total_frames - 1))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise Exception(f"Cannot read frame at timestamp {timestamp}s")
        
        output_path = os.path.join(temp_dir, f"frame_{timestamp:.1f}s.jpg")
        cv2.imwrite(output_path, frame)
        cap.release()
        
        return output_path


class SegmentFeatureExtractor:
    """Video segment feature extraction and clustering"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize ResNet for feature extraction
        self.feature_model = resnet50(pretrained=True)
        self.feature_model.fc = torch.nn.Identity()
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_video_segment(self, video_path: str, start_second: float, end_second: float, output_path: str) -> bool:
        """Extract video segment using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start_second),
                '-t', str(end_second - start_second),
                '-c', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_message(f"ffmpeg error: {result.stderr}")
                return False
            
            return os.path.exists(output_path)
        except Exception as e:
            log_message(f"Video segment extraction failed: {str(e)}")
            return False
    
    def extract_features_from_segment(self, video_path: str, sample_rate: int = 5) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """Extract features from video segment"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        features = []
        frame_timestamps = []
        frame_count = 0
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                    feature = self.feature_model(input_tensor)
                    features.append(feature.cpu().numpy())
                    frame_timestamps.append(frame_count / fps)
                
                frame_count += 1
        
        cap.release()
        
        if not features:
            return None, None
            
        return np.concatenate(features, axis=0), frame_timestamps
    
    def cluster_segment_frames(self, features: np.ndarray, n_clusters: int = 10) -> np.ndarray:
        """Cluster frame features"""
        if len(features) < n_clusters:
            n_clusters = max(1, len(features))
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predictions = kmeans.fit_predict(features_scaled)
        
        return predictions
    
    def get_representative_frames(self, video_path: str, start_second: float, end_second: float, 
                                 n_clusters: int = 10, temp_dir: str = None) -> List[Dict[str, Any]]:
        """Get representative frames from video segment"""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract video segment
            segment_path = os.path.join(temp_dir, "segment.mp4")
            if not self.extract_video_segment(video_path, start_second, end_second, segment_path):
                return []
            
            # Extract features
            features, frame_timestamps = self.extract_features_from_segment(segment_path)
            if features is None:
                return []
            
            # Cluster
            predictions = self.cluster_segment_frames(features, n_clusters)
            
            # Find representative frames for each cluster
            representative_frames = []
            unique_clusters = np.unique(predictions)
            
            for cluster_id in unique_clusters:
                cluster_indices = np.where(predictions == cluster_id)[0]
                cluster_features = features[cluster_indices]
                
                # Calculate cluster center
                cluster_center = np.mean(cluster_features, axis=0)
                
                # Find frame closest to center
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                center_idx = cluster_indices[np.argmin(distances)]
                
                # Calculate actual timestamp in original video
                relative_timestamp = frame_timestamps[center_idx]
                actual_timestamp = start_second + relative_timestamp
                
                representative_frames.append({
                    'cluster_id': int(cluster_id),
                    'timestamp': actual_timestamp,
                    'relative_timestamp': relative_timestamp
                })
            
            # Sort by timestamp
            representative_frames.sort(key=lambda x: x['timestamp'])
            
            return representative_frames
            
        except Exception as e:
            log_message(f"Representative frame extraction failed: {str(e)}")
            return []


def main():
    """Test frame extraction functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test frame extractor')
    parser.add_argument('--video-path', required=True, help='Video file path')
    parser.add_argument('--temp-dir', default='tmp', help='Temporary directory')
    parser.add_argument('--test-clustering', action='store_true', help='Test clustering functionality')
    
    args = parser.parse_args()
    
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Test basic frame extraction
    extractor = FrameExtractor()
    log_message(f"Testing frame extraction on: {args.video_path}")
    
    try:
        frame_path, timestamp = extractor.extract_random_frame(args.video_path, args.temp_dir)
        log_message(f"Random frame extracted: {frame_path} at {timestamp:.1f}s")
        
        # Test timestamp-based extraction
        test_timestamp = 30.0
        frame_path_2 = extractor.extract_frame_at_timestamp(args.video_path, test_timestamp, args.temp_dir)
        log_message(f"Frame at {test_timestamp}s extracted: {frame_path_2}")
        
    except Exception as e:
        log_message(f"Frame extraction failed: {str(e)}")
    
    # Test clustering if requested
    if args.test_clustering:
        try:
            log_message("Testing clustering functionality...")
            segment_extractor = SegmentFeatureExtractor()
            
            # Test on first 60 seconds
            representative_frames = segment_extractor.get_representative_frames(
                args.video_path, 0, 60, n_clusters=5, temp_dir=args.temp_dir
            )
            
            log_message(f"Found {len(representative_frames)} representative frames:")
            for frame in representative_frames:
                log_message(f"  Cluster {frame['cluster_id']}: {frame['timestamp']:.1f}s")
                
        except Exception as e:
            log_message(f"Clustering test failed: {str(e)}")


if __name__ == "__main__":
    main() 