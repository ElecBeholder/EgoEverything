#!/usr/bin/env python3
"""
Aria Everyday Activities dataset downloader and processor
"""

import json
import subprocess
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
import argparse
import time
import multiprocessing as mp
from typing import Optional, List

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

try:
    import mediapipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class AriaDownloaderProcessor:
    """Aria dataset downloader and processor"""
    
    def __init__(self, json_path: str = "Data/AriaEverydayActivities_download_urls.json", 
                 output_dir: str = "Data"):
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_sequences(self, limit: Optional[int] = None) -> List[str]:
        """Load sequence list"""
        with open(self.json_path, 'r') as f:
            sequences = list(json.load(f)['sequences'].keys())
        if limit:
            sequences = sequences[:limit]
        print(f"Found {len(sequences)} sequences to process")
        return sequences
    
    def download_and_process_all(self, limit: Optional[int] = None):
        """Download and process all sequences with multiprocessing"""
        sequences = self.load_sequences(limit)
        total = len(sequences)
        
        download_queue = mp.Queue()
        process_queue = mp.Queue()
        
        for seq in sequences:
            download_queue.put(seq)
        
        manager = mp.Manager()
        download_counter = manager.Value('i', 0)
        process_counter = manager.Value('i', 0)
        
        download_process = mp.Process(target=download_worker, 
                                    args=(download_queue, process_queue, download_counter, total, 
                                          str(self.json_path), str(self.output_dir)))
        process_process = mp.Process(target=process_worker, 
                                   args=(download_queue, process_queue, process_counter, total, str(self.output_dir)))
        
        download_process.start()
        process_process.start()
        download_process.join()
        process_process.join()
        
        print("All tasks completed!")


def download_worker(download_queue, process_queue, counter, total, json_path, output_dir):
    """Download worker process"""
    while True:
        try:
            sequence = download_queue.get_nowait()
            counter.value += 1
            
            # Check if output files already exist
            seq_dir = Path(output_dir) / sequence
            mp4_file = seq_dir / f"{sequence}.mp4"
            csv_file = seq_dir / f"{sequence}_tracking.csv"
            
            if seq_dir.exists() and mp4_file.exists() and csv_file.exists():
                print(f"Skipping {sequence} - already processed [{counter.value}/{total}]")
                continue
            
            print(f"Downloading {sequence}... [{counter.value}/{total}]")
            
            cmd_vrs = ["aria_dataset_downloader", "-c", json_path, "-o", output_dir, "-l", sequence, "--data_types", "0"]
            cmd_gaze = ["aria_dataset_downloader", "-c", json_path, "-o", output_dir, "-l", sequence, "--data_types", "4"]
            
            result1 = subprocess.run(cmd_vrs, capture_output=True, text=True)
            result2 = subprocess.run(cmd_gaze, capture_output=True, text=True)
            
            if result1.returncode == 0 and result2.returncode == 0:
                print(f"Downloaded {sequence} [{counter.value}/{total}]")
                process_queue.put(sequence)
        except:
            break


def process_worker(download_queue, process_queue, counter, total, output_dir):
    """Process worker process"""
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
    
    # Initialize MediaPipe
    hands = None
    if MEDIAPIPE_AVAILABLE:
        try:
            import mediapipe
            mp_hands = mediapipe.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                                 min_detection_confidence=0.1, min_tracking_confidence=0.1)
        except:
            pass
    
    def get_hand_centers(frame):
        if not hands:
            return None, None
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            left_hand_center = None
            right_hand_center = None
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    h, w = frame.shape[:2]
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    hand_label = handedness.classification[0].label
                    if hand_label == 'Left':
                        right_hand_center = (center_x, center_y)
                    else:
                        left_hand_center = (center_x, center_y)
            return left_hand_center, right_hand_center
        except:
            return None, None
    
    def get_gaze_point(mps, timestamp_ns, device_calib, rgb_calib):
        try:
            gaze = mps.get_general_eyegaze(timestamp_ns)
            if gaze is None:
                return None
            depth_m = gaze.depth if hasattr(gaze, 'depth') and gaze.depth else 1.0
            projection = get_gaze_vector_reprojection(gaze, "camera-rgb", device_calib, rgb_calib, depth_m)
            if projection is not None:
                return (int(projection[0]), int(projection[1]))
            return None
        except:
            return None
    
    while True:
        try:
            sequence = process_queue.get_nowait()
            counter.value += 1
            print(f"Processing {sequence} [{counter.value}/{total}]")
            
            seq_dir = Path(output_dir) / sequence
            vrs_files = list(seq_dir.glob("*.vrs"))
            if not vrs_files:
                continue
            
            # Find eye gaze CSV directly in mps/eye_gaze
            extract_dir = seq_dir / "mps" / "eye_gaze"
            if not extract_dir.exists():
                continue
            csv_files = list(extract_dir.glob("*.csv"))
            if not csv_files:
                continue
            
            vrs_provider = data_provider.create_vrs_data_provider(str(vrs_files[0]))
            if not vrs_provider:
                continue
            
            device_calib = vrs_provider.get_device_calibration()
            rgb_stream = StreamId("214-1")
            rgb_calib = device_calib.get_camera_calib("camera-rgb")
            
            paths_provider = MpsDataPathsProvider(str(seq_dir / "mps"))
            mps = MpsDataProvider(paths_provider.get_data_paths())
            
            if not mps.has_general_eyegaze():
                continue
            
            config = vrs_provider.get_image_configuration(rgb_stream)
            width, height = config.image_width, config.image_height
            total_frames = vrs_provider.get_num_data(rgb_stream)
            
            video_output = seq_dir / f"{sequence}.mp4"
            csv_output = seq_dir / f"{sequence}_tracking.csv"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_output), fourcc, 20, (height, width))
            tracking_data = []
            
            for frame_idx in range(total_frames):
                image_data = vrs_provider.get_image_data_by_index(rgb_stream, frame_idx)
                if not image_data:
                    continue
                
                frame = image_data[0].to_numpy_array()
                timestamp_ns = image_data[1].capture_timestamp_ns
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                gaze_point = get_gaze_point(mps, timestamp_ns, device_calib, rgb_calib)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                left_hand, right_hand = get_hand_centers(frame)
                
                tracking_data.append({
                    'frame_idx': frame_idx,
                    'timestamp_ns': timestamp_ns,
                    'gaze_x': height - gaze_point[1] if gaze_point else None,
                    'gaze_y': gaze_point[0] if gaze_point else None,
                    'left_hand_x': left_hand[0] if left_hand else None,
                    'left_hand_y': left_hand[1] if left_hand else None,
                    'right_hand_x': right_hand[0] if right_hand else None,
                    'right_hand_y': right_hand[1] if right_hand else None
                })
                out.write(frame)
            
            out.release()
            pd.DataFrame(tracking_data).to_csv(csv_output, index=False)
            
            # Cleanup
            for f in seq_dir.glob("*.vrs"):
                f.unlink()
            if (seq_dir / "mps").exists():
                shutil.rmtree(seq_dir / "mps")
            if (seq_dir / ".download_status.json").exists():
                (seq_dir / ".download_status.json").unlink()
            
            print(f"Processed {sequence} [{counter.value}/{total}]")
            
        except:
            time.sleep(1)
            if download_queue.empty() and process_queue.empty():
                break


def main():
    parser = argparse.ArgumentParser(description="Aria dataset downloader and processor")
    parser.add_argument("--json", default="Data/AriaEverydayActivities_download_urls.json",
                       help="Download URLs JSON file path")
    parser.add_argument("--output", default="Data", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    
    args = parser.parse_args()
    
    processor = AriaDownloaderProcessor(json_path=args.json, output_dir=args.output)
    processor.download_and_process_all(limit=args.limit)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main() 