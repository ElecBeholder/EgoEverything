#!/usr/bin/env python3
import os
import json
import torch
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
from openai import OpenAI
import threading
import queue
import time

# Default configurations
DEFAULT_API_KEY = "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"
DEFAULT_DATASET_PATH = "/home/wang/AriaEveryday_activaties/"
DEFAULT_JSON_PATH = "/home/wang/AriaEveryday_activaties/AriaEverydayActivities_download_urls.json"

class FeatureExtractor:
    """GPU-based feature extraction and clustering"""
    
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

    def extract_features(self, video_path, sample_rate=5):
        """Extract video features using ResNet"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        features = []
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
                
                frame_count += 1
        
        cap.release()
        return np.concatenate(features, axis=0), fps
    
    def cluster_segments(self, features, n_clusters=8):
        """Cluster features into segments"""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Adjust cluster number based on data size
        #effective_clusters = min(n_clusters, max(2, len(features) // 20))
        
        #kmeans = KMeans(n_clusters=effective_clusters, random_state=42)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        predictions = kmeans.fit_predict(features_scaled)
        
        # Temporal smoothing
        smoothed = predictions.copy()
        window_size = 5
        for i in range(len(predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            window_preds = predictions[start:end]
            unique, counts = np.unique(window_preds, return_counts=True)
            smoothed[i] = unique[np.argmax(counts)]
        
        return smoothed
    
    def predictions_to_segments(self, predictions, fps, min_duration=2.0):
        """Convert frame predictions to time segments"""
        segments = []
        if len(predictions) == 0:
            return segments
        
        current_action = predictions[0]
        start_frame = 0
        
        for i in range(1, len(predictions)):
            if predictions[i] != current_action:
                end_frame = i - 1
                duration = (end_frame - start_frame + 1) / fps
                
                if duration >= min_duration:
                    segments.append({
                        'start_time': start_frame / fps,
                        'end_time': (end_frame + 1) / fps,
                        'duration': duration
                    })
                
                current_action = predictions[i]
                start_frame = i
        
        # Handle last segment
        end_frame = len(predictions) - 1
        duration = (end_frame - start_frame + 1) / fps
        if duration >= min_duration:
            segments.append({
                'start_time': start_frame / fps,
                'end_time': (end_frame + 1) / fps,
                'duration': duration
            })
        
        return segments
    
    def process_video_features(self, video_path, n_clusters=15):
        """Extract features and generate segments"""
        features, fps = self.extract_features(video_path)
        effective_fps = fps / 5  # sample_rate = 5
        
        predictions = self.cluster_segments(features, n_clusters=n_clusters)
        segments = self.predictions_to_segments(predictions, effective_fps)
        
        return segments

class GeminiAnalyzer:
    """Gemini API-based video analysis"""
    
    def __init__(self, api_key):
        # Initialize Gemini client
        self.gemini_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def extract_key_frame(self, video_path, timestamp, output_path):
        """Extract frame at specific timestamp"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise Exception("Cannot read frame")
        
        cv2.imwrite(output_path, frame)
        cap.release()
        return output_path
    
    def encode_image_base64(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image type
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/jpeg'
        
        return f"data:{mime_type};base64,{base64_image}"
    
    def analyze_with_gemini(self, video_path, segments, tmp_dir):
        """Analyze key frames with Gemini 2.5 Pro"""
        # Extract key frames
        frame_paths = []
        segments_info = []
        
        for i, segment in enumerate(segments, 1):
            mid_time = (segment['start_time'] + segment['end_time']) / 2
            frame_path = f"{tmp_dir}/frame_{i}_{mid_time:.1f}s.jpg"
            self.extract_key_frame(video_path, mid_time, frame_path)
            frame_paths.append(frame_path)
            
            start_min, start_sec = divmod(int(segment['start_time']), 60)
            end_min, end_sec = divmod(int(segment['end_time']), 60)
            segments_info.append(f"{i}. {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}")
        
        segments_text = "\n".join(segments_info)
        
        # Build prompt
        prompt = f"""I will show you {len(segments)} key frames, each representing a time segment in the video. This is a first-person perspective video.

Time segment list:
{segments_text}

Please describe the main activities of the protagonist during each time segment based on these key frames.

Analysis requirements:
1. Each key frame corresponds to a time segment in order
2. Describe the main behaviors and actions the person might be performing during that time segment
3. Infer the person's behavior based on the environment, objects, and hand movements in the image
4. Keep descriptions concise and clear, no more than 20 words per time segment
5. Focus on the person's behavioral actions rather than environmental descriptions

Please respond in the following format:
1. 00:00 - 00:15: [Activity description]
2. 00:15 - 00:30: [Activity description]
...and so on

Please begin the analysis:"""
        
        # Build message content
        content = [{"type": "text", "text": prompt}]
        
        for frame_path in frame_paths:
            base64_image = self.encode_image_base64(frame_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            })
        
        messages = [
            {"role": "system", "content": "You are a professional video analysis assistant, skilled at understanding human behavior in first-person videos."},
            {"role": "user", "content": content}
        ]
        
        # Call Gemini API
        response = self.gemini_client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=messages,
            temperature=1,
            max_tokens=65536
        )
        
        response_text = response.choices[0].message.content if response.choices else ""
        return response_text.strip()
    
    def parse_gemini_response(self, response_text, segments):
        """Parse Gemini response into structured data"""
        descriptions = {}
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if '. ' in line and ':' in line and '-' in line:
                try:
                    parts = line.split('. ', 1)
                    if len(parts) == 2:
                        segment_num = int(parts[0])
                        rest = parts[1]
                        
                        if ':' in rest:
                            desc_part = rest.split(':', 1)
                            if len(desc_part) == 2:
                                description = desc_part[1].strip()
                                # Remove any remaining timestamp patterns and markdown symbols
                                import re
                                description = re.sub(r'^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\s*', '', description)
                                description = re.sub(r'^\d{2}\s*-\s*\d{2}:\d{2}:\s*', '', description)
                                description = re.sub(r'^\*+\s*', '', description)  # Remove markdown asterisks
                                description = description.strip()
                                descriptions[segment_num] = description
                except:
                    continue
        
        # Match descriptions to segments
        results = []
        for i, segment in enumerate(segments, 1):
            description = descriptions.get(i, "No description available")
            results.append({
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'description': description
            })
        
        return results
    
    def process_segments(self, video_path, segments, output_path, tmp_dir):
        """Process segments with Gemini analysis and save results"""
        if not segments:
            return None
        
        # Analyze with Gemini
        response_text = self.analyze_with_gemini(video_path, segments, tmp_dir)
        
        # Parse and format results
        results = self.parse_gemini_response(response_text, segments)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

def load_video_sequences(json_path):
    """Load video sequences from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    for seq_id, seq_data in data['sequences'].items():
        if 'video_main_rgb' in seq_data:
            video_info = seq_data['video_main_rgb']
            sequences.append({
                'sequence_id': seq_id,
                'filename': video_info['filename'],
                'download_url': video_info['download_url']
            })
    
    return sequences

def producer_worker(sequences, dataset_path, tmp_dir, n_clusters, work_queue, progress_lock, pbar, n_consumers):
    """Producer thread: Extract features and cluster videos"""
    feature_extractor = FeatureExtractor()
    
    for seq in sequences:
        seq_id = seq['sequence_id']
        video_dir = os.path.join(dataset_path, seq_id)
        video_path = os.path.join(video_dir, f"{seq_id}.mp4")
        
        with progress_lock:
            pbar.set_description(f"Extracting features: {seq_id}")
        
        if not os.path.exists(video_path):
            with progress_lock:
                pbar.set_description(f"Skipping {seq_id} (video not found)")
                pbar.update(1)
            continue
        
        output_path = os.path.join(video_dir, f"{seq_id}_summary.json")
        
        # Skip if already processed
        if os.path.exists(output_path):
            with progress_lock:
                pbar.set_description(f"Skipping {seq_id} (already processed)")
                pbar.update(1)
            continue
        
        try:
            # Extract features and cluster
            segments = feature_extractor.process_video_features(video_path, n_clusters)
            
            if segments:
                # Create sequence-specific tmp directory
                seq_tmp_dir = os.path.join(tmp_dir, seq_id)
                os.makedirs(seq_tmp_dir, exist_ok=True)
                
                # Push to queue for Gemini analysis
                work_item = {
                    'seq_id': seq_id,
                    'video_path': video_path,
                    'segments': segments,
                    'output_path': output_path,
                    'tmp_dir': seq_tmp_dir
                }
                work_queue.put(work_item)
                
                with progress_lock:
                    pbar.set_description(f"Features extracted: {seq_id}")
            else:
                with progress_lock:
                    pbar.set_description(f"No segments found: {seq_id}")
                    pbar.update(1)
                
        except Exception as e:
            with progress_lock:
                pbar.set_description(f"Error extracting features from {seq_id}: {str(e)}")
                pbar.update(1)
    
    # Signal end of work - send None for each consumer
    for _ in range(n_consumers):
        work_queue.put(None)

def consumer_worker(api_key, work_queue, progress_lock, pbar):
    """Consumer thread: Analyze segments with Gemini"""
    gemini_analyzer = GeminiAnalyzer(api_key)
    
    while True:
        work_item = work_queue.get()
        
        if work_item is None:
            # End of work signal
            work_queue.task_done()
            break
        
        seq_id = work_item['seq_id']
        video_path = work_item['video_path']
        segments = work_item['segments']
        output_path = work_item['output_path']
        tmp_dir = work_item['tmp_dir']
        
        with progress_lock:
            pbar.set_description(f"Analyzing with Gemini: {seq_id}")
        
        try:
            # Process with Gemini
            results = gemini_analyzer.process_segments(video_path, segments, output_path, tmp_dir)
            
            # Clean up tmp files
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            
            with progress_lock:
                pbar.set_description(f"Completed: {seq_id}")
                pbar.update(1)
                
        except Exception as e:
            with progress_lock:
                pbar.set_description(f"Error analyzing {seq_id}: {str(e)}")
                pbar.update(1)
        
        work_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description='LLM-based Ego-Video Activities Summarizer')
    parser.add_argument('--api-key', default=DEFAULT_API_KEY, 
                       help='OpenRouter API key')
    parser.add_argument('--dataset-path', default=DEFAULT_DATASET_PATH,
                       help='Aria Everyday type dataset directory')
    parser.add_argument('--json-path', default=DEFAULT_JSON_PATH,
                       help='Download URLs JSON file path')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of videos to process')
    parser.add_argument('--n-clusters', type=int, default=12,
                       help='Number of clusters for video segmentation (default: 12)')
    parser.add_argument('--n-LLMs', type=int, default=5,
                       help='Number of parallel Gemini API threads (default: 5)')
    
    args = parser.parse_args()
    
    # Load video sequences
    sequences = load_video_sequences(args.json_path)
    
    if args.limit:
        sequences = sequences[:args.limit]
    
    print(f"Processing {len(sequences)} videos with {args.n_LLMs} parallel Gemini threads...")
    
    # Create tmp directory
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create progress tracking
    progress_lock = threading.Lock()
    pbar = tqdm(total=len(sequences), desc="Analyzing videos", unit="video")
    
    # Create work queue for communication between producer and consumers
    work_queue = queue.Queue(maxsize=args.n_LLMs * 5)  # Buffer size
    
    try:
        # Start producer thread (feature extraction and clustering)
        producer_thread = threading.Thread(
            target=producer_worker,
            args=(sequences, args.dataset_path, tmp_dir, args.n_clusters, work_queue, progress_lock, pbar, args.n_LLMs)
        )
        producer_thread.start()
        
        # Start consumer threads (Gemini analysis)
        consumer_threads = []
        for i in range(args.n_LLMs):
            consumer_thread = threading.Thread(
                target=consumer_worker,
                args=(args.api_key, work_queue, progress_lock, pbar)
            )
            consumer_thread.start()
            consumer_threads.append(consumer_thread)
        
        # Wait for producer to finish
        producer_thread.join()
        
        # Wait for all work items to be processed
        work_queue.join()
        
        # Wait for all consumer threads to finish
        for thread in consumer_threads:
            thread.join()
        
        pbar.close()
    
    finally:
        # Clean up tmp directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 