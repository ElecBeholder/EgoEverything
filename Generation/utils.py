#!/usr/bin/env python3
"""
Utility functions for VQA generation
"""
import os
import base64
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional


def log_message(message: str) -> None:
    """Simple logging with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 format for API"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    if image_path.lower().endswith('.png'):
        mime_type = 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
    elif image_path.lower().endswith('.webp'):
        mime_type = 'image/webp'
    else:
        mime_type = 'image/jpeg'
    
    return f"data:{mime_type};base64,{base64_image}"


def load_video_sequences(json_path: str) -> List[Dict[str, str]]:
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


def save_final_result(result: Dict[str, Any], output_path: str) -> None:
    """Save final VQA result to JSON file"""
    # Only create directory if output_path contains a directory
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if it's not empty
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def cleanup_temp_files(temp_dir: str) -> None:
    """Clean up temporary files"""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def create_temp_dir(base_dir: str, video_id: str) -> str:
    """Create temporary directory for video processing"""
    temp_dir = os.path.join(base_dir, f"tmp_{video_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def safe_get_response_content(response) -> Optional[str]:
    """Safely extract content from API response"""
    try:
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                return message.content
        return None
    except Exception:
        return None


def safe_get_token_usage(response) -> Optional[Dict[str, int]]:
    """Safely extract token usage from API response"""
    try:
        if hasattr(response, 'usage') and response.usage:
            return {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0)
            }
        return None
    except Exception:
        return None


def get_video_duration_minutes(video_path: str) -> float:
    """Get video duration in minutes"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        duration_seconds = frame_count / fps
        return duration_seconds / 60.0
    except Exception:
        return 0.0 