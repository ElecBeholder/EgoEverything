#!/usr/bin/env python3
"""
Main VQA generation pipeline
"""
import os
import argparse
import random
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    from .utils import (
        log_message, load_video_sequences, save_final_result, 
        cleanup_temp_files, create_temp_dir, get_video_duration_minutes
    )
    from .video_loader import VideoLoader
    from .frame_extractor import FrameExtractor
    from .object_detector import ObjectDetector
    from .gaze_processor import ObjectSelector
    from .qa_generator import QAGenerator
    from .mcq_refiner import MCQRefiner
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import (
        log_message, load_video_sequences, save_final_result, 
        cleanup_temp_files, create_temp_dir, get_video_duration_minutes
    )
    from video_loader import VideoLoader
    from frame_extractor import FrameExtractor
    from object_detector import ObjectDetector
    from gaze_processor import ObjectSelector
    from qa_generator import QAGenerator
    from mcq_refiner import MCQRefiner


def extract_scenario_from_qa_response(qa_response: str) -> str:
    """Extract scenario from QA generation response"""
    try:
        # Look for scenario pattern
        scenario_pattern = r"[Ss]cenario:\s*\n?(.*?)(?=\n[Qq]uestion:|$)"
        scenario_match = re.search(scenario_pattern, qa_response, re.DOTALL)
        
        if scenario_match:
            scenario = scenario_match.group(1).strip()
            return scenario
        
        # Alternative pattern
        lines = qa_response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('scenario:'):
                scenario = line.split(':', 1)[1].strip()
                # Check next line for continuation
                if i + 1 < len(lines) and not lines[i + 1].strip().lower().startswith(('question:', 'answer:', 'evidence:')):
                    scenario += " " + lines[i + 1].strip()
                return scenario
        
        return "No scenario found"
    except Exception:
        return "Failed to extract scenario"


class VQAGenerationPipeline:
    """Main VQA generation pipeline"""
    
    def __init__(self, api_key: str, dataset_path: str, temp_base_dir: str = "tmp"):
        self.api_key = api_key
        self.dataset_path = dataset_path
        self.temp_base_dir = temp_base_dir
        
        # Initialize components
        self.video_loader = VideoLoader(dataset_path)
        self.frame_extractor = FrameExtractor()
        self.object_detector = ObjectDetector(api_key)
        self.object_selector = ObjectSelector()
        self.qa_generator = QAGenerator(api_key)
        self.mcq_refiner = MCQRefiner(api_key)
        
        # Create base temp directory
        os.makedirs(temp_base_dir, exist_ok=True)
    
    def generate_single_vqa(self, video_path: str, sequence_id: str) -> Optional[Dict[str, Any]]:
        """Generate a single VQA for a video"""
        temp_dir = create_temp_dir(self.temp_base_dir, f"{sequence_id}_{datetime.now().strftime('%H%M%S')}")
        
        try:
            # Load video summary
            video_summary = self.video_loader.load_video_summary(video_path)
            if not video_summary:
                log_message(f"No summary found for {sequence_id}, skipping")
                return None
            
            # Load gaze data
            gaze_data = self.video_loader.load_gaze_data(video_path)
            
            # Extract random keyframe
            keyframe_path, timestamp = self.frame_extractor.extract_random_frame(video_path, temp_dir)
            log_message(f"Extracted keyframe at {timestamp:.1f}s: {keyframe_path}")
            
            # Detect objects
            log_message("Starting object detection")
            detection_response = self.object_detector.detect_objects(keyframe_path)
            detected_objects = self.object_detector.parse_detection_results(detection_response, keyframe_path)
            
            if not detected_objects:
                log_message("No objects detected, skipping")
                return None
            
            log_message(f"Detected {len(detected_objects)} objects")
            
            # Select key object based on gaze
            selected_object = self.object_selector.select_key_object(
                detected_objects, gaze_data, timestamp, sigma=400
            )
            
            if not selected_object:
                log_message("No object selected, skipping")
                return None
            
            log_message(f"Selected key object: {selected_object['name']}")
            
            # Create visualization
            viz_path = self.object_detector.visualize_detections(
                keyframe_path, detected_objects, selected_object, 
                self.object_selector.gaze_processor.get_gaze_point_at_timestamp(gaze_data, timestamp)
            )
            
            # Generate QA
            log_message("Starting QA generation")
            qa_response = self.qa_generator.generate_qa(
                video_path, keyframe_path, timestamp, video_summary, selected_object, temp_dir
            )
            
            # Refine to MCQ
            log_message("Starting MCQ refinement")
            mcq_result = self.mcq_refiner.process_qa_to_mcq(qa_response)
            
            if not mcq_result['success']:
                log_message(f"MCQ refinement failed: {mcq_result.get('error', 'Unknown error')}")
                return None
            
            # Extract scenario from QA response
            scenario = extract_scenario_from_qa_response(qa_response)
            
            # Create result in template format
            result = {
                'key_frame_timestamp': timestamp,
                'key_object': {
                    'name': selected_object['name'],
                    'bbox': selected_object['bbox']  # Use original bbox (not normalized)
                },
                'raw_output': {
                    'raw_qa': f"{mcq_result['original_question']}? {mcq_result['original_answer']}",
                    'CoT': qa_response,  # Full Gemini output including chain of thought
                    'scenario': scenario
                },
                'token_usage': self.qa_generator.total_tokens['total_tokens'],
                'question': mcq_result['refined_question'],
                'answer': mcq_result['options'],
                'correct': mcq_result['correct_answer_index']
            }
            
            log_message("VQA generation completed successfully")
            return result
            
        except Exception as e:
            log_message(f"VQA generation failed: {str(e)}")
            return None
            
        finally:
            # Clean up temporary files
            cleanup_temp_files(temp_dir)
    
    def calculate_questions_per_video(self, video_path: str, questions_per_minute: float) -> int:
        """Calculate how many questions to generate for a video based on duration"""
        duration_minutes = get_video_duration_minutes(video_path)
        if duration_minutes == 0:
            return 1  # Default to 1 question if duration cannot be determined
        
        num_questions = max(1, int(duration_minutes * questions_per_minute))
        return num_questions
    
    def process_videos(self, sequences: List[Dict[str, str]], questions_per_minute: float, 
                      output_path: str, dataset_name: str) -> None:
        """Process multiple videos and generate VQAs"""
        video_results = []  # List of videos, each with their QA pairs
        total_videos = len(sequences)
        total_questions_generated = 0
        
        for i, sequence in enumerate(sequences, 1):
            sequence_id = sequence['sequence_id']
            video_path = self.video_loader.get_video_path(sequence_id)
            
            log_message(f"Processing video {i}/{total_videos}: {sequence_id}")
            
            # Check if video exists
            if not self.video_loader.validate_video_exists(video_path):
                log_message(f"Video not found: {video_path}")
                continue
            
            # Check if summary exists
            summary = self.video_loader.load_video_summary(video_path)
            if not summary:
                log_message(f"No summary found for {sequence_id}")
                continue
            
            # Calculate number of questions for this video
            num_questions = self.calculate_questions_per_video(video_path, questions_per_minute)
            log_message(f"Generating {num_questions} questions for {sequence_id}")
            
            # Generate multiple VQAs for this video
            video_qa_pairs = []
            for q in range(num_questions):
                log_message(f"Generating question {q+1}/{num_questions} for {sequence_id}")
                
                result = self.generate_single_vqa(video_path, sequence_id)
                
                if result:
                    video_qa_pairs.append(result)
                    total_questions_generated += 1
                    log_message(f"Successfully generated question {q+1} for {sequence_id}")
                else:
                    log_message(f"Failed to generate question {q+1} for {sequence_id}")
            
            # Add video with its QA pairs to results if any questions were generated
            if video_qa_pairs:
                video_results.append({
                    'video_name': sequence_id,
                    'QA': video_qa_pairs
                })
        
        # Create final output in template format
        final_output = {
            'generation_date': datetime.now().strftime('%Y-%m-%d,%H:%M:%S'),
            'dataset': dataset_name,
            'result': video_results
        }
        
        save_final_result(final_output, output_path)
        log_message(f"Generated {total_questions_generated} questions for {len(video_results)} videos, saved to {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VQA Generation Pipeline')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--dataset-path', required=True, help='Dataset directory path')
    parser.add_argument('--json-path', required=True, help='Dataset JSON file path')
    parser.add_argument('--dataset-name', required=True, help='Dataset name for output file')
    parser.add_argument('--limit', type=int, help='Limit number of videos to process')
    parser.add_argument('--questions-per-minute', type=float, default=1.0,
                       help='Number of questions to generate per minute of video (default: 1.0)')
    parser.add_argument('--output-path', default=None,
                       help='Output JSON file path (default: <dataset-name>_vqa.json)')
    parser.add_argument('--temp-dir', default='tmp',
                       help='Temporary directory for processing (default: tmp)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = f"{args.dataset_name}_vqa.json"
    
    # Load video sequences
    log_message(f"Loading video sequences from {args.json_path}")
    sequences = load_video_sequences(args.json_path)
    
    if args.limit:
        sequences = sequences[:args.limit]
    
    log_message(f"Processing {len(sequences)} videos with {args.questions_per_minute} questions per minute")
    log_message(f"Dataset: {args.dataset_name}")
    log_message(f"Output will be saved to: {args.output_path}")
    
    # Create pipeline
    pipeline = VQAGenerationPipeline(
        api_key=args.api_key,
        dataset_path=args.dataset_path,
        temp_base_dir=args.temp_dir
    )
    
    try:
        # Process videos
        pipeline.process_videos(sequences, args.questions_per_minute, args.output_path, args.dataset_name)
        log_message("VQA generation pipeline completed successfully")
        
    except Exception as e:
        log_message(f"Pipeline failed: {str(e)}")
        raise
    
    finally:
        # Clean up base temp directory
        cleanup_temp_files(args.temp_dir)


# For testing individual components
def test_single_video():
    """Test pipeline on a single video"""
    parser = argparse.ArgumentParser(description='Test VQA Generation on Single Video')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--video-path', required=True, help='Single video file path')
    parser.add_argument('--sequence-id', required=True, help='Video sequence ID')
    parser.add_argument('--dataset-name', default='test_dataset', help='Dataset name for output')
    parser.add_argument('--temp-dir', default='tmp', help='Temporary directory')
    
    args = parser.parse_args()
    
    # Extract dataset path from video path
    dataset_path = os.path.dirname(os.path.dirname(args.video_path))
    
    # Create pipeline
    pipeline = VQAGenerationPipeline(
        api_key=args.api_key,
        dataset_path=dataset_path,
        temp_base_dir=args.temp_dir
    )
    
    try:
        log_message(f"Testing single video: {args.video_path}")
        result = pipeline.generate_single_vqa(args.video_path, args.sequence_id)
        
        if result:
            # Format result according to template
            test_output = {
                'generation_date': datetime.now().strftime('%Y-%m-%d,%H:%M:%S'),
                'dataset': args.dataset_name,
                'result': [
                    {
                        'video_name': args.sequence_id,
                        'QA': [result]
                    }
                ]
            }
            
            output_path = f"{args.dataset_name}_test_{args.sequence_id}_vqa.json"
            save_final_result(test_output, output_path)
            log_message(f"Test completed successfully, result saved to {output_path}")
        else:
            log_message("Test failed")
            
    except Exception as e:
        log_message(f"Test failed: {str(e)}")
        raise
    
    finally:
        cleanup_temp_files(args.temp_dir)


if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.argv.pop(1)  # Remove 'test' argument
        test_single_video()
    else:
        main() 