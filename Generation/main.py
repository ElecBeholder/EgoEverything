#!/usr/bin/env python3
"""
Main VQA generation pipeline
"""
import os
import argparse
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
try:
    from .utils import (
        log_message, log_simple, set_log_verbose, load_video_sequences, save_final_result, 
        cleanup_temp_files, create_temp_dir, get_video_duration_minutes
    )
    from .video_loader import VideoLoader
    from .frame_extractor import FrameExtractor, SegmentFeatureExtractor
    from .object_detector import ObjectDetector
    from .gaze_processor import ObjectSelector
    from .qa_generator import QAGenerator
    from .qa_reviewer import QAReviewerRefiner
    from .evidence_extractor import EvidenceTimestampExtractor
    from .object_sampler import ObjectSampler
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import (
        log_message, log_simple, set_log_verbose, load_video_sequences, save_final_result, 
        cleanup_temp_files, create_temp_dir, get_video_duration_minutes
    )
    from video_loader import VideoLoader
    from frame_extractor import FrameExtractor, SegmentFeatureExtractor
    from object_detector import ObjectDetector
    from gaze_processor import ObjectSelector
    from qa_generator import QAGenerator
    from qa_reviewer import QAReviewerRefiner
    from evidence_extractor import EvidenceTimestampExtractor
    from object_sampler import ObjectSampler



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
        self.qa_reviewer = QAReviewerRefiner(api_key)
        self.evidence_extractor = EvidenceTimestampExtractor(api_key)
        self.object_sampler = ObjectSampler(api_key)
        self.sampling_density = 60.0
        self.qa_n_llms = 30
        
        # Create base temp directory
        os.makedirs(temp_base_dir, exist_ok=True)
    
    def generate_single_vqa(self, video_path: str, sequence_id: str,
                            preselected: Optional[Dict[str, Any]] = None,
                            gaze_data: Optional[Any] = None,
                            qa_generator: Optional[QAGenerator] = None,
                            qa_reviewer: Optional[QAReviewerRefiner] = None,
                            video_summary_preloaded: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a single VQA for a video"""
        # Create a per-question unique temp directory to avoid collisions across threads
        import threading
        unique_suffix = f"{datetime.now().strftime('%H%M%S_%f')}_{threading.get_ident()}"
        temp_dir = create_temp_dir(self.temp_base_dir, f"{sequence_id}_{unique_suffix}")
        
        try:
            # Load video summary
            video_summary = video_summary_preloaded if video_summary_preloaded is not None else self.video_loader.load_video_summary(video_path)
            if not video_summary:
                log_simple(f"No summary found for {sequence_id}, skipping")
                return None
            
            # Load gaze if not provided
            if gaze_data is None:
                gaze_data = self.video_loader.load_gaze_data(video_path)

            # If preselected provided, use it; otherwise, sample one via ObjectSampler
            if preselected is not None:
                keyframe_path = preselected['keyframe_path']
                timestamp = preselected['timestamp']
                selected_object = preselected['selected_object']
                log_message(f"Using preselected keyframe {timestamp:.1f}s and object {selected_object['name']}")
            else:
                log_simple("Sampling key frame and key object via ObjectSampler")
                sampler_results = self.object_sampler.process_video(
                    video_path=video_path,
                    questions_per_minute=1.0,
                    temp_dir=temp_dir,
                    gaze_data=gaze_data,
                    sampling_density=self.sampling_density,
                    num_key_object_samples=1
                )
                if not sampler_results or not sampler_results.get('key_objects'):
                    log_simple("ObjectSampler returned no key objects, skipping")
                    return None
                sampled = sampler_results['key_objects'][0]
                timestamp = sampled['keyframe']['timestamp']
                keyframe_path = sampled['keyframe']['frame_path']
                # Ensure bbox passed to QA is STRICTLY Gemini-format; fallback by converting
                instance = sampled['instance']
                if 'gemini_bbox' in instance and instance['gemini_bbox'] is not None:
                    gemini_bbox = instance['gemini_bbox']
                elif 'normalized_bbox' in instance and instance['normalized_bbox'] is not None:
                    x0, y0, x1, y1 = instance['normalized_bbox']
                    gemini_bbox = [y0, x0, y1, x1]
                else:
                    x0, y0, x1, y1 = instance['bbox']
                    img_w = instance.get('image_width', 1) or 1
                    img_h = instance.get('image_height', 1) or 1
                    gemini_bbox = [
                        int(y0 / img_h * 1000),
                        int(x0 / img_w * 1000),
                        int(y1 / img_h * 1000),
                        int(x1 / img_w * 1000)
                    ]
                selected_object = {
                    'name': instance['object_name'],
                    'gemini_bbox': gemini_bbox,            # [ymin,xmin,ymax,xmax] in 0-1000
                    'bbox': instance.get('bbox')            # pixel coords [x0,y0,x1,y1] on keyframe
                }
                log_simple(f"Sampled keyframe {timestamp:.1f}s and object {selected_object['name']}")
            
            # Generate QA
            log_simple("Starting QA generation")
            local_qa_gen = qa_generator if qa_generator is not None else self.qa_generator
            qa_result = local_qa_gen.generate_qa(
                video_path, keyframe_path, timestamp, video_summary, selected_object, temp_dir
            )
            
            if not qa_result['success']:
                log_simple("QA generation failed")
                return None
            
            # Extract evidence timestamps
            log_simple("Extracting evidence timestamps")
            evidence_timestamps = self.evidence_extractor.extract_timestamps(qa_result['evidence'])
            
            # Extract evidence images before review
            log_simple("Extracting evidence images for review")
            evidence_images = []
            segment_extractor = SegmentFeatureExtractor()
            
            # Extract frames from single frame timestamps
            if evidence_timestamps.get("frames"):
                for frame_ts in evidence_timestamps["frames"]:
                    try:
                        frame_path = self.frame_extractor.extract_frame_at_timestamp(
                            video_path, frame_ts, temp_dir
                        )
                        evidence_images.append({
                            "type": "frame",
                            "timestamp": frame_ts,
                            "path": frame_path
                        })
                        log_message(f"Extracted evidence frame at {frame_ts}s")
                    except Exception as e:
                        log_message(f"Failed to extract frame at {frame_ts}s: {str(e)}")
            
            # Extract representative frames from segments
            if evidence_timestamps.get("segments"):
                for segment in evidence_timestamps["segments"]:
                    try:
                        # Get representative frames from segment
                        representative_frames = segment_extractor.get_representative_frames(
                            video_path, segment['start'], segment['end'], 
                            n_clusters=5, temp_dir=temp_dir
                        )
                        
                        for frame_data in representative_frames[:3]:  # Take up to 3 frames per segment
                            frame_path = self.frame_extractor.extract_frame_at_timestamp(
                                video_path, frame_data['timestamp'], temp_dir
                            )
                            evidence_images.append({
                                "type": "segment",
                                "timestamp": frame_data['timestamp'],
                                "segment_range": f"{segment['start']}s-{segment['end']}s",
                                "path": frame_path
                            })
                        log_message(f"Extracted {len(representative_frames[:3])} frames from segment {segment['start']}s-{segment['end']}s")
                    except Exception as e:
                        log_message(f"Failed to extract segment {segment['start']}-{segment['end']}: {str(e)}")
            
            log_message(f"Total evidence images extracted: {len(evidence_images)}")
            
            # Review and refine QA to MCQ with evidence images
            log_simple("Starting QA review and MCQ refinement with evidence images")
            local_reviewer = qa_reviewer if qa_reviewer is not None else self.qa_reviewer
            review_result = local_reviewer.review_and_refine(
                qa_result['question'], qa_result['answer'], evidence_images,
                video_path, video_summary, temp_dir
            )
            
            # Log detailed review and refinement results
            if review_result['success']:
                parsed_result = review_result['parsed_result']
                log_simple(f"QA Review and refinement completed")
                log_message("=== REVIEW & REFINEMENT RESULTS ===")
                log_message(f"Review Checklist:")
                for check_name, check_result in parsed_result['review_checklist'].items():
                    log_message(f"  - {check_name}: {check_result}")
                log_message(f"Refinement Rationale: {parsed_result['refinement_rationale']}")
                log_message(f"Refined Question: {parsed_result['refined_question']}")
                log_message(f"Options: {parsed_result['refined_options']}")
                log_message(f"Correct Answer: {parsed_result['correct_answer']}")
                log_message(f"Distinction Notes: {parsed_result['distinction_notes']}")
                log_message("=== END REVIEW & REFINEMENT ===")
            else:
                log_simple("QA review and refinement failed")
                return None
            
            # Create result with refined MCQ from review
            # Convert correct answer letter to index
            correct_index = None
            if parsed_result['correct_answer'] in 'ABCDE':
                correct_index = ord(parsed_result['correct_answer']) - ord('A')
            
            result = {
                'key_frame_timestamp': timestamp,
                'key_object': {
                    'name': selected_object['name'],
                    'bbox': selected_object['bbox']  # Use original bbox (not normalized)
                },
                'raw_output': {
                    'raw_qa': f"{qa_result['question']}? {qa_result['answer']}",
                    'CoT': qa_result['raw_response'],  # Full Gemini output including chain of thought
                    'scenario': qa_result['scenario'],
                    'original_question': qa_result['question'],
                    'original_answer': qa_result['answer']
                },
                'token_usage': (qa_generator.total_tokens['total_tokens'] if qa_generator is not None else self.qa_generator.total_tokens['total_tokens']),
                'review_result': {
                    'review_checklist': parsed_result['review_checklist'],
                    'refinement_rationale': parsed_result['refinement_rationale'],
                    'distinction_notes': parsed_result['distinction_notes']
                },
                # Refined MCQ fields
                'question': parsed_result['refined_question'] if parsed_result['refined_question'] else qa_result['question'],
                'answer': parsed_result['refined_options'] if parsed_result['refined_options'] else [qa_result['answer'], "Option B", "Option C", "Option D", "Option E"],
                'correct': correct_index if correct_index is not None else 0
            }
            
            log_simple("VQA generation completed successfully")
            return result
            
        except Exception as e:
            log_simple(f"VQA generation failed: {str(e)}")
            return None
            
        finally:
            # Clean up temporary files
            cleanup_temp_files(temp_dir)
    
    def calculate_questions_per_video(self, video_path: str, question_factor: int, gaze_data) -> Tuple[int, Dict[str, Any]]:
        """Determine QA count = question_factor * unique_object_ids via ObjectSampler pre-pass"""
        temp_dir = create_temp_dir(self.temp_base_dir, f"prepass_{datetime.now().strftime('%H%M%S')}")
        try:
            sampler_results = self.object_sampler.process_video(
                video_path=video_path,
                questions_per_minute=1.0,
                temp_dir=temp_dir,
                gaze_data=gaze_data,
                sampling_density=1.0,
                num_key_object_samples=0
            )
            unique_ids = sampler_results['summary']['unique_objects'] if sampler_results else 1
            return max(1, question_factor * unique_ids), sampler_results
        finally:
            cleanup_temp_files(temp_dir)
    
    def process_videos(self, sequences: List[Dict[str, str]], question_factor: int, 
                      output_path: str, dataset_name: str) -> None:
        """Process multiple videos and generate VQAs with incremental saving"""
        total_videos = len(sequences)
        total_questions_generated = 0
        
        for i, sequence in enumerate(sequences, 1):
            sequence_id = sequence['sequence_id']
            video_path = self.video_loader.get_video_path(sequence_id)
            
            log_simple(f"Processing video {i}/{total_videos}: {sequence_id}")
            
            # Check if video exists
            if not self.video_loader.validate_video_exists(video_path):
                log_simple(f"Video not found: {video_path}")
                continue
            
            # Check if summary exists
            summary = self.video_loader.load_video_summary(video_path)
            if not summary:
                log_simple(f"No summary found for {sequence_id}")
                continue
            
            # Load gaze data
            gaze_data = self.video_loader.load_gaze_data(video_path)
            # Run sampler once to get unique objects and sample list sized by question_factor
            log_simple("Running ObjectSampler pre-pass to sample key objects for this video")
            temp_dir_pre = create_temp_dir(self.temp_base_dir, f"{sequence_id}_sampler_{datetime.now().strftime('%H%M%S')}")
            sampler_results = self.object_sampler.process_video(
                video_path=video_path,
                questions_per_minute=1.0,
                temp_dir=temp_dir_pre,
                gaze_data=gaze_data,
                sampling_density=self.sampling_density,
                num_key_object_samples=None,
                question_factor=question_factor
            )
            key_objects = sampler_results.get('key_objects', []) if sampler_results else []
            num_questions = len(key_objects)
            unique_ids = sampler_results['summary']['unique_objects'] if sampler_results else 0
            log_simple(f"Generating {num_questions} questions for {sequence_id} (factor {question_factor} * unique_objects={unique_ids})")
            
            # Generate multiple VQAs for this video using multiple Gemini threads
            import threading, queue
            work_queue = queue.Queue()
            result_queue = queue.Queue()
            progress_lock = threading.Lock()
            
            # Prepare work items (preselected samples)
            for q in range(num_questions):
                sample = key_objects[q]
                instance = sample['instance']
                # Build Gemini-format bbox strictly for batch generation
                if 'gemini_bbox' in instance and instance['gemini_bbox'] is not None:
                    gemini_bbox = instance['gemini_bbox']
                elif 'normalized_bbox' in instance and instance['normalized_bbox'] is not None:
                    x0, y0, x1, y1 = instance['normalized_bbox']
                    gemini_bbox = [y0, x0, y1, x1]
                else:
                    x0, y0, x1, y1 = instance['bbox']
                    img_w = instance.get('image_width', 1) or 1
                    img_h = instance.get('image_height', 1) or 1
                    gemini_bbox = [
                        int(y0 / img_h * 1000),
                        int(x0 / img_w * 1000),
                        int(y1 / img_h * 1000),
                        int(x1 / img_w * 1000)
                    ]
                preselected = {
                    'keyframe_path': sample['keyframe']['frame_path'],
                    'timestamp': sample['keyframe']['timestamp'],
                    'selected_object': {
                        'name': instance['object_name'],
                        'gemini_bbox': gemini_bbox,       # [ymin,xmin,ymax,xmax] 0-1000
                        'bbox': instance.get('bbox')       # pixel [x0,y0,x1,y1]
                    }
                }
                work_queue.put(preselected)

            # Sentinel for consumers
            for _ in range(self.qa_n_llms):
                work_queue.put(None)

            def qa_consumer_thread(thread_id: int):
                # Each thread should have its own generator/reviewer (separate conversations)
                local_qa = QAGenerator(self.api_key)
                local_reviewer = QAReviewerRefiner(self.api_key)
                local_results = []
                while True:
                    item = work_queue.get()
                    if item is None:
                        work_queue.task_done()
                        break
                    try:
                        res = self.generate_single_vqa(
                            video_path=video_path,
                            sequence_id=sequence_id,
                            preselected=item,
                            gaze_data=gaze_data,
                            qa_generator=local_qa,
                            qa_reviewer=local_reviewer,
                            video_summary_preloaded=summary
                        )
                        result_queue.put(res)
                    except Exception:
                        result_queue.put(None)
                    finally:
                        work_queue.task_done()

            # Start consumers
            consumers = []
            for i_th in range(self.qa_n_llms):
                t = threading.Thread(target=qa_consumer_thread, args=(i_th,))
                t.start()
                consumers.append(t)

            # Wait for all to complete
            work_queue.join()
            for t in consumers:
                t.join()

            # Collect results (preserve order roughly by taking non-None)
            video_qa_pairs = []
            while not result_queue.empty():
                r = result_queue.get()
                if r:
                    video_qa_pairs.append(r)
                    total_questions_generated += 1

            # Incremental write to JSON per video
            if video_qa_pairs:
                self._append_video_results_incremental(
                    output_path=output_path,
                    dataset_name=dataset_name,
                    video_name=sequence_id,
                    qa_pairs=video_qa_pairs
                )

            # Now that all questions for this video are done, clean up the sampler temp keyframes
            cleanup_temp_files(temp_dir_pre)
        
        # Final log summary from file
        try:
            data = self._load_existing_output(output_path)
            if data:
                summary = data.get('summary', {})
                log_simple(f"Generated {summary.get('total_questions_generated', 0)} questions for {summary.get('total_videos_processed', 0)} videos, saved to {output_path}")
        except Exception:
            pass

    def _load_existing_output(self, output_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(output_path):
            return None
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log_message(f"Failed to load existing output: {e}")
            return None

    def _append_video_results_incremental(self, output_path: str, dataset_name: str,
                                          video_name: str, qa_pairs: List[Dict[str, Any]]) -> None:
        data = self._load_existing_output(output_path)
        if not data or 'result' not in data:
            data = {
                'generation_date': datetime.now().strftime('%Y-%m-%d,%H:%M:%S'),
                'dataset': dataset_name,
                'summary': {
                    'total_videos_processed': 0,
                    'total_questions_generated': 0
                },
                'result': []
            }

        data['result'].append({'video_name': video_name, 'QA': qa_pairs})
        data['summary']['total_videos_processed'] = len(data['result'])
        try:
            prev = int(data['summary'].get('total_questions_generated', 0))
        except Exception:
            prev = 0
        data['summary']['total_questions_generated'] = prev + len(qa_pairs)

        save_final_result(data, output_path)
        log_simple(f"Completed video {video_name}: {len(qa_pairs)} QA pairs -> {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VQA Generation Pipeline')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--dataset-path', required=True, help='Dataset directory path')
    parser.add_argument('--json-path', required=True, help='Dataset JSON file path')
    parser.add_argument('--dataset-name', required=True, help='Dataset name for output file')
    parser.add_argument('--limit', help='Limit videos: either N (first N) or RANGE like start-end (1-based, inclusive)')
    parser.add_argument('--question-factor', type=int, default=4,
                       help='Question factor: generate <factor * unique_object_ids> QA per video (default: 4)')
    parser.add_argument('--sampling-density', type=float, default=60.0,
                       help='Sampler density for initial key frame extraction (default: 60, ~one per second)')
    parser.add_argument('--output-path', default=None,
                       help='Output JSON file path (default: <dataset-name>_vqa.json)')
    parser.add_argument('--temp-dir', default='tmp',
                       help='Temporary directory for processing (default: tmp)')
    parser.add_argument('--n-llms', type=int, default=5,
                        help='Number of parallel Gemini API threads for object detection (default: 5)')
    parser.add_argument('--qa-n-llms', type=int, default=30,
                        help='Number of parallel Gemini API threads for QA generation (default: 30)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable detailed logging output (default: simple logging)')
    
    args = parser.parse_args()
    
    # Set global log verbosity based on argument
    set_log_verbose(args.verbose)
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = f"{args.dataset_name}_vqa.json"
    
    # Load video sequences
    log_simple(f"Loading video sequences from {args.json_path}")
    sequences = load_video_sequences(args.json_path)
    
    # Apply limit: support N or start-end (1-based, inclusive)
    total_videos_all = len(sequences)
    if args.limit:
        s = str(args.limit)
        try:
            if '-' in s or ':' in s:
                delim = '-' if '-' in s else ':'
                parts = [p.strip() for p in s.split(delim) if p.strip()]
                if len(parts) == 2:
                    start_idx = max(1, int(parts[0]))
                    end_idx = min(total_videos_all, int(parts[1]))
                    if start_idx <= end_idx:
                        sequences = sequences[start_idx - 1:end_idx]
                        log_simple(f"Limiting videos to range {start_idx}-{end_idx} (of {total_videos_all})")
                    else:
                        log_simple(f"Invalid limit range: {s}. start > end. No videos will be processed.")
                        sequences = []
                else:
                    log_simple(f"Invalid limit format: {s}. Expected 'start-end'. Ignoring limit.")
            else:
                n = int(s)
                if n >= 0:
                    sequences = sequences[:n]
                    log_simple(f"Limiting to first {n} videos (of {total_videos_all})")
        except Exception as e:
            log_simple(f"Failed to parse --limit '{s}': {e}. Ignoring limit.")
    
    log_simple(f"Processing {len(sequences)} videos with question factor {args.question_factor} and sampling density {args.sampling_density}")
    log_simple(f"Dataset: {args.dataset_name}")
    log_simple(f"Output will be saved to: {args.output_path}")
    
    # Create pipeline
    pipeline = VQAGenerationPipeline(
        api_key=args.api_key,
        dataset_path=args.dataset_path,
        temp_base_dir=args.temp_dir
    )
    # Propagate sampling density to pipeline
    pipeline.sampling_density = args.sampling_density
    # Propagate n_llms default to object sampler
    pipeline.object_sampler.n_llms = args.n_llms
    pipeline.qa_n_llms = args.qa_n_llms
    
    try:
        # Process videos
        pipeline.process_videos(sequences, args.question_factor, args.output_path, args.dataset_name)
        log_simple("VQA generation pipeline completed successfully")
        
    except Exception as e:
        log_simple(f"Pipeline failed: {str(e)}")
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
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable detailed logging output (default: simple logging)')
    
    args = parser.parse_args()
    
    # Set global log verbosity based on argument
    set_log_verbose(args.verbose)
    
    # Extract dataset path from video path
    dataset_path = os.path.dirname(os.path.dirname(args.video_path))
    
    # Create pipeline
    pipeline = VQAGenerationPipeline(
        api_key=args.api_key,
        dataset_path=dataset_path,
        temp_base_dir=args.temp_dir
    )
    
    try:
        log_simple(f"Testing single video: {args.video_path}")
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
            log_simple(f"Test completed successfully, result saved to {output_path}")
        else:
            log_simple("Test failed")
            
    except Exception as e:
        log_simple(f"Test failed: {str(e)}")
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