# VQA Generation Pipeline

A modular Video Question Answering (VQA) generation system that processes ego-centric videos to generate multiple choice questions.

## Overview

This pipeline takes ego-centric videos and their summaries, extracts key frames, detects objects, selects focus objects based on gaze data, and generates natural question-answer pairs that are refined into multiple choice format.

## Architecture

The system is divided into the following modules:

- **utils.py**: Common utilities and helper functions
- **video_loader.py**: Video and summary data loading
- **frame_extractor.py**: Key frame extraction and video clustering
- **object_detector.py**: Object detection using Gemini API
- **gaze_processor.py**: Gaze data processing and object selection
- **qa_generator.py**: Question-answer generation with tool calling
- **mcq_refiner.py**: Multiple choice question refinement
- **main.py**: Main pipeline orchestration

## Installation

1. Install required dependencies:
```bash
pip install torch torchvision opencv-python pillow pandas numpy scikit-learn openai
```

2. Ensure you have ffmpeg installed for video processing:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Usage

### Full Pipeline

Generate VQAs for multiple videos with controlled question density:

```bash
cd VQAGenerator_fix/Generation
python run_vqa_generation.py \
    --api-key "your-openrouter-api-key" \
    --dataset-path "/path/to/dataset" \
    --json-path "/path/to/dataset.json" \
    --dataset-name "AriaEveryday_Activities" \
    --limit 10 \
    --questions-per-minute 1.5
```

### Parameters

- `--api-key`: OpenRouter API key for accessing Gemini models
- `--dataset-path`: Path to the dataset directory containing video folders
- `--json-path`: Path to the JSON file containing video metadata
- `--dataset-name`: Dataset name (used for output filename)
- `--limit`: (Optional) Maximum number of videos to process
- `--questions-per-minute`: Number of questions to generate per minute of video
- `--output-path`: (Optional) Output JSON file path (default: <dataset-name>_vqa.json)
- `--temp-dir`: Temporary directory for processing (default: tmp)

### Test Single Video

Test the pipeline on a single video:

```bash
python run_vqa_generation.py test \
    --api-key "your-api-key" \
    --video-path "/path/to/video.mp4" \
    --sequence-id "video_id" \
    --dataset-name "test_dataset" \
    --temp-dir "tmp"
```

### Individual Module Testing

Each module can be tested independently:

#### Video Loader
```bash
python video_loader.py \
    --dataset-path "/path/to/dataset" \
    --sequence-id "loc1_script1_seq1_rec1"
```

#### Frame Extractor
```bash
python frame_extractor.py \
    --video-path "/path/to/video.mp4" \
    --temp-dir "tmp" \
    --test-clustering
```

#### Object Detector
```bash
python object_detector.py \
    --api-key "your-api-key" \
    --image-path "/path/to/image.jpg" \
    --visualize
```

#### Gaze Processor
```bash
python gaze_processor.py \
    --gaze-csv "/path/to/tracking.csv" \
    --timestamp 30.0
```

#### QA Generator
```bash
python qa_generator.py \
    --api-key "your-api-key" \
    --video-path "/path/to/video.mp4" \
    --keyframe-path "/path/to/keyframe.jpg" \
    --timestamp 30.0 \
    --temp-dir "tmp"
```

#### MCQ Refiner
```bash
python mcq_refiner.py \
    --api-key "your-api-key" \
    --question "What color is the cup?" \
    --answer "The cup is blue"
```

## Input Requirements

### Dataset Structure
```
dataset_path/
├── sequence_id_1/
│   ├── sequence_id_1.mp4
│   ├── sequence_id_1_summary.json
│   └── sequence_id_1_tracking.csv (optional)
├── sequence_id_2/
│   ├── sequence_id_2.mp4
│   ├── sequence_id_2_summary.json
│   └── sequence_id_2_tracking.csv (optional)
└── ...
```

### Summary File Format
```json
[
  {
    "start_time": 0.0,
    "end_time": 15.2,
    "description": "Person opens laptop and starts typing"
  },
  {
    "start_time": 15.2,
    "end_time": 30.5,
    "description": "Person picks up coffee mug and takes a sip"
  }
]
```

### Metadata JSON Format
```json
{
  "sequences": {
    "sequence_id_1": {
      "video_main_rgb": {
        "filename": "sequence_id_1.mp4",
        "download_url": "..."
      }
    }
  }
}
```

## Output Format

The pipeline generates a JSON file containing:

```json
{
  "generation_date": "2024-01-01,12:00:00",
  "dataset": "AriaEveryday_Activities",
  "result": [
    {
      "video_name": "loc5_script4_seq6_rec1",
      "QA": [
        {
          "key_frame_timestamp": 113.1,
          "key_object": {
            "name": "coffee_mug",
            "bbox": [100, 150, 200, 250]
          },
          "raw_output": {
            "raw_qa": "Where did I place the coffee mug? You placed the coffee mug on the desk next to the laptop",
            "CoT": "<Complete Gemini output including chain of thought>",
            "scenario": "User is looking for their coffee mug after finishing work"
          },
          "token_usage": 2300,
          "question": "Where did I put my coffee mug?",
          "answer": [
            "On the desk next to the laptop",
            "On the kitchen counter", 
            "On the coffee table",
            "On the bookshelf",
            "On the windowsill"
          ],
          "correct": 0
        }
      ]
    }
  ]
}
```

## Key Features

1. **Modular Design**: Each component can be tested and used independently
2. **Gaze-based Object Selection**: Uses eye tracking data to select relevant objects
3. **Smart Frame Sampling**: Uses clustering to find representative frames
4. **Natural Language Generation**: Creates conversational questions and answers
5. **Automatic MCQ Creation**: Converts QA pairs into challenging multiple choice questions
6. **Configurable Question Density**: Control how many questions to generate per video
7. **Comprehensive Logging**: Simple, clear progress tracking
8. **Temporary File Management**: Automatic cleanup of processing files

## API Models Used

- **Object Detection**: `google/gemini-2.5-pro`
- **QA Generation**: `google/gemini-2.5-pro` with tool calling
- **MCQ Refinement**: `google/gemini-2.5-pro` with structured output

## Error Handling

The system includes robust error handling:
- Videos without summaries are skipped
- Failed object detection attempts are logged and skipped  
- Tool calling failures are handled gracefully
- Temporary files are always cleaned up
- Partial results are saved even if some videos fail

## Performance Notes

- Processing time depends on video length and question density
- GPU acceleration is used for video feature extraction when available
- Multiple questions for the same video use different random keyframes
- Token usage is tracked and reported for cost monitoring 