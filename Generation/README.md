# VQA Generation Pipeline

A modular Video Question Answering (VQA) generation system that processes ego-centric videos to generate high-quality multiple choice questions with built-in quality review.

## Overview

This pipeline takes ego-centric videos and their summaries, extracts key frames, detects objects, selects focus objects based on gaze data, and generates natural question-answer pairs that are automatically refined into multiple choice format with quality assurance.

## Architecture

The system consists of the following core modules:

- **utils.py**: Common utilities and helper functions
- **video_loader.py**: Video and summary data loading
- **object_sampler.py**: Integrated keyframe extraction, object detection, CLIP-based clustering, and gaze-based object selection
- **qa_generator.py**: Question-answer generation with tool calling and MCQ refinement
- **qa_reviewer.py**: Automated quality review and feedback system
- **object_detector.py**: Gemini-based object detection
- **frame_extractor.py**: Frame extraction and video clustering utilities
- **main.py**: Main pipeline orchestration with multi-threading

## Installation

1. Install required dependencies:
```bash
pip install torch torchvision opencv-python pillow pandas numpy scikit-learn openai clip-by-openai
```

2. Ensure you have ffmpeg installed for video processing:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Usage

### Quick Start with Example Config

The easiest way to run the pipeline is using the example configuration script:

```bash
cd /home/wang/VQAGenerator/VQAGenerator_fix/Generation
bash example_config.sh
```

Edit `example_config.sh` to configure:
- API key
- Dataset paths
- Question generation parameters
- Number of parallel threads
- Verbose logging

### Full Pipeline

Generate VQAs for multiple videos:

```bash
python main.py \
    --api-key "your-openrouter-api-key" \
    --dataset-path "/path/to/dataset" \
    --json-path "/path/to/dataset.json" \
    --dataset-name "AriaEveryday_Activities" \
    --limit 1:5 \
    --question-factor 4 \
    --sampling-density 60 \
    --n-llms 5 \
    --qa-n-llms 30 \
    --temp-dir "tmp" \
    --verbose
```

### Parameters

**Required:**
- `--api-key`: OpenRouter API key for accessing Gemini models
- `--dataset-path`: Path to the dataset directory containing video folders
- `--json-path`: Path to the JSON file containing video metadata
- `--dataset-name`: Dataset name (used for output filename)

**Optional:**
- `--limit`: Video range to process (e.g., `1:5` for videos 1-4, or `10` for first 10 videos)
- `--question-factor`: Multiplier for number of questions per video (default: 4)
- `--sampling-density`: Frame sampling density - frames per second to sample (default: 60, i.e., ~1 frame/sec)
- `--n-llms`: Number of parallel threads for object detection (default: 5)
- `--qa-n-llms`: Number of parallel threads for QA generation (default: 30)
- `--output-path`: Output JSON file path (default: `<dataset-name>_vqa.json`)
- `--temp-dir`: Temporary directory for processing (default: `tmp`)
- `--verbose`: Enable detailed logging including Agent responses and tool calls

### Test Single Video

Test the pipeline on a single video:

```bash
python main.py test \
    --api-key "your-api-key" \
    --video-path "/path/to/video.mp4" \
    --sequence-id "video_id" \
    --temp-dir "tmp" \
    --verbose
```

### Individual Module Testing

Each module can be tested independently:

#### Object Sampler
```bash
python object_sampler.py \
    --video-path "/path/to/video.mp4" \
    --api-key "your-api-key" \
    --num-samples 5 \
    --temp-dir "tmp"
```

#### Object Detector
```bash
python object_detector.py \
    --api-key "your-api-key" \
    --image-path "/path/to/image.jpg" \
    --visualize
```

#### QA Generator
```bash
python qa_generator.py \
    --api-key "your-api-key" \
    --video-path "/path/to/video.mp4" \
    --timestamp 30.0 \
    --temp-dir "tmp"
```

## Input Requirements

### Dataset Structure
```
dataset_path/
├── sequence_id_1/
│   ├── sequence_id_1.mp4
│   ├── sequence_id_1_summary.json
│   └── sequence_id_1_tracking.csv (optional, for gaze-based selection)
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

### Gaze Tracking CSV (Optional)
```csv
timestamp_ns,x,y
1234567890,512.5,384.2
1234567900,510.1,382.8
...
```

## Output Format

The pipeline generates a JSON file containing:

```json
{
  "generation_date": "2024-01-01,12:00:00",
  "dataset": "AriaEveryday_Activities",
  "summary": {
    "total_videos_processed": 10,
    "total_questions_generated": 120
  },
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
            "raw_qa": "Where did I place the coffee mug? On the desk",
            "CoT": "<Complete Agent reasoning process>",
            "scenario": "User is looking for their coffee mug"
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
          "correct": 0,
          "review_passed": true,
          "review_attempts": 1
        }
      ]
    }
  ]
}
```

## Pipeline Workflow

### 1. Object Sampling & Selection
- Extract keyframes based on `sampling_density` (e.g., 60 = ~1 frame/sec)
- Detect objects in each keyframe using Gemini API
- Cluster similar objects using CLIP features
- Select focus objects based on gaze data (if available) or random sampling
- Generate `question_factor × unique_objects` questions

### 2. QA Generation with Tool Calling
The Agent can use these tools:
- **REFINE_SEGMENT**: Extract multiple frames from a time range for context
- **REFINE_FRAME**: Extract a specific frame at a timestamp
- **REQUEST_REVIEW**: Submit generated MCQ for quality review

### 3. Quality Review System
The Reviewer checks:
- **Fact Verification**: Evidence clarity and additional verification
- **Ambiguity Review**: Viewpoint-independent descriptions and object uniqueness
- **Logic Chain**: Answer is the only logical conclusion from evidence
- **Wording**: Natural phrasing, no timestamps, no video awareness

### 4. Multi-threading Architecture
- Object detection: `n_llms` parallel threads (default: 5)
- QA generation: `qa_n_llms` parallel threads (default: 30)
- Each thread has independent API client to avoid conversation pollution

## Logging

### Simple Mode (default)
```
[12:34:56] Processing video 1/10: loc5_script4_seq6_rec1
[12:34:58] Sampled 150 objects, selected 12 focus objects
[12:35:00] Generating QA for object 'coffee_mug' at 113.1s
[12:35:05] Review passed
[12:35:05] QA generation completed
[12:35:05] Total tokens: 23456
```

### Verbose Mode (`--verbose`)
Includes:
- Agent raw responses
- Messages sent to Agent
- Tool calls with parameters
- Token usage with cache statistics
- Review feedback details

## Advanced Configuration

### Adjusting Question Density
- Higher `question_factor`: More questions per video
- Lower `sampling_density`: More keyframes sampled (e.g., 30 = ~2 frames/sec)

### Optimizing Performance
- Increase `qa_n_llms` for faster parallel QA generation
- Use GPU for CLIP feature extraction (automatic if available)
- Adjust `n_llms` based on API rate limits

### Quality Control
- Review system automatically provides feedback for refinement
- Maximum 3 review attempts per question
- Questions marked with `review_passed` flag in output

## License

See repository root for license information.
