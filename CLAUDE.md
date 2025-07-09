# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a VQA (Visual Question Answering) Generator project that processes AR/VR video recordings to generate question-answer pairs. The system uses Aria glasses recordings and performs computer vision analysis to create training data for VQA models.

## Key Components

### Main Scripts

1. **ai_video_agent.py**: Core VQA generation pipeline
   - Uses VideoLLaMA3 model for video analysis and object detection
   - Integrates with Llama 4 Maverick API for question generation
   - Processes gaze tracking data from Aria recordings
   - Generates contextual questions based on detected objects and user gaze

2. **aria_downloader_processor.py**: Aria dataset downloader and processor
   - Downloads Aria Everyday Activities dataset sequences
   - Processes VRS files and extracts video/gaze data
   - Generates MP4 videos with gaze and hand tracking visualization
   - Uses MediaPipe for hand tracking detection

3. **gaze_hand_visualizer.py**: Visualization tool for tracking data
   - Creates videos with overlaid gaze points and hand tracking
   - Supports batch processing of multiple sequences
   - Customizable colors and visualization parameters

### Evaluation Tools

Located in `Evaluation/` directory:
- **gaze_video_renderer.py**: Advanced video renderer with gaze and hand tracking
- **run_gaze_renderer.py**: Simple runner script for the renderer
- Various output videos with different visualization modes

## Dependencies

This is a Python project with the following key dependencies:
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers for VideoLLaMA3
- **opencv-python**: Computer vision operations
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **projectaria_tools**: Aria glasses data processing
- **mediapipe**: Hand tracking detection
- **openai**: API client for Llama interactions

## Configuration

### Environment Setup
- Primary conda environment: `generatedata`
- CUDA device: `cuda:0` (hardcoded)
- Uses OpenRouter API for Llama 4 Maverick access

### Key Configuration Variables
- `VIDEO_PATH`: Path to input video file
- `OPENROUTER_API_KEY`: API key for Llama model access
- Device configuration for VideoLLaMA3 model

## Common Development Tasks

### Running the Main Pipeline
```bash
conda activate generatedata
python ai_video_agent.py
```

### Processing Aria Dataset
```bash
python aria_downloader_processor.py --limit 10
```

### Generating Visualizations
```bash
python gaze_hand_visualizer.py --data_dir Data
```

### Evaluation Tools
```bash
cd Evaluation
python run_gaze_renderer.py
```

## Architecture Notes

### Video Processing Pipeline
1. **Frame Extraction**: Random or timestamp-based frame sampling
2. **Object Detection**: VideoLLaMA3 model processes frames to detect objects with bounding boxes
3. **Gaze Integration**: Aria gaze data is used to select relevant objects probabilistically
4. **Question Generation**: Llama 4 Maverick generates contextual questions via function calling

### Data Flow
- Input: Aria VRS files with gaze/hand tracking data
- Processing: Computer vision analysis with VideoLLaMA3 and MediaPipe
- Output: VQA pairs with supporting evidence and visualizations

### Key Design Patterns
- **Agent-based architecture**: VideoAnalysisAgent class encapsulates core functionality
- **Multi-modal processing**: Combines video, gaze, and hand tracking data
- **Function calling**: Llama API uses structured function calls for refinement
- **Probabilistic selection**: Gaze-guided object selection using Gaussian distribution

## File Structure Conventions

- Main scripts in root directory
- Evaluation tools in `Evaluation/` subdirectory
- Data stored in `Data/` directory (sequences organized by name)
- Output videos and logs generated in working directory
- Tracking data stored as CSV files alongside video files

## Important Implementation Details

### Coordinate System Handling
- Aria videos require 90-degree rotation for proper orientation
- Gaze coordinates are reprojected to video frame coordinates
- Hand tracking coordinates are adjusted for rotated frames

### Memory Management
- CUDA cache clearing implemented for VideoLLaMA3
- Temporary files are cleaned up after processing
- Model loading is optimized with device mapping

### Error Handling
- Retry logic for API calls and model inference
- Graceful handling of missing gaze/hand data
- Fallback mechanisms for failed object detection

## Testing and Validation

- No formal test framework detected
- Manual validation through visualization outputs
- Token usage tracking for API costs
- Progress logging with timestamps throughout processing