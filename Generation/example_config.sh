#!/bin/bash
# Example configuration script for VQA Generation Pipeline

# Set your API key
export API_KEY="sk-or-v1-your-openrouter-api-key-here"

# Set dataset paths
export DATASET_PATH="/path/to/your/AriaEveryday_activities"
export JSON_PATH="/path/to/your/AriaEverydayActivities_download_urls.json"
export DATASET_NAME="AriaEveryday_Activities"

# Run full pipeline
echo "Running VQA Generation Pipeline..."
python run_vqa_generation.py \
    --api-key "$API_KEY" \
    --dataset-path "$DATASET_PATH" \
    --json-path "$JSON_PATH" \
    --dataset-name "$DATASET_NAME" \
    --limit 5 \
    --questions-per-minute 1.0 \
    --temp-dir "tmp"

echo "Pipeline completed. Check the output JSON file for results."

# Example for testing single video
# echo "Testing single video..."
# python run_vqa_generation.py test \
#     --api-key "$API_KEY" \
#     --video-path "$DATASET_PATH/loc1_script1_seq1_rec1/loc1_script1_seq1_rec1.mp4" \
#     --sequence-id "loc1_script1_seq1_rec1" \
#     --dataset-name "$DATASET_NAME" \
#     --temp-dir "tmp" 