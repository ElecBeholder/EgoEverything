#!/bin/bash
# Example configuration script for VQA Generation Pipeline

# Set your API key
export API_KEY="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"

# Set dataset paths
export DATASET_PATH="/home/wang/AriaEveryday_activaties"
export JSON_PATH="/home/wang/AriaEveryday_activaties/AriaEverydayActivities_download_urls.json"
export DATASET_NAME="AriaEveryday_Activities"

# Run full pipeline (verbose logging)
#python main.py \
#    --api-key "$API_KEY" \
#    --dataset-path "$DATASET_PATH" \
#    --json-path "$JSON_PATH" \
#    --dataset-name "$DATASET_NAME" \
#    --limit 1:2 \
#    --question-factor 4 \
#    --sampling-density 60 \
#    --n-llms 20 \
#    --qa-n-llms 60 \
#    --temp-dir "tmp" \
#    --verbose

# Run test pipeline (simple logging - default)
python main.py \
    --api-key "$API_KEY" \
    --dataset-path "$DATASET_PATH" \
    --json-path "$JSON_PATH" \
    --dataset-name "$DATASET_NAME" \
    --limit 27:29 \
    --question-factor 1 \
    --sampling-density 5 \
    --n-llms 20 \
    --qa-n-llms 1 \
    --temp-dir "tmp" \
    --verbose
