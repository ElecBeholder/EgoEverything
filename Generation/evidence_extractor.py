#!/usr/bin/env python3
"""
Evidence Timestamp Extractor using lightweight LLM with structured output
"""
import json
from typing import Dict, Any
from openai import OpenAI
try:
    from .utils import log_message, safe_get_response_content, safe_get_token_usage
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import log_message, safe_get_response_content, safe_get_token_usage


class EvidenceTimestampExtractor:
    """Extracts timestamps from evidence text using lightweight LLM"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    def extract_timestamps(self, evidence_text: str) -> Dict[str, Any]:
        """Extract timestamps from evidence text with structured output"""
        log_message("Extracting timestamps from evidence text")
        
        system_prompt = """You must extract ALL timestamps from the evidence text. Do not miss any timestamps.

Rules:
1. FRAMES: Extract every single timestamp mentioned
2. SEGMENTS: Extract time ranges  
3. Return only numeric values in seconds
4. MUST respond in valid JSON format only

Example input: "At 0.0 seconds, object on table. At 2.0 seconds, still there. From 5s to 8s, moving."
Example output: {"frames": [0.0, 2.0], "segments": [{"start": 5.0, "end": 8.0}]}

Response format: {"frames": [list of numbers], "segments": [{"start": number, "end": number}]}"""
        
        user_message = f"Evidence text:\n{evidence_text}\n\nExtract all timestamps."
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=512
            )
        except Exception:
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=512
            )
        
        # Track token usage
        token_usage = safe_get_token_usage(response)
        if token_usage:
            for key in self.total_tokens:
                if key in token_usage:
                    self.total_tokens[key] += token_usage[key]
        
        # Parse response
        response_content = safe_get_response_content(response)
        timestamps = json.loads(response_content)
        
        # Ensure structure and validate data
        timestamps.setdefault("frames", [])
        timestamps.setdefault("segments", [])
        
        timestamps["frames"] = [
            float(f) for f in timestamps["frames"] 
            if isinstance(f, (int, float))
        ]
        timestamps["segments"] = [
            {"start": float(s["start"]), "end": float(s["end"])}
            for s in timestamps["segments"]
            if isinstance(s, dict) and "start" in s and "end" in s
        ]
        
        log_message(f"Extracted {len(timestamps['frames'])} frames, {len(timestamps['segments'])} segments")
        return timestamps
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage"""
        return self.total_tokens


def main():
    """Test evidence timestamp extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test evidence timestamp extractor')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--evidence', required=True, help='Evidence text to extract timestamps from')
    
    args = parser.parse_args()
    
    # Test extraction
    extractor = EvidenceTimestampExtractor(args.api_key)
    
    timestamps = extractor.extract_timestamps(args.evidence)
    
    print("\nExtracted Timestamps:")
    print(json.dumps(timestamps, indent=2))
    print(f"\nToken usage: {extractor.get_token_usage()}")


if __name__ == "__main__":
    main()
