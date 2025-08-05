#!/usr/bin/env python3
"""
VQA Generation Pipeline Runner
Direct execution script for the VQA generation system
"""

import sys
import os

# Add current directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import main functionality
from main import main, test_single_video

if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.argv.pop(1)  # Remove 'test' argument
        test_single_video()
    else:
        main() 