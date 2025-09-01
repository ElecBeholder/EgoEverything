#!/usr/bin/env python3
"""
Object detection using Gemini API
"""
import re
import cv2
import os
from PIL import Image
from typing import List, Dict, Any
from openai import OpenAI
try:
    from .utils import log_message, encode_image_to_base64, safe_get_response_content
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from utils import log_message, encode_image_to_base64, safe_get_response_content


class ObjectDetector:
    """Object detection using Gemini API"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def detect_objects(self, image_path: str) -> str:
        """Detect objects in image using Gemini"""
        log_message("Starting object detection with Gemini")
        
        base64_image = encode_image_to_base64(image_path)
        
        # Detection prompt - keep original prompt unchanged
        detection_prompt = """Please analyze this image and identify all clearly visible objects with their locations. 

IMPORTANT REQUIREMENTS:
1. Focus on objects with distinct, recognizable features
2. Exclude background elements like walls, floors, ceilings, or lighting unless they are specific objects
3. Only include objects that are clearly visible and well-defined
4. Provide accurate bounding box coordinates for each object

For each detected object, provide the results in this exact format:
[ymin,xmin,ymax,xmax]:<object_name>

Where:
- ymin, xmin, ymax, xmax are coordinates in range 0-1000 (normalized coordinates)
- ymin, xmin: top-left corner coordinates
- ymax, xmax: bottom-right corner coordinates
- object_name: clear, specific name of the object

Example format:
[100,200,300,400]:coffee_mug
[50,150,250,350]:laptop
[200,300,400,500]:chair

Please analyze the image carefully and provide all clearly visible objects with their bounding boxes."""

        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": detection_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            }
                        ]
                    }
                ],
                temperature=1,
                max_tokens=65536
            )
            
            response_text = safe_get_response_content(response)
            if response_text:
                log_message("Object detection completed")
                return response_text
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            log_message(f"Object detection failed: {str(e)}")
            raise Exception(f"Object detection failed: {str(e)}")
    
    def parse_detection_results(self, response_text: str, image_path: str, debug: bool = False) -> List[Dict[str, Any]]:
        """Parse Gemini detection results into structured format"""
        objects = []
        
        if debug:
            log_message("=== DEBUG: Raw Gemini Response ===")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            log_message("=== End Raw Response ===")
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
        
        log_message(f"Image dimensions: {img_width} x {img_height}")
        
        # Parse format: [ymin,xmin,ymax,xmax]:<object_name>
        pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]:([^,\n\r]+)'
        matches = re.findall(pattern, response_text)
        
        # Try alternative formats if no matches found
        if len(matches) == 0:
            # Format with spaces
            pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]:\s*([^\n\r]+)'
            matches2 = re.findall(pattern2, response_text)
            if matches2:
                matches = matches2
                log_message(f"Using alternative format (with spaces): found {len(matches)} detections")
            
            # Format without colon
            if len(matches) == 0:
                pattern3 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([^\n\r\[]+)'
                matches3 = re.findall(pattern3, response_text)
                if matches3:
                    matches = matches3
                    log_message(f"Using alternative format (no colon): found {len(matches)} detections")
            
            # Try JSON-like format 
            if len(matches) == 0:
                json_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?"label"\s*:\s*"([^"]+)"'
                matches4 = re.findall(json_pattern, response_text)
                if matches4:
                    matches = matches4
                    log_message(f"Using JSON format: found {len(matches)} detections")
        
        if len(matches) > 0:
            log_message(f"Successfully found {len(matches)} object detections")
        else:
            log_message("No object detections found with any format")
        
        for match in matches:
            try:
                ymin, xmin, ymax, xmax, object_name = match
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                
                # Clean object name - remove JSON artifacts
                object_name = object_name.strip()
                # Remove JSON-like prefixes and suffixes
                object_name = re.sub(r'^[,\s]*"?label"?\s*:\s*"?', '', object_name)
                object_name = re.sub(r'"?\s*[,}]\s*$', '', object_name)
                object_name = object_name.strip(' ",}{:')
                
                # Skip if object name is empty after cleaning
                if not object_name:
                    log_message("Skipping object with empty name")
                    continue
                
                # Validate coordinates
                if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 
                       0 <= ymax <= 1000 and 0 <= xmax <= 1000):
                    log_message(f"Skipping object {object_name}: coordinates out of range")
                    continue
                
                if ymin >= ymax or xmin >= xmax:
                    log_message(f"Skipping object {object_name}: invalid bbox order")
                    continue
                
                # Convert to pixel coordinates for visualization
                pixel_bbox = [
                    int(xmin / 1000 * img_width),  # x0
                    int(ymin / 1000 * img_height), # y0
                    int(xmax / 1000 * img_width),  # x1
                    int(ymax / 1000 * img_height)  # y1
                ]
                
                # Keep normalized coordinates
                normalized_bbox = [xmin, ymin, xmax, ymax]  # [x0,y0,x1,y1] format
                
                objects.append({
                    'name': object_name,
                    'bbox': pixel_bbox,  # [x0,y0,x1,y1] pixel coordinates
                    'normalized_bbox': normalized_bbox,  # [x0,y0,x1,y1] normalized
                    'gemini_bbox': [ymin, xmin, ymax, xmax]  # Original Gemini format
                })
                
                log_message(f"Parsed object: {object_name}")
                
            except Exception as e:
                log_message(f"Failed to parse object: {str(e)}")
                continue
        
        log_message(f"Successfully parsed {len(objects)} objects")
        return objects
    
    def visualize_detections(self, image_path: str, objects: List[Dict[str, Any]], 
                           selected_object: Dict[str, Any] = None, gaze_point: tuple = None) -> str:
        """Visualize detected objects with bounding boxes"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return image_path
            
            img_height, img_width = image.shape[:2]
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
            
            # Draw object bounding boxes
            for i, obj in enumerate(objects):
                bbox = obj['bbox']
                
                # Validate bbox is within image bounds
                if (bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= img_width and 
                    bbox[3] <= img_height and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                    
                    color = colors[i % len(colors)]
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Highlight selected object
                    if selected_object and obj['name'] == selected_object['name']:
                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
                    
                    # Add label
                    label = f"{i+1}. {obj['name']}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    label_y = max(bbox[1] - 5, label_size[1] + 5)
                    
                    cv2.rectangle(image, (bbox[0], label_y - label_size[1] - 5),
                                 (min(bbox[0] + label_size[0], img_width), label_y), color, -1)
                    cv2.putText(image, label, (bbox[0], label_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw gaze point if provided
            if gaze_point:
                gaze_x, gaze_y = gaze_point
                if 0 <= gaze_x <= img_width and 0 <= gaze_y <= img_height:
                    cv2.circle(image, (int(gaze_x), int(gaze_y)), 15, (0, 0, 255), 3)
                    cv2.circle(image, (int(gaze_x), int(gaze_y)), 5, (0, 0, 255), -1)
                    cv2.line(image, (int(gaze_x) - 20, int(gaze_y)), 
                            (int(gaze_x) + 20, int(gaze_y)), (0, 0, 255), 2)
                    cv2.line(image, (int(gaze_x), int(gaze_y) - 20), 
                            (int(gaze_x), int(gaze_y) + 20), (0, 0, 255), 2)
            
            # Save visualization
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_detections.jpg"
            cv2.imwrite(output_path, image)
            
            return output_path
            
        except Exception as e:
            log_message(f"Visualization failed: {str(e)}")
            return image_path


def main():
    """Test object detection functionality"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Test object detector')
    parser.add_argument('--api-key', required=True, help='OpenRouter API key')
    parser.add_argument('--image-path', required=True, help='Image file path')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        log_message(f"Image file not found: {args.image_path}")
        return
    
    # Test object detection
    detector = ObjectDetector(args.api_key)
    
    try:
        # Detect objects
        response_text = detector.detect_objects(args.image_path)
        log_message("Raw detection response:")
        print(response_text)
        
        # Parse results
        objects = detector.parse_detection_results(response_text, args.image_path)
        
        log_message(f"Detected {len(objects)} objects:")
        for i, obj in enumerate(objects):
            log_message(f"{i+1}. {obj['name']} at {obj['bbox']}")
        
        # Create visualization if requested
        if args.visualize and objects:
            viz_path = detector.visualize_detections(args.image_path, objects)
            log_message(f"Visualization saved: {viz_path}")
            
    except Exception as e:
        log_message(f"Detection test failed: {str(e)}")


if __name__ == "__main__":
    main() 