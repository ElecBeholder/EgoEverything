#!/usr/bin/env python3
"""
AI Video Analysis Agent - 优化版本
"""

import torch
import cv2
import random
import numpy as np
import os
import time
import json
import re
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from openai import OpenAI

# 配置
device = "cuda:0"
#VIDEO_PATH = "Data/loc5_script4_seq6_rec1/loc5_script4_seq6_rec1.mp4"
VIDEO_PATH = "Data/loc3_script5_seq6_rec1/loc3_script5_seq6_rec1.mp4"
#VIDEO_PATH = "Data/loc4_script1_seq1_rec1/loc4_script1_seq1_rec1.mp4"
OPENROUTER_API_KEY = "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"

def log_timestamp(message):
    """统一的时间戳日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class VideoAnalysisAgent:
    def __init__(self):
        self.model = None
        self.processor = None
        self.patch_description = ""
        self.full_video_description = ""
        self.patch_timestamp = 0.0
        self.detected_objects = []
        self.selected_object = None
        self.gaze_data = None
        self.current_gaze_point = None
        self.gemini_log = []
        
    def initialize_videollama3(self):
        if self.model is None:
            log_timestamp("正在加载VideoLLaMA3模型...")
            model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, device_map={"": device},
                torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            log_timestamp("VideoLLaMA3模型加载完成")
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_gaze_data(self, video_path):
        try:
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
            
            if os.path.exists(csv_path):
                self.gaze_data = pd.read_csv(csv_path)
                return True
            return False
        except:
            return False
    
    def extract_frame(self, video_path, timestamp=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if timestamp is None:
            frame_num = random.randint(0, total_frames - 1)
            self.patch_timestamp = frame_num / fps
            output_path = f"keyframe_{self.patch_timestamp:.1f}s.jpg"
        else:
            frame_num = int(timestamp * fps)
            frame_num = max(0, min(frame_num, total_frames - 1))
            output_path = f"temp_frame_{timestamp:.1f}s.jpg"
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            raise Exception(f"无法读取帧")
        
        cv2.imwrite(output_path, frame)
        cap.release()
        return output_path
    
    def get_gaze_point(self, target_timestamp):
        if self.gaze_data is None:
            return None
        
        try:
            video_start_time_ns = self.gaze_data['timestamp_ns'].iloc[0]
            target_timestamp_ns = video_start_time_ns + int(target_timestamp * 1e9)
            time_diff = np.abs(self.gaze_data['timestamp_ns'] - target_timestamp_ns)
            closest_idx = time_diff.idxmin()
            
            closest_row = self.gaze_data.iloc[closest_idx]
            gaze_x, gaze_y = closest_row['gaze_x'], closest_row['gaze_y']
            
            if pd.isna(gaze_x) or pd.isna(gaze_y):
                return None
            
            return (float(gaze_x), float(gaze_y))
        except:
            return None
    
    def parse_objects(self, response, image_path):
        objects = []
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
        
        sections = response.split("Object:")
        for section in sections[1:]:
            try:
                lines = section.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                object_name = lines[0].strip()
                location_line = None
                description_parts = []
                position_parts = []
                current_section = None
                
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith("Location:"):
                        location_line = line
                    elif line.startswith("Description:"):
                        current_section = "description"
                        description_parts.append(line.replace("Description:", "").strip())
                    elif line.startswith("Position:"):
                        current_section = "position"
                        position_parts.append(line.replace("Position:", "").strip())
                    elif line.startswith("Context:"):
                        current_section = "position"
                        position_parts.append(line.replace("Context:", "").strip())
                    elif current_section and line:
                        if current_section == "description":
                            description_parts.append(line)
                        else:
                            position_parts.append(line)
                
                bbox = None
                if location_line:
                    bbox_match = re.search(r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]', location_line)
                    if not bbox_match:
                        bbox_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', location_line)
                    
                    if bbox_match:
                        x0, y0, x1, y1 = map(int, bbox_match.groups())
                        bbox = [
                            int(x0 / 1000 * img_width), int(y0 / 1000 * img_height),
                            int(x1 / 1000 * img_width), int(y1 / 1000 * img_height)
                        ]
                
                if bbox and object_name:
                    objects.append({
                        'name': object_name,
                        'bbox': bbox,
                        'description': ' '.join(description_parts),
                        'position': ' '.join(position_parts)
                    })
            except:
                continue
        
        return objects
    
    def select_object_by_gaze(self, objects, gaze_point, sigma=400):
        if not objects:
            return None
        
        if gaze_point is None:
            return random.choice(objects)
        
        gaze_x, gaze_y = gaze_point
        probabilities = []
        
        for obj in objects:
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            dist_sq = (center_x - gaze_x)**2 + (center_y - gaze_y)**2
            prob = np.exp(-dist_sq / (2 * sigma**2))
            probabilities.append(prob)
        
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(objects)] * len(objects)
        
        selected_idx = np.random.choice(len(objects), p=probabilities)
        return objects[selected_idx]
    
    def visualize_keyframe(self, image_path, objects, gaze_point=None):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return
            
            img_height, img_width = image.shape[:2]
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
            
            for i, obj in enumerate(objects):
                bbox = obj['bbox']
                if (bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] <= img_width and 
                    bbox[3] <= img_height and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                    
                    color = colors[i % len(colors)]
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    if hasattr(self, 'selected_object') and self.selected_object and obj['name'] == self.selected_object['name']:
                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
                    
                    label = f"{i+1}. {obj['name']}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    label_y = max(bbox[1] - 5, label_size[1] + 5)
                    
                    cv2.rectangle(image, (bbox[0], label_y - label_size[1] - 5),
                                 (min(bbox[0] + label_size[0], img_width), label_y), color, -1)
                    cv2.putText(image, label, (bbox[0], label_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if gaze_point:
                gaze_x, gaze_y = gaze_point
                if 0 <= gaze_x <= img_width and 0 <= gaze_y <= img_height:
                    cv2.circle(image, (int(gaze_x), int(gaze_y)), 15, (0, 0, 255), 3)
                    cv2.circle(image, (int(gaze_x), int(gaze_y)), 5, (0, 0, 255), -1)
                    cv2.line(image, (int(gaze_x) - 20, int(gaze_y)), 
                            (int(gaze_x) + 20, int(gaze_y)), (0, 0, 255), 2)
                    cv2.line(image, (int(gaze_x), int(gaze_y) - 20), 
                            (int(gaze_x), int(gaze_y) + 20), (0, 0, 255), 2)
            
            output_path = f"keyframe_{self.patch_timestamp:.1f}s.jpg"
            cv2.imwrite(output_path, image)
        except:
            pass
    
    def log_gemini_interaction(self, role, content, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime('%H:%M:%S')
        
        self.gemini_log.append({
            'timestamp': timestamp,
            'role': role,
            'content': content
        })
    
    def process_videollama3(self, image_path, task_type="object_detection"):
        self.initialize_videollama3()
        
        conversation = [
            {
                "role": "system",
                "content": "You are an expert visual analysis assistant specialized in object detection and detailed description."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_path}},
                    {"type": "text", "text": "Please analyze this image and provide a comprehensive description of all visible objects with their exact locations. For each object you identify, provide:\n\n" +
                    "1. **Object Name**: Clear identification of what the object is\n" +
                    "2. **Bounding Box**: Exact coordinates in format [[x0,y0,x1,y1]] where (x0,y0) is top-left corner and (x1,y1) is bottom-right corner\n" +
                    "3. **Object Attributes**: Describe physical properties (color, size, shape, texture, material, condition)\n" +
                    "4. **Spatial Position**: Describe where the object is located in the physical space (e.g., 'on the table', 'hanging on the wall', 'placed in the corner', 'sitting on the floor', 'mounted above the fireplace')\n\n" +
                    "**Output Format for each object:**\n" +
                    "Object: [object_name]\n" +
                    "Location: [[x0,y0,x1,y1]]\n" +
                    "Description: [detailed description]\n" +
                    "Position: [spatial position description]\n\n" +
                    "**Important Guidelines:**\n" +
                    "- Include ALL clearly visible objects (furniture, tools, containers, appliances, etc.)\n" +
                    "- Provide accurate bounding box coordinates for each object\n" +
                    "- Be precise with object identification - only describe what you can clearly see\n" +
                    "- Use specific descriptive language for colors, materials, and conditions\n" +
                    "- For spatial position, focus on WHERE the object is located, not what it might be used for\n" +
                    "- Avoid speculation about object function or purpose\n" +
                    "- Organize your response with clear separation between objects\n" +
                    "- Do not include background elements like walls, floors, or lighting unless they are specific objects"}
                ]
            },
        ]
        
        for attempt in range(3):
            try:
                inputs = self.processor(conversation=conversation, add_system_prompt=True, 
                                      add_generation_prompt=True, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
                output_ids = self.model.generate(**inputs, max_new_tokens=2000, temperature=0.1, no_repeat_ngram_size=10)
                response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                if response and len(response.strip()) > 50:
                    return response
                else:
                    log_timestamp(f"尝试 {attempt + 1}: 分析结果为空")
                    
            except Exception as e:
                log_timestamp(f"尝试 {attempt + 1} 失败: {str(e)}")
                
            if attempt < 2:
                time.sleep(2)
        
        raise Exception("物体检测失败")
    
    def process_video_segment(self, start_second=None, end_second=None):
        if start_second is not None and end_second is not None:
            log_timestamp(f"正在分析视频片段: {start_second}s - {end_second}s")
        else:
            log_timestamp("正在分析完整视频")
        
        self.clear_cache()
        self.initialize_videollama3()
        
        if start_second is not None and end_second is not None:
            time_instruction = f"**IMPORTANT**: Focus ONLY on the time range from {start_second} seconds to {end_second} seconds in the video. Ignore all content outside this time range."
            task_description = f"**TASK**: Analyze the camera holder's actions and behaviors within the specified time range ({start_second}s - {end_second}s) in this first-person perspective video."
        else:
            time_instruction = "**IMPORTANT**: Analyze the entire video from beginning to end."
            task_description = "**TASK**: Analyze the camera holder's actions and behaviors throughout this entire first-person perspective video."
        
        video_conversation = [
            {
                "role": "system",
                "content": "You are an expert video analysis assistant specialized in first-person perspective behavioral analysis and temporal action segmentation."
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": VIDEO_PATH, "fps": 1, "max_frames": 100}},
                    {"type": "text", "text": f"{time_instruction}\n\n{task_description}\n\n" +
                    "## **Output Format:**\n" +
                    "List each meaningful action with its timestamp in the following format:\n\n" +
                    "**Action [N]: [start_time]s - [end_time]s**\n" +
                    "[Brief description of the action]\n\n" +
                    "## **Analysis Guidelines:**\n" +
                    "1. **ONLY include meaningful actions:**\n" +
                    "   • Large-scale movements (walking, moving between locations)\n" +
                    "   • Movement from one location to another\n" +
                    "   • Interactions with objects (picking up, putting down, using, manipulating)\n" +
                    "   • Significant changes in camera direction or focus\n\n" +
                    "2. **DO NOT include:**\n" +
                    "   • Minor head movements or camera adjustments\n" +
                    "   • Simply looking at something without interaction\n" +
                    "   • Pauses or waiting periods\n" +
                    "   • Trivial movements\n\n" +
                    "3. **Timestamp Requirements:**\n" +
                    "   • Provide accurate start and end times for each action\n" +
                    "   • Use format: [start_time]s - [end_time]s\n" +
                    "   • Times should be in seconds (e.g., 10s - 15s)\n\n" +
                    "4. **Action Descriptions:**\n" +
                    "   • Be concise but specific\n" +
                    "   • Focus on what the camera holder is doing\n" +
                    "   • Include relevant object names when applicable\n\n" +
                    "**Output only the actions in the specified format. Do not include any other text or explanations.**"}
                ]
            },
        ]
        
        for attempt in range(3):
            try:
                video_inputs = self.processor(conversation=video_conversation, add_system_prompt=True, 
                                            add_generation_prompt=True, return_tensors="pt")
                video_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in video_inputs.items()}
                if "pixel_values" in video_inputs:
                    video_inputs["pixel_values"] = video_inputs["pixel_values"].to(torch.bfloat16)
                
                video_output_ids = self.model.generate(**video_inputs, max_new_tokens=2000, temperature=0.1)
                video_response = self.processor.batch_decode(video_output_ids, skip_special_tokens=True)[0].strip()
                
                if video_response and len(video_response.strip()) > 50:
                    log_timestamp("视频分析完成")
                    return video_response
                else:
                    log_timestamp(f"尝试 {attempt + 1}: 分析结果为空")
                    
            except Exception as e:
                log_timestamp(f"尝试 {attempt + 1} 失败: {str(e)}")
                
            if attempt < 2:
                time.sleep(3)
        
        raise Exception("视频分析失败")
    
    def setup_llama_api(self):
        log_timestamp("正在初始化Llama 4 Maverick API")
        
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "REFINE_SEGMENT",
                    "description": "Returns a finer-grain breakdown of that time interval, further subdivided into shorter scene/event descriptions with timestamps",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_second": {"type": "number", "description": "Start time in seconds"},
                            "end_second": {"type": "number", "description": "End time in seconds"}
                        },
                        "required": ["start_second", "end_second"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "REFINE_FRAME",
                    "description": "Extracts and analyzes a specific frame at the given timestamp, providing detailed object detection with names, locations, attributes, and spatial positions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp_second": {"type": "number", "description": "Timestamp in seconds to extract and analyze the frame"}
                        },
                        "required": ["timestamp_second"]
                    }
                }
            }
        ]
        
        system_message = """##Overall Task
Imagine you are a user who has been wearing an AR/VR headset for an extended period, during which the device continuously records your surroundings. From this full recording, I will select a short clip: video V. Your job is to envision a daily-life scenario S that occurs any amount of time after the events shown in video V have ended.

Within this scenario S, think of a question Q that the user might naturally ask the AR/VR device; the answer to this question must require the device to review video V. The question Q should be rooted in everyday life, described as unambiguously as possible, and fully consistent with scenario S. Then provide the correct answer A to that question. Answer A must be absolutely accurate and unambiguous.

## Context you will receive
1. **Segment list**: an ordered set of action descriptions in video V with timestamp. 
2. **QA key frame**: a detailed object detection analysis of a single frame sampled from somewhere in the video V with timestamp. 
3. **Selected Object**: One object from the key frame has been specifically chosen. Your eventual question or answer **MUST** be related to this selected object.
**CRITICAL** The analysis may contain errors, please cross-verify all facts using independent sources.

## Tools you can call
You **CANNOT** see raw video, but you can make openai style function calls to gather more information:

**REFINE_SEGMENT(start_second, end_second)**  
 • Analyzes a specific time interval in detail
 • Returns temporal actions and behaviors during that period
 • Best for understanding: movements, actions, interactions over time
 • Use when you need to know "what happened when"

**REFINE_FRAME(timestamp_second)**  
 • Extracts and analyzes a single frame at the specified timestamp
 • Returns detailed object detection: names, locations, attributes, spatial positions
 • Best for understanding: what objects are present, their properties, spatial layout
 • Use when you need to know "what objects were there at that moment"

##Complete overall task in following steps
1. Analyze full segment list and the QA key-frame description. Identify the selected object information provided in the context.
2. Brainstorm situations in which you might ask the device a daily-life question that requires information from the video V to answer.
3. Use the REFINE_SEGMENT and REFINE_FRAME tools to gather additional information needed for giving the question and answer.
 • After each function response, briefly reflect on what you learned before deciding whether another call is necessary.
 • Feel free to chain function calls: study responses, think, then request another refinement until you believe you understand enough to craft a good question-and-answer pair.
4. Use the given tools to cross-verify each facts in QA.
 • **CRITICAL** Questions and Answers must be supported by facts from at least **TWO** independent sources (frames or segment analyses)
 • **DO NOT** use duplicate timestamps for cross-verify; instead, you may use other timestamps for frame refine verification or different time intervals for segment refine verification.
5. When satisfied, produce your final output (see format below) and stop calling function.
 • Strictly follow the required format and do not generate any additional content.
 • **CRITICAL** The question or answer **MUST** be related to this selected object.

##Final output format
scenario:
<one-sentence description of the daily-life scenario when the user would ask>
Question:
<what the user says to the AR/VR assistant>
Answer:
<absolutely accurate and unambiguous answer to the Question>
Evidence:
<quote to frame or segment refine for cross-verify and QA evidence>

##Important Notes (**CRITICAL**)
1. Regarding scenario:
 • It must depict an everyday situation.
 • Scenario S should take place some time after video V ends. It does not have to be directly related to video V, but the question Q must be relevant to Scenario S.
2. Regarding QA:
 • Descriptions must be precise and answers absolutely correct. Use enough qualifiers (location, appearance) to make each object unambiguous.
 • Q or A must involve selected object.
 • No speculation: the absence of evidence in the video does not prove something never happened.
 • Q should be realistic, as if asked by an actual user.
 • If these conditions cannot be met, create a new QA pair.
3. Regarding Evidence:
 • Every fact in the QA must be backed by at least two independent information sources. If this cannot be satisfied, create a new QA pair.
4. output:
 • Never invent facts, rely only on the provided descriptions.
 • **CRITICAL** Use the standard OpenAI function-calling format for tool calls; do not invoke tools with plain text."""
        
        return client, tools, system_message
    
    def get_response_text_safely(self, response):
        try:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    return message.content
            return None
        except:
            return None
    
    def get_token_usage_safely(self, response):
        try:
            if hasattr(response, 'usage') and response.usage:
                return {
                    'prompt_token_count': getattr(response.usage, 'prompt_tokens', 0),
                    'candidates_token_count': getattr(response.usage, 'completion_tokens', 0),
                    'total_token_count': getattr(response.usage, 'total_tokens', 0)
                }
            return None
        except:
            return None
    
    def parse_text_function_calls(self, response_content):
        """解析文本中的JSON函数调用"""
        text_function_calls = []
        if response_content:
            try:
                # 模式1: 完整格式 {"type": "function", "name": "...", "parameters": {...}}
                full_pattern = r'\{[^{}]*"type"[^{}]*"function"[^{}]*"name"[^{}]*"parameters"[^{}]*\{[^{}]*\}[^{}]*\}'
                full_matches = re.findall(full_pattern, response_content)
                
                # 模式2: 简化格式 {"name": "...", "parameters": {...}}
                simple_pattern = r'\{[^{}]*"name"[^{}]*"parameters"[^{}]*\{[^{}]*\}[^{}]*\}'
                simple_matches = re.findall(simple_pattern, response_content)
                
                all_matches = full_matches + simple_matches
                
                for match in all_matches:
                    try:
                        func_json = json.loads(match)
                        # 检查是否包含必要字段
                        if "name" in func_json and "parameters" in func_json:
                            # 检查函数名是否是我们支持的
                            if func_json["name"] in ["REFINE_SEGMENT", "REFINE_FRAME"]:
                                # 标准化格式，确保有type字段
                                if "type" not in func_json:
                                    func_json["type"] = "function"
                                text_function_calls.append(func_json)
                    except:
                        continue
            except:
                pass
        return text_function_calls
    
    def create_api_call(self, client, messages, tools):
        """创建API调用"""
        return client.chat.completions.create(
            model="google/gemini-2.5-pro",
            #model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1,
            max_tokens=3000
        )
    
    def process_function_call(self, func_call, messages, client, tools, total_token_usage):
        """处理单个函数调用"""
        if func_call['name'] == "REFINE_SEGMENT":
            return self._handle_refine_segment(func_call, messages, client, tools, total_token_usage)
        elif func_call['name'] == "REFINE_FRAME":
            return self._handle_refine_frame(func_call, messages, client, tools, total_token_usage)
        return None
    
    def _handle_refine_segment(self, func_call, messages, client, tools, total_token_usage):
        """处理REFINE_SEGMENT函数调用"""
        args = json.loads(func_call['arguments'])
        start_second = float(args["start_second"])
        end_second = float(args["end_second"])
        
        log_timestamp(f"Llama请求细化分析: {start_second}s - {end_second}s")
        self.log_gemini_interaction("Function Call", f"REFINE_SEGMENT({start_second}, {end_second})")
        
        try:
            refined_result = self.process_video_segment(start_second, end_second)
            self.log_gemini_interaction("Refined Analysis", refined_result)
            
            messages.append({
                "role": "tool",
                "tool_call_id": func_call['call_id'],
                "content": refined_result
            })
            
            return self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_SEGMENT")
            
        except Exception as e:
            error_msg = f"Error analyzing segment {start_second}-{end_second}: {str(e)}"
            log_timestamp(f"分析失败: {error_msg}")
            self.log_gemini_interaction("Error", error_msg)
            
            try:
                messages.append({
                    "role": "tool",
                    "tool_call_id": func_call['call_id'],
                    "content": error_msg
                })
                
                return self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_SEGMENT error")
            except:
                return None
    
    def _handle_refine_frame(self, func_call, messages, client, tools, total_token_usage):
        """处理REFINE_FRAME函数调用"""
        args = json.loads(func_call['arguments'])
        timestamp_second = float(args["timestamp_second"])
        
        log_timestamp(f"Llama请求帧分析: {timestamp_second}s")
        self.log_gemini_interaction("Function Call", f"REFINE_FRAME({timestamp_second})")
        
        try:
            temp_frame_path = self.extract_frame(VIDEO_PATH, timestamp_second)
            frame_analysis = self.process_videollama3(temp_frame_path)
            
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            self.log_gemini_interaction("Frame Analysis", frame_analysis)
            
            messages.append({
                "role": "tool",
                "tool_call_id": func_call['call_id'],
                "content": frame_analysis
            })
            
            return self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_FRAME")
            
        except Exception as e:
            error_msg = f"Error analyzing frame at {timestamp_second}s: {str(e)}"
            log_timestamp(f"帧分析失败: {error_msg}")
            self.log_gemini_interaction("Error", error_msg)
            
            try:
                messages.append({
                    "role": "tool",
                    "tool_call_id": func_call['call_id'],
                    "content": error_msg
                })
                
                return self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_FRAME error")
            except:
                return None
    
    def _make_api_call_with_token_tracking(self, client, messages, tools, total_token_usage, operation_name):
        """执行API调用并跟踪token使用"""
        response = self.create_api_call(client, messages, tools)
        
        token_usage = self.get_token_usage_safely(response)
        if token_usage:
            for key in total_token_usage:
                total_token_usage[key] += token_usage[key]
            self.log_gemini_interaction("Token Usage", f"{operation_name} response: {token_usage}")
        
        response_text = self.get_response_text_safely(response)
        if response_text:
            self.log_gemini_interaction("Llama Response", response_text)
        else:
            self.log_gemini_interaction("Llama Response", "继续思考中...")
        
        return response
    
    def send_continue_message(self, client, messages, tools):
        continue_prompts = [
            "Continue your analysis.",
            "Complete your response."
        ]
        
        for prompt in continue_prompts:
            try:
                continue_messages = messages + [{"role": "user", "content": prompt}]
                response = self.create_api_call(client, continue_messages, tools)
                text = self.get_response_text_safely(response)
                if text:
                    return response, text
                time.sleep(1)
            except:
                continue
        
        return None, None

    def interact_with_llama(self):
        client, tools, system_message = self.setup_llama_api()
        
        total_token_usage = {
            'prompt_token_count': 0,
            'candidates_token_count': 0,
            'total_token_count': 0
        }
        
        # 构建初始消息
        selected_object_info = ""
        if self.selected_object:
            selected_object_info = f"""
Selected Object for Question Focus:
Object Name: {self.selected_object['name']}
Object Location: {self.selected_object['bbox']}
Object Description: {self.selected_object['description']}
Object Position: {self.selected_object['position']}
"""
        
        user_prompt = f"""Here is the QA key-frame description:
Key frame sampled from video at {self.patch_timestamp:.1f} seconds with object detection analysis
{self.patch_description}
{selected_object_info}
Here is the Full segment list description:
{self.full_video_description}"""
        
        self.log_gemini_interaction("User Prompt", user_prompt)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        # 初始API调用
        response = self.create_api_call(client, messages, tools)
        
        # 统计初始响应的token使用
        token_usage = self.get_token_usage_safely(response)
        if token_usage:
            for key in total_token_usage:
                total_token_usage[key] += token_usage[key]
            self.log_gemini_interaction("Token Usage", f"Initial response: {token_usage}")
        
        response_text = self.get_response_text_safely(response)
        if response_text:
            self.log_gemini_interaction("Llama Response", response_text)
        else:
            self.log_gemini_interaction("Llama Response", "Function call initiated")
        
        # 处理函数调用循环
        max_iterations = 10
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            # 检查标准工具调用和文本格式函数调用
            has_tool_calls = hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls
            response_content = self.get_response_text_safely(response)
            text_function_calls = self.parse_text_function_calls(response_content) if not has_tool_calls else []
            
            # 如果没有任何函数调用，跳出循环
            if not has_tool_calls and not text_function_calls:
                break
            
            # 准备函数调用处理
            function_calls_to_process = []
            
            if has_tool_calls:
                messages.append(response.choices[0].message)
                for tool_call in response.choices[0].message.tool_calls:
                    function_calls_to_process.append({
                        'call_id': tool_call.id,
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments
                    })
            else:
                messages.append({"role": "assistant", "content": response_content})
                for i, func_call in enumerate(text_function_calls):
                    function_calls_to_process.append({
                        'call_id': f"text_call_{iteration_count}_{i}",
                        'name': func_call['name'],
                        'arguments': json.dumps(func_call['parameters'])
                    })
            
            # 处理函数调用
            for func_call in function_calls_to_process:
                response = self.process_function_call(func_call, messages, client, tools, total_token_usage)
                if response is None:
                    break
        
        # 获取最终结果
        final_result = self.get_response_text_safely(response)
        
        if not final_result:
            log_timestamp("Llama没有返回最终结果，尝试请求继续...")
            response, final_result = self.send_continue_message(client, messages, tools)
            
            if response:
                token_usage = self.get_token_usage_safely(response)
                if token_usage:
                    for key in total_token_usage:
                        total_token_usage[key] += token_usage[key]
                    self.log_gemini_interaction("Token Usage", f"Continue message response: {token_usage}")
        
        if not final_result:
            final_result = "Llama分析未能完成，请检查API状态或重试。"
            log_timestamp(f"使用默认响应")
        
        log_timestamp("Llama分析完成")
        print(final_result)
        
        # 输出总的token使用统计
        if total_token_usage['total_token_count'] > 0:
            print(f"\n=== Token使用统计 ===")
            print(f"输入Token数量: {total_token_usage['prompt_token_count']:,}")
            print(f"输出Token数量: {total_token_usage['candidates_token_count']:,}")
            print(f"总Token数量: {total_token_usage['total_token_count']:,}")
            
            self.log_gemini_interaction("Final Token Usage", 
                f"Total usage - Prompt: {total_token_usage['prompt_token_count']:,}, "
                f"Candidates: {total_token_usage['candidates_token_count']:,}, "
                f"Total: {total_token_usage['total_token_count']:,}")
            
            log_timestamp(f"Token使用统计 - 总计: {total_token_usage['total_token_count']:,} tokens")
        
        self.save_gemini_log()
        return final_result
    
    def save_gemini_log(self):
        with open("gemini_log.txt", "w", encoding="utf-8") as f:
            f.write("=== Llama完整交互记录 ===\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for entry in self.gemini_log:
                f.write(f"[{entry['timestamp']}] {entry['role']}:\n")
                f.write(f"{entry['content']}\n\n")
                f.write("-" * 80 + "\n\n")

def main():
    log_timestamp("启动AI Video Analysis Agent")
    
    if not os.path.exists(VIDEO_PATH):
        log_timestamp(f"视频文件不存在: {VIDEO_PATH}")
        return
    
    agent = VideoAnalysisAgent()
    
    try:
        log_timestamp("初始化完成")
        
        agent.load_gaze_data(VIDEO_PATH)
        
        log_timestamp("正在抽取关键帧")
        image_path = agent.extract_frame(VIDEO_PATH)
        log_timestamp(f"关键帧抽取完成: {image_path}")
        
        log_timestamp("正在进行物体检测")
        response = agent.process_videollama3(image_path)
        agent.patch_description = response
        agent.detected_objects = agent.parse_objects(response, image_path)
        
        if agent.detected_objects:
            agent.current_gaze_point = agent.get_gaze_point(agent.patch_timestamp)
            agent.selected_object = agent.select_object_by_gaze(
                agent.detected_objects, agent.current_gaze_point, sigma=400)
            
            log_timestamp(f"选中物体: {agent.selected_object['name']}")
            agent.visualize_keyframe(image_path, agent.detected_objects, agent.current_gaze_point)
        
        log_timestamp("物体检测完成")
        
        result = agent.process_video_segment()
        agent.full_video_description = result
        
        agent.interact_with_llama()
        
        log_timestamp("分析完成")
        
    except Exception as e:
        log_timestamp(f"执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()