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
import base64
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from openai import OpenAI
# 新增导入用于视频处理和聚类
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import subprocess
import tempfile

# 配置
device = "cuda:0"
#VIDEO_PATH = "/home/wang/AriaEveryday_activaties/loc5_script4_seq6_rec1/loc5_script4_seq6_rec1.mp4"
VIDEO_PATH = "/home/wang/AriaEveryday_activaties/loc3_script5_seq6_rec1/loc3_script5_seq6_rec1.mp4"
#VIDEO_PATH = "/home/wang/AriaEveryday_activaties/loc4_script1_seq1_rec1/loc4_script1_seq1_rec1.mp4"
OPENROUTER_API_KEY = "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"

def log_timestamp(message):
    """统一的时间戳日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class SegmentFeatureExtractor:
    """用于视频片段聚类的特征提取器"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化ResNet用于特征提取
        self.feature_model = resnet50(pretrained=True)
        self.feature_model.fc = torch.nn.Identity()
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_video_segment(self, video_path, start_second, end_second, output_path):
        """使用ffmpeg提取视频片段"""
        try:
            cmd = [
                'ffmpeg', '-y',  # 覆盖输出文件
                '-i', video_path,
                '-ss', str(start_second),
                '-t', str(end_second - start_second),
                '-c', 'copy',  # 不重新编码，直接复制
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_timestamp(f"ffmpeg 错误: {result.stderr}")
                return False
            
            return os.path.exists(output_path)
        except Exception as e:
            log_timestamp(f"视频切割失败: {str(e)}")
            return False

    def extract_features_from_segment(self, video_path, sample_rate=5):
        """从视频片段提取特征"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        features = []
        frame_timestamps = []
        frame_count = 0
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                    feature = self.feature_model(input_tensor)
                    features.append(feature.cpu().numpy())
                    frame_timestamps.append(frame_count / fps)
                
                frame_count += 1
        
        cap.release()
        
        if not features:
            return None, None
            
        return np.concatenate(features, axis=0), frame_timestamps
    
    def cluster_segment_frames(self, features, n_clusters=10):
        """对帧特征进行聚类"""
        if len(features) < n_clusters:
            n_clusters = max(1, len(features))
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predictions = kmeans.fit_predict(features_scaled)
        
        return predictions
    
    def get_representative_frames(self, video_path, start_second, end_second, n_clusters=10):
        """获取视频片段的代表帧"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 提取视频片段
            segment_path = os.path.join(temp_dir, "segment.mp4")
            if not self.extract_video_segment(video_path, start_second, end_second, segment_path):
                return []
            
            # 提取特征
            features, frame_timestamps = self.extract_features_from_segment(segment_path)
            if features is None:
                return []
            
            # 聚类
            predictions = self.cluster_segment_frames(features, n_clusters)
            
            # 为每个聚类找到代表帧（聚类中心最近的帧）
            representative_frames = []
            unique_clusters = np.unique(predictions)
            
            for cluster_id in unique_clusters:
                cluster_indices = np.where(predictions == cluster_id)[0]
                cluster_features = features[cluster_indices]
                
                # 计算聚类中心
                cluster_center = np.mean(cluster_features, axis=0)
                
                # 找到最接近中心的帧
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                center_idx = cluster_indices[np.argmin(distances)]
                
                # 计算在原视频中的实际时间戳
                relative_timestamp = frame_timestamps[center_idx]
                actual_timestamp = start_second + relative_timestamp
                
                representative_frames.append({
                    'cluster_id': int(cluster_id),
                    'timestamp': actual_timestamp,
                    'relative_timestamp': relative_timestamp
                })
            
            # 按时间戳排序
            representative_frames.sort(key=lambda x: x['timestamp'])
            
            return representative_frames

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
        # 新增：用于视频处理的特征提取器
        self.feature_extractor = None
        # 新增：Gemini物体检测器
        self.object_detector = None
        # 新增：QA精炼代理
        self.qa_refine_agent = None
        
    def initialize_feature_extractor(self):
        """初始化特征提取器用于视频聚类"""
        if self.feature_extractor is None:
            log_timestamp("正在加载特征提取器...")
            self.feature_extractor = SegmentFeatureExtractor()
            log_timestamp("特征提取器加载完成")
    
    def initialize_object_detector(self):
        """初始化Gemini物体检测器"""
        if self.object_detector is None:
            self.object_detector = GeminiObjectDetector()
    
    def initialize_qa_refine_agent(self):
        """初始化QA精炼代理"""
        if self.qa_refine_agent is None:
            self.qa_refine_agent = QARefineAgent()
    
    def detect_objects_with_gemini(self, image_path):
        """使用Gemini进行物体检测"""
        self.initialize_object_detector()
        
        # 使用Gemini检测物体
        response_text = self.object_detector.detect_objects(image_path)
        
        # 记录Gemini的响应
        self.patch_description = response_text
        
        # 解析检测结果
        detected_objects = self.object_detector.parse_gemini_objects(response_text, image_path)
        
        return detected_objects
    
    def load_video_summary(self, video_path):
        """加载视频的summary.json文件"""
        try:
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            summary_path = os.path.join(video_dir, f"{video_name}_summary.json")
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                # 将summary数据转换为我们需要的格式
                segments_text = []
                for segment in summary_data:
                    start_time = segment['start_time']
                    end_time = segment['end_time']
                    description = segment['description']
                    
                    # 直接使用秒格式，不转换为分钟
                    segments_text.append(f"{start_time}s-{end_time}s:{description}")
                
                return "\n".join(segments_text)
            else:
                log_timestamp(f"未找到summary文件: {summary_path}")
                return None
        except Exception as e:
            log_timestamp(f"加载summary文件失败: {str(e)}")
            return None
        
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
                if len(lines) < 2:
                    continue
                
                object_name = lines[0].strip()
                location_line = None
                
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith("Location:"):
                        location_line = line
                        break
                
                bbox = None
                if location_line:
                    bbox_match = re.search(r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]', location_line)
                    if not bbox_match:
                        bbox_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', location_line)
                    
                    if bbox_match:
                        x0, y0, x1, y1 = map(int, bbox_match.groups())
                        # 保存像素坐标用于可视化
                        pixel_bbox = [
                            int(x0 / 1000 * img_width), int(y0 / 1000 * img_height),
                            int(x1 / 1000 * img_width), int(y1 / 1000 * img_height)
                        ]
                        # 保存归一化坐标用于Gemini
                        normalized_bbox = [x0, y0, x1, y1]
                        bbox = pixel_bbox
                
                if bbox and object_name:
                    objects.append({
                        'name': object_name,
                        'bbox': bbox,
                        'normalized_bbox': normalized_bbox if bbox_match else None
                    })
            except:
                continue
        
        return objects
    
    def convert_bbox_to_gemini_format(self, normalized_bbox):
        """
        转换bbox格式为Gemini的[ymin,xmin,ymax,xmax]格式
        输入的坐标已经是归一化到0-1000范围的
        """
        # 如果输入已经是Gemini格式[ymin,xmin,ymax,xmax]，直接返回
        if len(normalized_bbox) == 4:
            # 检查是否是[x0,y0,x1,y1]格式，需要转换为[ymin,xmin,ymax,xmax]
            x0, y0, x1, y1 = normalized_bbox
            return [y0, x0, y1, x1]  # [ymin, xmin, ymax, xmax]
        
        return normalized_bbox
    
    def encode_image_to_base64(self, image_path):
        """
        将图片编码为base64格式，用于OpenRouter API
        """
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 确定图片类型
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # 默认
        
        return f"data:{mime_type};base64,{base64_image}"
    
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
            
            # 保存带bbox的图片到不同的文件名，不覆盖原图
            output_path = f"keyframe_{self.patch_timestamp:.1f}s_with_bbox.jpg"
            cv2.imwrite(output_path, image)
            log_timestamp(f"保存带bbox的可视化图片: {output_path}")
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
                "content": "You are an expert visual analysis assistant specialized in object detection."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_path}},
                    {"type": "text", "text": "Please analyze this image and identify all visible objects with their locations. For each object you identify, provide:\n\n" +
                    "1. **Object Name**: Clear identification of what the object is\n" +
                    "2. **Bounding Box**: Exact coordinates in format [[x0,y0,x1,y1]] where (x0,y0) is top-left corner and (x1,y1) is bottom-right corner\n\n" +
                    "**Output Format for each object:**\n" +
                    "Object: [object_name]\n" +
                    "Location: [[x0,y0,x1,y1]]\n\n" +
                    "**Important Guidelines:**\n" +
                    "- Include ALL clearly visible objects (furniture, tools, containers, appliances, etc.)\n" +
                    "- Provide accurate bounding box coordinates for each object\n" +
                    "- Be precise with object identification - only describe what you can clearly see\n" +
                    "- Do not include background elements like walls, floors, or lighting unless they are specific objects\n" +
                    "- Organize your response with clear separation between objects"}
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
            # 对于特定片段，使用聚类方法返回代表帧
            return self.process_segment_with_clustering(start_second, end_second)
        else:
            log_timestamp("正在加载完整视频描述")
            # 对于完整视频，从summary.json文件加载
            summary_description = self.load_video_summary(VIDEO_PATH)
            if summary_description:
                log_timestamp("从summary.json加载视频描述完成")
                return summary_description
            else:
                log_timestamp("未找到summary.json文件，使用fallback方法")
                return "无法加载视频描述，summary.json文件不存在"
    
    def process_segment_with_clustering(self, start_second, end_second):
        """使用聚类方法处理视频片段，返回代表帧信息"""
        try:
            self.initialize_feature_extractor()
            
            # 获取代表帧
            representative_frames = self.feature_extractor.get_representative_frames(
                VIDEO_PATH, start_second, end_second, n_clusters=10
            )
            
            if not representative_frames:
                return f"无法从时间段 {start_second}s - {end_second}s 提取代表帧"
            
            # 提取代表帧并保存为图片
            frame_info = []
            for i, frame_data in enumerate(representative_frames):
                timestamp = frame_data['timestamp']
                cluster_id = frame_data['cluster_id']
                
                # 提取并保存帧
                frame_path = f"cluster_frame_{cluster_id}_{timestamp:.1f}s.jpg"
                try:
                    self.extract_frame(VIDEO_PATH, timestamp)
                    # 重命名为更好的文件名
                    temp_frame_path = f"temp_frame_{timestamp:.1f}s.jpg"
                    if os.path.exists(temp_frame_path):
                        os.rename(temp_frame_path, frame_path)
                        log_timestamp(f"保存聚类代表帧: {frame_path}")
                    
                    frame_info.append({
                        'timestamp': timestamp,
                        'cluster_id': cluster_id,
                        'frame_path': frame_path
                    })
                except Exception as e:
                    log_timestamp(f"提取帧 {timestamp}s 失败: {str(e)}")
            
            # 构建返回信息
            if frame_info:
                result_lines = [f"视频片段 {start_second}s - {end_second}s 聚类分析结果:"]
                result_lines.append(f"通过聚类分析，该片段被分为 {len(frame_info)} 个子片段，以下是代表帧:")
                
                for frame in frame_info:
                    result_lines.append(f"• 子片段 {frame['cluster_id']}: 代表帧时间戳 {frame['timestamp']:.1f}s")
                
                result_lines.append(f"\n代表帧已保存，可用于进一步分析。")
                
                return "\n".join(result_lines)
            else:
                return f"无法从时间段 {start_second}s - {end_second}s 提取有效的代表帧"
                
        except Exception as e:
            log_timestamp(f"聚类分析失败: {str(e)}")
            return f"聚类分析失败: {str(e)}"
    
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
                    "description": "Samples multiple frames within a specified timestamp range and returns them in chronological order for visual analysis.",
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
                    "description": "Extracts a specific frame at the given timestamp and returns the actual image for visual analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp_second": {"type": "number", "description": "Timestamp in seconds to extract the frame image"}
                        },
                        "required": ["timestamp_second"]
                    }
                }
            }
        ]
        
        system_message = """##Overall Task
Imagine you are a user who has been wearing an AR/VR headset for an extended period, during which the device continuously recorded your surroundings. From this full recording, I will select a short clip: video V. Your job is to envision a daily-life scenario S that occurs any amount of time after the events shown in video V have ended.

Within this scenario S, think of a question Q that the user might naturally ask the AR/VR device; the answer to this question must require the device to review video V. The question Q should be rooted in everyday life, described as unambiguously as possible, and fully consistent with scenario S. Then provide the correct answer A to that question. Answer A must be absolutely accurate and unambiguous.

## Context you will receive
1. **Segment list**: an ordered set of action descriptions in video V with timestamp. 
2. **QA key frame**: a single frame image from video V with timestamp, showing visual content at that moment.
3. **Selected Object**: One object from the key frame has been specifically chosen, with its bounding box coordinates. Your eventual question or answer **MUST** be related to this selected object.
**CRITICAL** The analysis may contain errors, please cross-verify all facts using independent sources.

## Tools you can call
You **CANNOT** see raw video, but you can make openai style function calls to gather more information:

**REFINE_SEGMENT(start_second, end_second)**  
 • Extracts multiple frames within a specified timestamp range
 • Returns actual frames in chronological order for visual analysis
 • Best for understanding: movements, actions, interactions over time
 • Use when you need to know "what happened when"

**REFINE_FRAME(timestamp_second)**  
 • Extracts a single frame at the specified timestamp for visual analysis.
 • Returns the actual frame image at that timestamp
 • Best for understanding: what objects are present, their properties, spatial layout
 • Use when you need to know "what objects were there at that moment"

##Complete overall task in following steps
1. Analyze full segment list and the QA key-frame image. Identify the selected object information provided in the context.
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
    
    def get_reasoning_safely(self, response):
        """安全地获取思维链内容"""
        try:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                # 检查是否有reasoning字段
                if hasattr(message, 'reasoning') and message.reasoning:
                    return message.reasoning
                # 检查是否有reasoning_details字段（某些模型可能使用这个）
                elif hasattr(message, 'reasoning_details') and message.reasoning_details:
                    return str(message.reasoning_details)
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
            #model="openai/o4-mini",  # 使用high版本，相当于effort="high"
            #model="anthropic/claude-sonnet-4",
            #model="meta-llama/llama-4-maverick-17b-128e-instruct",
            #model="meta-llama/llama-3.2-90b-vision-instruct",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=1,
            max_tokens=65536
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
            # 使用聚类方法获取代表帧
            self.initialize_feature_extractor()
            
            representative_frames = self.feature_extractor.get_representative_frames(
                VIDEO_PATH, start_second, end_second, n_clusters=10
            )
            
            if not representative_frames:
                error_msg = f"无法从时间段 {start_second}s - {end_second}s 提取代表帧"
                messages.append({
                    "role": "tool",
                    "tool_call_id": func_call['call_id'],
                    "content": error_msg
                })
                return self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_SEGMENT error")
            
            # 提取并保存代表帧
            frame_paths = []
            frame_descriptions = []
            
            for frame_data in representative_frames:
                timestamp = frame_data['timestamp']
                cluster_id = frame_data['cluster_id']
                
                try:
                    # 提取帧
                    temp_frame_path = self.extract_frame(VIDEO_PATH, timestamp)
                    
                    # 重命名为更好的文件名
                    frame_path = f"refine_segment_{start_second:.1f}-{end_second:.1f}s_cluster_{cluster_id}_{timestamp:.1f}s.jpg"
                    if os.path.exists(temp_frame_path):
                        os.rename(temp_frame_path, frame_path)
                        frame_paths.append(frame_path)
                        frame_descriptions.append(f"{timestamp:.1f}s")
                        log_timestamp(f"保存REFINE_SEGMENT代表帧: {frame_path}")
                except Exception as e:
                    log_timestamp(f"提取帧 {timestamp}s 失败: {str(e)}")
            
            if not frame_paths:
                error_msg = f"无法提取时间段 {start_second}s - {end_second}s 的有效代表帧"
                messages.append({
                    "role": "tool",
                    "tool_call_id": func_call['call_id'],
                    "content": error_msg
                })
                return self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_SEGMENT error")
            
            # 构建包含代表帧的响应
            frame_info_text = f"Analysis completed. The following are representative frames sampled from different timestamps:"
            for i, desc in enumerate(frame_descriptions, 1):
                frame_info_text += f"\n{i}. {desc}"
            
            self.log_gemini_interaction("Refined Analysis", frame_info_text)
            
            # 添加工具响应消息
            messages.append({
                "role": "tool",
                "tool_call_id": func_call['call_id'],
                "content": frame_info_text
            })
            
            # 立即添加包含所有代表帧的用户消息
            content = [
                {"type": "text", "text": f"Here are the representative frames from {start_second}s - {end_second}s, each sampled from the specified timestamp:"}
            ]
            
            for i, frame_path in enumerate(frame_paths):
                timestamp = frame_descriptions[i]
                content.append({
                    "type": "text", 
                    "text": f"Frame sampled at {timestamp}:"
                })
                base64_image = self.encode_image_to_base64(frame_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                })
            
            messages.append({
                "role": "user",
                "content": content
            })
            
            response = self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_SEGMENT")
            
            # 不删除代表帧文件，让用户可以查看
            return response
            
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
            # 为refine_frame提取的图片使用更好的命名，不删除让用户可以查看
            refine_frame_path = f"refine_frame_{timestamp_second:.1f}s.jpg"
            temp_frame_path = self.extract_frame(VIDEO_PATH, timestamp_second)
            
            # 重命名临时文件为更好的名称
            if os.path.exists(temp_frame_path):
                os.rename(temp_frame_path, refine_frame_path)
                log_timestamp(f"保存REFINE_FRAME图片: {refine_frame_path}")
            
            frame_content = f"Frame extracted at {timestamp_second}s from video. This frame image is now available for analysis."
            
            self.log_gemini_interaction("Frame Analysis", f"Frame image extracted and saved: {refine_frame_path}")
            
            # 添加工具响应消息
            messages.append({
                "role": "tool",
                "tool_call_id": func_call['call_id'],
                "content": frame_content
            })
            
            # 立即添加包含图片的用户消息
            base64_image = self.encode_image_to_base64(refine_frame_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is the frame at {timestamp_second}s. Please analyze this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            })
            
            response = self._make_api_call_with_token_tracking(client, messages, tools, total_token_usage, "REFINE_FRAME")
            
            # 不删除refine_frame文件，让用户可以查看
            # if os.path.exists(refine_frame_path):
            #     os.remove(refine_frame_path)
            
            return response
            
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
        
        # 记录Token使用
        token_usage = self.get_token_usage_safely(response)
        if token_usage:
            for key in total_token_usage:
                total_token_usage[key] += token_usage[key]
            self.log_gemini_interaction("Token Usage", f"{operation_name} response: {token_usage}")
        
        # 记录思维链内容
        reasoning = self.get_reasoning_safely(response)
        if reasoning:
            self.log_gemini_interaction("Reasoning (思维链)", f"{operation_name} reasoning:\n{reasoning}")
        
        # 记录响应内容
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
                
                # 记录reasoning tokens
                reasoning = self.get_reasoning_safely(response)
                if reasoning:
                    self.log_gemini_interaction("Reasoning (思维链)", f"Continue message reasoning:\n{reasoning}")
                
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
            # 使用Gemini原生的bbox格式
            gemini_bbox = self.selected_object.get('gemini_bbox', self.selected_object['normalized_bbox'])
            
            selected_object_info = f"""
Selected Object for Question Focus:
Object Name: {self.selected_object['name']}
Object Bounding Box: {gemini_bbox} (format: [ymin, xmin, ymax, xmax], normalized 0-1000)
"""
        
        user_prompt = f"""Here is the QA key-frame image:
This is a key frame sampled from video at {self.patch_timestamp:.1f} seconds.
{selected_object_info}
Here is the Full segment list description:
{self.full_video_description}"""
        
        self.log_gemini_interaction("User Prompt", user_prompt)
        
        # 构建包含图片的消息
        keyframe_path = f"keyframe_{self.patch_timestamp:.1f}s.jpg"
        base64_image = self.encode_image_to_base64(keyframe_path)
        
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            }
        ]
        
        # 初始API调用
        response = self.create_api_call(client, messages, tools)
        
        # 统计初始响应的token使用
        token_usage = self.get_token_usage_safely(response)
        if token_usage:
            for key in total_token_usage:
                total_token_usage[key] += token_usage[key]
            self.log_gemini_interaction("Token Usage", f"Initial response: {token_usage}")
        
        # 记录初始响应的思维链内容
        reasoning = self.get_reasoning_safely(response)
        if reasoning:
            self.log_gemini_interaction("Reasoning (思维链)", f"Initial reasoning:\n{reasoning}")
        
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
        
        # 开始QA Refine流程
        log_timestamp("=" * 50)
        log_timestamp("开始QA精炼流程...")
        
        try:
            # 初始化QA refine agent
            self.initialize_qa_refine_agent()
            
            # 从Llama响应中提取QA对
            question, answer = self.qa_refine_agent.extract_qa_from_llama_response(final_result)
            
            if question != "无法提取问题" and answer != "无法提取答案":
                # 使用Gemini进行QA精炼
                refined_response = self.qa_refine_agent.refine_qa_to_mcq(question, answer)
                
                if refined_response:
                    # 解析多选题响应
                    refined_question, options, correct_index = self.qa_refine_agent.parse_mcq_response(refined_response)
                    
                    # 随机打乱选项
                    shuffled_options, new_correct_index = self.qa_refine_agent.shuffle_options(options, correct_index)
                    
                    # 输出最终结果
                    log_timestamp("QA精炼完成！最终多选题结果：")
                    print("\n" + "=" * 60)
                    print("📝 最终多选题结果")
                    print("=" * 60)
                    print(f"\n问题: {refined_question}")
                    print("\n选项:")
                    for i, option in enumerate(shuffled_options):
                        marker = "✓" if i == new_correct_index else " "
                        print(f"  {chr(65+i)}) {option} {marker}")
                    print(f"\n正确答案: {chr(65+new_correct_index)}")
                    print("=" * 60)
                    
                    # 保存到Gemini日志
                    self.log_gemini_interaction("QA Refine Result", 
                        f"Final MCQ:\nQuestion: {refined_question}\n" +
                        f"Options: {shuffled_options}\n" +
                        f"Correct Answer: {chr(65+new_correct_index)}")
                    
                    # 保存原始refine响应用于调试
                    self.log_gemini_interaction("QA Refine Raw Response", 
                        f"Gemini原始响应:\n{refined_response}")
                    
                    # 保存最终结果到文件
                    mcq_result = {
                        "question": refined_question,
                        "options": shuffled_options,
                        "correct_answer_index": new_correct_index,
                        "correct_answer_letter": chr(65+new_correct_index),
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    with open("final_mcq_result.json", "w", encoding="utf-8") as f:
                        json.dump(mcq_result, f, ensure_ascii=False, indent=2)
                    
                    log_timestamp("多选题结果已保存到 final_mcq_result.json")
                    
                else:
                    log_timestamp("QA精炼失败，返回原始结果")
            else:
                log_timestamp("无法提取有效的QA对，跳过精炼步骤")
                
        except Exception as e:
            log_timestamp(f"QA精炼过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
            f.write("=== AI助手完整交互记录（包含思维链） ===\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for entry in self.gemini_log:
                f.write(f"[{entry['timestamp']}] {entry['role']}:\n")
                f.write(f"{entry['content']}\n\n")
                f.write("-" * 80 + "\n\n")

class GeminiObjectDetector:
    """独立的Gemini物体检测器"""
    
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """初始化Gemini客户端"""
        log_timestamp("正在初始化Gemini物体检测器...")
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        log_timestamp("Gemini物体检测器初始化完成")
    
    def encode_image_to_base64(self, image_path):
        """将图片编码为base64格式"""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'
        
        return f"data:{mime_type};base64,{base64_image}"
    
    def detect_objects(self, image_path):
        """使用Gemini检测图片中的物体"""
        log_timestamp("开始使用Gemini进行物体检测...")
        
        base64_image = self.encode_image_to_base64(image_path)
        
        # 构建检测prompt
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
                model="google/gemini-2.5-pro",
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
            
            response_text = response.choices[0].message.content
            log_timestamp("Gemini物体检测完成")
            return response_text
            
        except Exception as e:
            log_timestamp(f"Gemini物体检测失败: {str(e)}")
            raise Exception(f"Gemini物体检测失败: {str(e)}")
    
    def parse_gemini_objects(self, response_text, image_path):
        """解析Gemini的物体检测结果"""
        objects = []
        
        # 添加调试输出
        log_timestamp("=== Gemini原始响应内容 ===")
        print(response_text)
        log_timestamp("=== 响应内容结束 ===")
        
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
        
        log_timestamp(f"图片尺寸: {img_width} x {img_height}")
        
        # 解析格式：[ymin,xmin,ymax,xmax]:<object_name>
        pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]:([^,\n\r]+)'
        matches = re.findall(pattern, response_text)
        
        log_timestamp(f"正则表达式匹配结果数量: {len(matches)}")
        if matches:
            log_timestamp("匹配的原始内容:")
            for i, match in enumerate(matches):
                log_timestamp(f"  {i+1}: {match}")
        
        # 尝试其他可能的格式
        if len(matches) == 0:
            log_timestamp("尝试其他可能的格式...")
            
            # 格式1: [ymin, xmin, ymax, xmax]: object_name (带空格)
            pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]:\s*([^\n\r]+)'
            matches2 = re.findall(pattern2, response_text)
            log_timestamp(f"格式2匹配数量: {len(matches2)}")
            if matches2:
                matches = matches2
                log_timestamp("使用格式2的匹配结果")
            
            # 格式3: [ymin,xmin,ymax,xmax] object_name (没有冒号)
            if len(matches) == 0:
                pattern3 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*([^\n\r\[]+)'
                matches3 = re.findall(pattern3, response_text)
                log_timestamp(f"格式3匹配数量: {len(matches3)}")
                if matches3:
                    matches = matches3
                    log_timestamp("使用格式3的匹配结果")
            
            # 格式4: 查找任何包含数字的方括号
            if len(matches) == 0:
                pattern4 = r'\[([^\]]+)\][:\s]*([^\n\r\[]+)'
                all_brackets = re.findall(pattern4, response_text)
                log_timestamp(f"所有方括号内容匹配数量: {len(all_brackets)}")
                for bracket_content, obj_name in all_brackets:
                    log_timestamp(f"  方括号内容: '{bracket_content}' -> 物体: '{obj_name.strip()}'")
        
        log_timestamp(f"最终匹配结果数量: {len(matches)}")
        
        for match in matches:
            try:
                ymin, xmin, ymax, xmax, object_name = match
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                object_name = object_name.strip()
                
                log_timestamp(f"正在处理物体: {object_name}, 坐标: [{ymin},{xmin},{ymax},{xmax}]")
                
                # 验证坐标范围
                if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 
                       0 <= ymax <= 1000 and 0 <= xmax <= 1000):
                    log_timestamp(f"跳过无效坐标的物体: {object_name} - 坐标超出范围")
                    continue
                
                if ymin >= ymax or xmin >= xmax:
                    log_timestamp(f"跳过无效bbox的物体: {object_name} - 坐标顺序错误")
                    continue
                
                # 转换为像素坐标
                pixel_bbox = [
                    int(xmin / 1000 * img_width),  # x0
                    int(ymin / 1000 * img_height), # y0
                    int(xmax / 1000 * img_width),  # x1
                    int(ymax / 1000 * img_height)  # y1
                ]
                
                # 保存归一化坐标用于后续处理
                normalized_bbox = [xmin, ymin, xmax, ymax]  # 转换为[x0,y0,x1,y1]格式
                
                objects.append({
                    'name': object_name,
                    'bbox': pixel_bbox,  # [x0,y0,x1,y1]像素坐标
                    'normalized_bbox': normalized_bbox,  # [x0,y0,x1,y1]归一化坐标
                    'gemini_bbox': [ymin, xmin, ymax, xmax]  # 保存原始Gemini格式
                })
                
                log_timestamp(f"✓ 成功添加物体: {object_name} at {pixel_bbox}")
                
            except Exception as e:
                log_timestamp(f"解析物体失败: {str(e)}, 原始match: {match}")
                continue
        
        log_timestamp(f"最终成功解析 {len(objects)} 个有效物体")
        return objects

class QARefineAgent:
    """用于将QA对转换为多选题格式的AI代理"""
    
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """初始化Gemini客户端"""
        log_timestamp("正在初始化QA Refine Agent...")
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        log_timestamp("QA Refine Agent初始化完成")
    
    def load_prompt_template(self):
        """使用内置的prompt模板"""
        return self.get_default_prompt()
    
    def get_default_prompt(self):
        """提供默认的prompt模板"""
        return """I have a Video Question Answering (VQA) pair that was generated by a large language model. The model watched a video and produced a question and an answer.

Here is the QA pair:
Question:
<question>
Answer:
<answer>

**Task Procedure:**
1. Carefully read and understand the provided VQA data. 
2. Rewrite the question so that:
 • it sounds like something a person would naturally ask in daily life
 • it can be answered succinctly
 • all descriptive modifiers as well as any time / location details from the original are fully preserved 
3. Convert the Answer into a multiple-choice question (MCQ) format with five options

**Requirements:**
1. The rewritten question must be phrased in a natural, conversational way that a human would use in daily life. It should not sound like a formal test question
2. You must create five options in total. One option must be the correct answer. The other four options must be incorrect distractors
3. These distractors should be challenging and highly plausible to confuse someone who has not paid close attention to the details. The distractors **MUST** be close in meaning to make them challenging
4. If the original question is inherently a Yes/No question, keep it in Yes/No form, but still generate five options. Each option should begin with "Yes," or "No," followed by a parallel detail

**Format Consistency**
All five answer options must be stylistically consistent. They should have a similar length, structure, and level of detail. This is to ensure one option doesn't stand out simply because of its format

Please only return the Question and five options, as well as the correct answer."""
    
    def extract_qa_from_llama_response(self, llama_response):
        """从Llama响应中提取Question和Answer"""
        try:
            # 查找Question和Answer的模式
            question_pattern = r"[Qq]uestion:\s*\n?(.*?)(?=\n[Aa]nswer:|$)"
            answer_pattern = r"[Aa]nswer:\s*\n?(.*?)(?=\n[Ee]vidence:|$)"
            
            question_match = re.search(question_pattern, llama_response, re.DOTALL)
            answer_match = re.search(answer_pattern, llama_response, re.DOTALL)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                log_timestamp(f"成功提取QA对:")
                log_timestamp(f"Question: {question}")
                log_timestamp(f"Answer: {answer}")
                
                return question, answer
            else:
                log_timestamp("无法从Llama响应中提取标准格式的QA对，尝试其他模式...")
                
                # 尝试其他可能的模式
                lines = llama_response.split('\n')
                question = None
                answer = None
                
                for i, line in enumerate(lines):
                    if line.strip().lower().startswith('question:'):
                        question = line.split(':', 1)[1].strip()
                        if i + 1 < len(lines):
                            question += " " + lines[i + 1].strip()
                    elif line.strip().lower().startswith('answer:'):
                        answer = line.split(':', 1)[1].strip()
                        if i + 1 < len(lines):
                            answer += " " + lines[i + 1].strip()
                
                if question and answer:
                    log_timestamp(f"使用备用方法提取QA对:")
                    log_timestamp(f"Question: {question}")
                    log_timestamp(f"Answer: {answer}")
                    return question, answer
                else:
                    log_timestamp("无法提取QA对，返回默认值")
                    return "无法提取问题", "无法提取答案"
                    
        except Exception as e:
            log_timestamp(f"提取QA对时出错: {str(e)}")
            return "提取问题失败", "提取答案失败"
    
    def refine_qa_to_mcq(self, question, answer):
        """将QA对转换为多选题格式"""
        try:
            # 加载prompt模板并替换占位符
            prompt_template = self.load_prompt_template()
            prompt = prompt_template.replace("<question>", question).replace("<answer>", answer)
            
            # 定义JSON Schema
            json_schema = {
                "name": "multiple_choice_question",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The refined question in natural, conversational language"
                        },
                        "option_a": {
                            "type": "string",
                            "description": "First answer option"
                        },
                        "option_b": {
                            "type": "string",
                            "description": "Second answer option"
                        },
                        "option_c": {
                            "type": "string",
                            "description": "Third answer option"
                        },
                        "option_d": {
                            "type": "string",
                            "description": "Fourth answer option"
                        },
                        "option_e": {
                            "type": "string",
                            "description": "Fifth answer option"
                        },
                        "correct_answer": {
                            "type": "string",
                            "description": "The letter of the correct answer (A, B, C, D, or E)"
                        }
                    },
                    "required": ["question", "option_a", "option_b", "option_c", "option_d", "option_e", "correct_answer"],
                    "additionalProperties": False
                }
            }
            
            log_timestamp("开始使用Gemini进行QA refinement（结构化输出）...")
            
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-pro",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                },
                temperature=1,
                max_tokens=65536
            )
            
            refined_response = response.choices[0].message.content
            log_timestamp("QA refinement（结构化输出）完成")
            
            return refined_response
            
        except Exception as e:
            log_timestamp(f"QA refinement失败: {str(e)}")
            return None
    

    
    def parse_mcq_response(self, refined_response):
        """解析JSON格式的多选题响应"""
        try:
            log_timestamp("开始解析JSON格式响应...")
            
            # 清理响应文本，提取JSON部分
            cleaned_response = refined_response.strip()
            
            # 如果响应被代码块包围，提取其中的JSON
            if cleaned_response.startswith('```'):
                # 提取```json 和 ``` 之间的内容，或者```和```之间的内容
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                if json_match:
                    cleaned_response = json_match.group(1).strip()
            
            # 如果有其他文本，尝试提取第一个完整的JSON对象
            brace_start = cleaned_response.find('{')
            if brace_start != -1:
                # 找到最后一个匹配的}
                brace_count = 0
                brace_end = brace_start
                for i, char in enumerate(cleaned_response[brace_start:], brace_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i
                            break
                cleaned_response = cleaned_response[brace_start:brace_end + 1]
            
            # 解析JSON
            parsed_data = json.loads(cleaned_response)
            
            # 提取字段
            refined_question = parsed_data.get('question', '解析问题失败')
            
            options = [
                parsed_data.get('option_a', '选项A'),
                parsed_data.get('option_b', '选项B'), 
                parsed_data.get('option_c', '选项C'),
                parsed_data.get('option_d', '选项D'),
                parsed_data.get('option_e', '选项E')
            ]
            
            correct_answer_letter = parsed_data.get('correct_answer', 'A').upper()
            if correct_answer_letter in 'ABCDE':
                correct_index = ord(correct_answer_letter) - ord('A')
            else:
                correct_index = 0
            
            log_timestamp(f"JSON解析成功：问题='{refined_question}', {len(options)}个选项, 正确答案={correct_answer_letter}")
            
            return refined_question, options, correct_index
            
        except Exception as e:
            log_timestamp(f"JSON解析失败: {str(e)}")
            # 返回默认值
            return "解析失败", ["选项A", "选项B", "选项C", "选项D", "选项E"], 0
    

    
    def shuffle_options(self, options, correct_index):
        """随机打乱选项顺序，并返回新的正确答案索引"""
        try:
            # 保存正确答案
            correct_answer = options[correct_index]
            
            # 创建选项索引列表并打乱
            indices = list(range(len(options)))
            random.shuffle(indices)
            
            # 重新排列选项
            shuffled_options = [options[i] for i in indices]
            
            # 找到正确答案的新位置
            new_correct_index = shuffled_options.index(correct_answer)
            
            log_timestamp(f"选项已随机打乱，正确答案从位置 {correct_index} 移动到位置 {new_correct_index}")
            
            return shuffled_options, new_correct_index
            
        except Exception as e:
            log_timestamp(f"打乱选项失败: {str(e)}")
            return options, correct_index

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
        agent.detected_objects = agent.detect_objects_with_gemini(image_path)
        
        if agent.detected_objects:
            agent.current_gaze_point = agent.get_gaze_point(agent.patch_timestamp)
            agent.selected_object = agent.select_object_by_gaze(
                agent.detected_objects, agent.current_gaze_point, sigma=400)
            
            log_timestamp(f"选中物体: {agent.selected_object['name']}")
            agent.visualize_keyframe(image_path, agent.detected_objects, agent.current_gaze_point)
        
        log_timestamp("物体检测完成")
        
        # 加载完整视频描述（从summary.json）
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
