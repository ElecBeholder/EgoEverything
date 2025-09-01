# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a modular Video Question Answering (VQA) generation system that processes ego-centric videos to generate multiple choice questions. The pipeline extracts key frames, detects objects, selects focus objects based on gaze data, and generates natural question-answer pairs that are refined into multiple choice format.

## 详细生成流程和逻辑

### 1. 整体流程架构 (VQAGenerationPipeline)

主流程在 `main.py` 的 `VQAGenerationPipeline.process_videos()` 中：

```
对于每个视频：
├── 1. 数据加载 (VideoLoader)
│   ├── 加载视频文件 (.mp4)
│   ├── 加载时间段摘要 (_summary.json)
│   └── 加载眼动数据 (_tracking.csv, 可选)
├── 2. 对象采样预处理 (ObjectSampler)
│   ├── 提取关键帧 (sampling_density控制, 默认60帧/秒)
│   ├── 多线程对象检测 (Gemini API, n_llms线程)
│   ├── CLIP特征聚类合并相似对象
│   └── 基于眼动数据选择关键对象 (question_factor倍数)
├── 3. 并行QA生成 (多线程)
│   ├── 为每个关键对象创建QA任务
│   ├── qa_n_llms个线程并行处理
│   └── 每线程独立的QA生成器和MCQ精化器
└── 4. 增量结果保存
    └── 每个视频完成后立即保存到JSON
```

### 2. 对象采样详细逻辑 (ObjectSampler)

#### 2.1 关键帧提取
```python
# 基于时间密度均匀采样
sampling_density = 60.0  # 默认每60帧采样1帧 (~1帧/秒)
keyframes = extract_keyframes_uniform(video_path, num_frames, temp_dir)
```

#### 2.2 对象检测 (多线程)
```python
# 多线程调用Gemini API进行对象检测
for keyframe in keyframes:
    # Gemini 2.5 Flash检测对象并返回边界框坐标
    detected_objects = object_detector.detect(keyframe)
    # 返回格式: {object_name, bbox, confidence, gemini_bbox}
```

#### 2.3 对象聚类合并
```python
# 使用CLIP特征计算相似度
for object in detected_objects:
    clip_feature = clip_model.encode(crop_object_region(object))
    
# 基于余弦相似度聚类相同对象实例
similarity_threshold = 0.85
merged_objects = cluster_similar_objects(objects, threshold)
```

#### 2.4 基于眼动的对象选择
```python
# 计算每个对象ID与眼动数据的距离
gaze_distances = {}
for object_id in unique_object_ids:
    instances = objects_by_id[object_id]
    distances = [calculate_gaze_distance(instance, gaze_data) for instance in instances]
    gaze_distances[object_id] = mean(distances)

# 高斯概率采样 - 距离越近概率越高
probabilities = [exp(-distance²/(2*σ²)) for distance in gaze_distances.values()]
selected_object_ids = random.choice(object_ids, p=probabilities, size=question_factor)
```

### 3. QA生成详细逻辑 (QAGenerator)

#### 3.1 工具调用系统
QA生成器配备两个工具供Gemini使用：

```python
tools = [
    "REFINE_SEGMENT(start_second, end_second)": "提取时间段内多帧图像",
    "REFINE_FRAME(timestamp_second)": "提取特定时间戳的单帧图像"
]
```

#### 3.2 生成流程
```python
# 系统提示要求Gemini扮演AR/VR用户
system_prompt = """
想象你是AR/VR设备用户，需要：
1. 分析视频片段和关键帧
2. 构思日常生活场景
3. 生成需要查看视频才能回答的问题
4. 使用工具验证事实并提供准确答案
"""

# 输入上下文
context = {
    "segment_list": video_summary,  # 时间段动作描述
    "keyframe_image": base64_image,  # 关键帧图像
    "selected_object": {             # 选中对象
        "name": object_name,
        "bbox": bounding_box_coords
    }
}

# Gemini使用工具进行多轮推理
conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": context},
    # Gemini可能调用REFINE_SEGMENT/REFINE_FRAME获取更多信息
    # 最终输出: Scenario + Question + Answer
]
```

### 4. MCQ精化逻辑 (MCQRefiner)

#### 4.1 结构化输出
```python
# 使用JSON Schema强制结构化输出
schema = {
    "refined_question": "优化后的问题",
    "options": ["选项A", "选项B", "选项C", "选项D", "选项E"],
    "correct_answer_index": 0,  # 正确答案索引
    "original_question": "原始问题",
    "original_answer": "原始答案"
}

response = gemini_api.call(
    messages=[{"role": "user", "content": qa_response}],
    response_format={"type": "json_object", "schema": schema}
)
```

### 5. 多线程并发控制

#### 5.1 对象检测线程池
```python
# 默认5个线程并行调用Gemini进行对象检测
n_llms = 5
with ThreadPoolExecutor(max_workers=n_llms) as executor:
    futures = [executor.submit(detect_objects, frame) for frame in keyframes]
```

#### 5.2 QA生成线程池
```python
# 默认30个线程并行生成QA
qa_n_llms = 30
for thread_id in range(qa_n_llms):
    # 每个线程独立的API客户端，避免对话污染
    local_qa_generator = QAGenerator(api_key)
    local_mcq_refiner = MCQRefiner(api_key)
    thread = Thread(target=qa_worker, args=(local_qa_generator, local_mcq_refiner))
```

### 6. 关键算法参数

#### 6.1 采样控制
- `sampling_density`: 关键帧采样密度 (默认60，约每秒1帧)
- `question_factor`: 问题数量倍数 (默认4，即 `4 × unique_objects` 个问题)
- `sigma`: 高斯采样参数 (默认400，控制眼动距离权重)

#### 6.2 相似度阈值
- `clip_similarity_threshold`: 0.85 (CLIP特征相似度阈值)
- `gaze_distance_threshold`: 像素距离计算眼动关注度

### 7. 输出格式
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
          "key_object": {"name": "coffee_mug", "bbox": [100,150,200,250]},
          "raw_output": {
            "raw_qa": "Where did I place the coffee mug? On the desk",
            "CoT": "完整的Gemini推理过程",
            "scenario": "用户寻找咖啡杯的场景"
          },
          "token_usage": 2300,
          "question": "Where did I put my coffee mug?",
          "answer": ["On the desk", "Kitchen counter", "Coffee table", "Bookshelf", "Windowsill"],
          "correct": 0
        }
      ]
    }
  ]
}
```

## Key Development Commands

### Running the Full Pipeline
```bash
python main.py \
    --api-key "your-openrouter-api-key" \
    --dataset-path "/path/to/dataset" \
    --json-path "/path/to/dataset.json" \
    --dataset-name "AriaEveryday_Activities" \
    --question-factor 4 \
    --sampling-density 60 \
    --n-llms 5 \
    --qa-n-llms 30
```

### Testing Single Video
```bash
python main.py test \
    --api-key "your-api-key" \
    --video-path "/path/to/video.mp4" \
    --sequence-id "video_id"
```

### Individual Module Testing
- `python object_detector.py --api-key "key" --image-path "/path" --visualize`
- `python object_sampler.py --video-path "/path" --api-key "key" --num-samples 5`
- `python qa_generator.py --api-key "key" --video-path "/path" --timestamp 30.0`

### Dependencies Installation
```bash
pip install torch torchvision opencv-python pillow pandas numpy scikit-learn openai clip-by-openai
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

## Architecture Components

- **main.py**: `VQAGenerationPipeline` - 主管道类，多线程协调
- **object_sampler.py**: `ObjectSampler` - 核心采样逻辑，CLIP聚类+眼动选择  
- **qa_generator.py**: `QAGenerator` - 工具调用式QA生成
- **mcq_refiner.py**: `MCQRefiner` - 结构化MCQ输出
- **object_detector.py**: `ObjectDetector` - Gemini对象检测
- **utils.py**: 公共工具函数和文件处理