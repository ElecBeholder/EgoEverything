# 眼动注视点和手部追踪视频渲染器

这个工具可以将眼动追踪数据中的注视点和MediaPipe检测的手部中心位置渲染到第一人称视频上。

## 功能特点

- 直接从VRS文件中提取RGB视频帧和眼动数据
- 使用MediaPipe模型进行实时手部追踪
- 自动修复色彩空间问题（RGB转BGR）
- 将注视点准确投影到视频帧上
- 标记左右手的中心位置（不同颜色区分）
- 支持自定义注视点和手部标记的大小、颜色等参数
- 包含实时进度条显示处理进度
- 代码简洁高效，专注核心功能

## 标记说明

- **红色圆点**: 眼动注视点
- **绿色圆点**: 左手中心位置
- **蓝色圆点**: 右手中心位置

## 文件说明

- `gaze_video_renderer.py`: 主要的渲染器类（支持眼动和手部追踪）
- `run_gaze_renderer.py`: 简单易用的运行脚本
- `README_gaze_renderer.md`: 使用说明（本文件）

## 依赖要求

确保安装了以下依赖：

```bash
# 安装MediaPipe
pip install mediapipe

# 其他依赖通常已包含在conda环境中
# opencv-python, numpy, tqdm, projectaria_tools
```

## 使用方法

### 1. 激活conda环境

```bash
conda activate generatedata
```

### 2. 进入Evaluation目录

```bash
cd Evaluation
```

### 3. 运行脚本

```bash
python run_gaze_renderer.py
```

## 自定义参数

您可以修改 `run_gaze_renderer.py` 中的参数来自定义处理：

```python
# 处理参数
start_frame = 0      # 开始处理的帧数
max_frames = 300     # 最大处理帧数（None表示处理所有帧）

# 注视点显示参数
gaze_radius = 15              # 注视点圆圈半径（像素）
gaze_color = (0, 0, 255)      # 注视点颜色 (B, G, R) - 红色

# 手部追踪参数
enable_hand_tracking = True   # 是否启用手部追踪
hand_radius = 15              # 手部中心点半径（像素）
left_hand_color = (0, 255, 0) # 左手颜色 (B, G, R) - 绿色
right_hand_color = (255, 0, 0) # 右手颜色 (B, G, R) - 蓝色

# 裁剪窗口视频参数
enable_crop_video = True      # 是否生成裁剪窗口视频
crop_size_ratio = 1/3         # 裁剪框大小比例（相对于原视频尺寸）
```

### 颜色说明（BGR格式）
- `(0, 0, 255)`: 红色
- `(0, 255, 0)`: 绿色  
- `(255, 0, 0)`: 蓝色
- `(0, 255, 255)`: 黄色
- `(255, 0, 255)`: 紫色

## 高级用法

直接使用 `GazeVideoRenderer` 类：

```python
from gaze_video_renderer import GazeVideoRenderer

# 创建渲染器
renderer = GazeVideoRenderer(
    vrs_path="path/to/recording.vrs",
    mps_dir="path/to/mps/data",
    output_path="output.mp4"
)

# 渲染视频（包含眼动和手部追踪 + 裁剪窗口）
renderer.render_video(
    start_frame=0,
    max_frames=100,
    gaze_radius=15,
    gaze_color=(0, 0, 255),      # 红色注视点
    enable_hand_tracking=True,
    hand_radius=15,
    left_hand_color=(0, 255, 0), # 绿色左手
    right_hand_color=(255, 0, 0), # 蓝色右手
    enable_crop_video=True,       # 生成裁剪窗口视频
    crop_size_ratio=1/3           # 裁剪框为原尺寸的1/3
)

# 只使用眼动追踪，不使用手部追踪，禁用裁剪视频
renderer.render_video(
    start_frame=0,
    max_frames=100,
    enable_hand_tracking=False,
    enable_crop_video=False
)

# 自定义裁剪窗口大小（1/2原尺寸）
renderer.render_video(
    start_frame=0,
    max_frames=100,
    crop_size_ratio=1/2          # 更大的裁剪窗口
)
```

## 输入文件要求

- **VRS文件**: 包含眼动追踪数据和RGB视频的原始记录文件
- **MPS数据**: 包含 `general_eye_gaze.csv` 的目录

## 输出

程序默认会生成两个MP4视频文件：

### 1. 主视频（含追踪点标记）
包含：
- 红色圆点标记的眼动注视点
- 绿色圆点标记的左手中心位置
- 蓝色圆点标记的右手中心位置

### 2. 裁剪窗口视频（_crop_windows.mp4）
特点：
- 以注视点、左手、右手为中心的三个裁剪窗口
- 窗口内显示原始视频内容
- 窗口外区域用黑色填充
- 不包含追踪点标记
- 窗口大小可调整（默认为原视频尺寸的1/3）

## 注意事项

1. 程序直接从VRS文件中提取RGB视频帧，确保坐标系统一致
2. 自动处理色彩空间转换（RGB→BGR），修复颜色显示问题
3. 输出视频会自动旋转90度以纠正Aria设备的原始方向
4. MediaPipe手部追踪可能在某些帧中检测不到手部，这是正常现象
5. 手部追踪会增加处理时间，可以通过设置 `enable_hand_tracking=False` 来禁用

## 故障排除

### 常见错误

1. **"无法打开VRS文件"**: 检查文件路径和文件完整性
2. **"没有找到眼动数据"**: 确保general_eye_gaze.csv文件存在
3. **"ModuleNotFoundError: No module named 'mediapipe'"**: 运行 `pip install mediapipe`
4. **"颜色显示不正常"**: 已修复色彩空间问题，现在应该正常显示

### 性能优化

- 代码已精简优化，但手部追踪会增加处理时间
- 可以通过禁用手部追踪来提高处理速度
- 对于测试，建议先处理100-300帧
- 确保有足够的磁盘空间存储输出视频

### MediaPipe相关

- MediaPipe的检测置信度可以在代码中调整
- 手部检测在复杂背景或光线不足时可能不稳定
- 左右手的判断基于MediaPipe的标准，可能与直觉相反（镜像效应） 