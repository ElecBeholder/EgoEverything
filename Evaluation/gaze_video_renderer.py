import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection


class GazeVideoRenderer:
    """眼动注视点和手部追踪视频渲染器"""
    
    def __init__(self, vrs_path, mps_dir, output_path):
        """
        初始化渲染器
        
        Args:
            vrs_path: VRS文件路径（包含眼动数据和RGB视频）
            mps_dir: MPS数据目录路径
            output_path: 输出视频路径
        """
        self.vrs_path = Path(vrs_path)
        self.mps_dir = Path(mps_dir)
        self.output_path = output_path
        
        # 注视点绘制参数
        self.gaze_radius = 15
        self.gaze_color = (0, 0, 255)  # BGR格式：红色
        
        # 手部追踪参数
        self.hand_radius = 15
        self.left_hand_color = (0, 255, 0)   # BGR格式：绿色
        self.right_hand_color = (255, 0, 0)  # BGR格式：蓝色
        
        # 初始化数据提供器
        self._init_data_providers()
        self._init_mediapipe()
    
    def _init_data_providers(self):
        """初始化数据提供器"""
        # 初始化VRS数据提供器
        self.vrs_provider = data_provider.create_vrs_data_provider(str(self.vrs_path))
        if not self.vrs_provider:
            raise ValueError(f"无法打开VRS文件: {self.vrs_path}")
        
        # 获取设备标定信息
        self.device_calib = self.vrs_provider.get_device_calibration()
        
        # RGB相机配置
        self.rgb_stream = StreamId("214-1")
        self.rgb_stream_label = "camera-rgb"
        self.rgb_calib = self.device_calib.get_camera_calib(self.rgb_stream_label)
        
        # 初始化MPS数据提供器（眼动数据）
        paths_provider = MpsDataPathsProvider(str(self.mps_dir))
        mps_paths = paths_provider.get_data_paths()
        self.mps = MpsDataProvider(mps_paths)
        
        if not self.mps.has_general_eyegaze():
            raise ValueError("没有找到眼动数据！")
    
    def _init_mediapipe(self):
        """初始化MediaPipe手部追踪"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1
        )
    
    def _get_hand_centers(self, frame):
        """
        获取手部中心位置
        
        Returns:
            tuple: (left_hand_center, right_hand_center) 或 (None, None)
        """
        # 转换为RGB格式供MediaPipe使用
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        left_hand_center = None
        right_hand_center = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 计算手部中心（所有关键点的平均位置）
                h, w = frame.shape[:2]
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # 判断左右手
                hand_label = handedness.classification[0].label
                if hand_label == 'Left':  # MediaPipe的Left是用户的右手（镜像）
                    right_hand_center = (center_x, center_y)
                else:  # Right是用户的左手
                    left_hand_center = (center_x, center_y)
        
        return left_hand_center, right_hand_center
    
    def _create_crop_windows_frame(self, frame, gaze_point, left_hand_center, right_hand_center, 
                                   crop_width, crop_height):
        """
        创建裁剪窗口帧，在三个追踪点周围保留原图像，其他区域用黑色填充
        
        Args:
            frame: 原始帧
            gaze_point: 注视点坐标 (x, y) 或 None
            left_hand_center: 左手中心坐标 (x, y) 或 None  
            right_hand_center: 右手中心坐标 (x, y) 或 None
            crop_width: 裁剪窗口宽度
            crop_height: 裁剪窗口高度
            
        Returns:
            裁剪窗口帧
        """
        h, w = frame.shape[:2]
        # 创建黑色遮罩
        crop_frame = np.zeros_like(frame)
        
        # 收集所有有效的追踪点
        points = []
        if gaze_point is not None:
            points.append(gaze_point)
        if left_hand_center is not None:
            points.append(left_hand_center)
        if right_hand_center is not None:
            points.append(right_hand_center)
        
        # 为每个追踪点创建裁剪窗口
        for point in points:
            cx, cy = point
            
            # 计算裁剪窗口边界，确保不超出图像范围
            x1 = max(0, cx - crop_width // 2)
            y1 = max(0, cy - crop_height // 2)
            x2 = min(w, cx + crop_width // 2)
            y2 = min(h, cy + crop_height // 2)
            
            # 将原图像的对应区域复制到裁剪帧
            crop_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        
        return crop_frame
    
    def _get_gaze_point(self, timestamp_ns):
        """获取注视点坐标"""
        gaze = self.mps.get_general_eyegaze(timestamp_ns)
        if gaze is None:
            return None
        
        depth_m = gaze.depth if hasattr(gaze, 'depth') and gaze.depth else 1.0
        
        projection = get_gaze_vector_reprojection(
            gaze, self.rgb_stream_label, self.device_calib, self.rgb_calib, depth_m
        )
        
        if projection is not None:
            return (int(projection[0]), int(projection[1]))
        return None
    
    def render_video(self, start_frame=0, max_frames=None, gaze_radius=15, gaze_color=(0, 0, 255), 
                     enable_hand_tracking=True, hand_radius=15, left_hand_color=(0, 255, 0), right_hand_color=(255, 0, 0),
                     enable_crop_video=True, crop_size_ratio=1/3):
        """
        渲染注视点和手部追踪到视频
        
        Args:
            start_frame: 开始帧数
            max_frames: 最大处理帧数
            gaze_radius: 注视点半径
            gaze_color: 注视点颜色 (B, G, R)
            enable_hand_tracking: 是否启用手部追踪
            hand_radius: 手部中心点半径
            left_hand_color: 左手颜色 (B, G, R)
            right_hand_color: 右手颜色 (B, G, R)
            enable_crop_video: 是否生成裁剪视频
            crop_size_ratio: 裁剪框大小比例（相对于原视频尺寸）
        """
        self.gaze_radius = gaze_radius
        self.gaze_color = gaze_color
        self.hand_radius = hand_radius
        self.left_hand_color = left_hand_color
        self.right_hand_color = right_hand_color
        
        print("开始处理视频...")
        
        # 获取帧数和配置
        total_frames = self.vrs_provider.get_num_data(self.rgb_stream)
        end_frame = min(start_frame + max_frames, total_frames) if max_frames else total_frames
        process_frames = end_frame - start_frame
        
        config = self.vrs_provider.get_image_configuration(self.rgb_stream)
        width, height = config.image_width, config.image_height
        
        print(f"处理第{start_frame}-{end_frame-1}帧，共{process_frames}帧")
        print(f"图像尺寸: {width}x{height}")
        
        # 创建视频写入器（旋转90度后宽高互换）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, 20, (height, width))
        
        # 创建裁剪视频写入器
        crop_out = None
        crop_output_path = self.output_path.replace('.mp4', '_crop_windows.mp4')
        if enable_crop_video:
            crop_out = cv2.VideoWriter(crop_output_path, fourcc, 20, (height, width))
            print(f"将同时生成裁剪窗口视频: {crop_output_path}")
        
        # 计算裁剪框尺寸
        crop_width = int(width * crop_size_ratio)
        crop_height = int(height * crop_size_ratio)
        
        # 处理每一帧
        for i in tqdm(range(process_frames), desc="渲染视频"):
            frame_idx = start_frame + i
            
            # 获取RGB图像和时间戳
            image_data = self.vrs_provider.get_image_data_by_index(self.rgb_stream, frame_idx)
            if not image_data:
                continue
                
            frame = image_data[0].to_numpy_array()
            timestamp_ns = image_data[1].capture_timestamp_ns
            
            # 修复色彩空间：RGB -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 获取追踪点坐标
            gaze_point = self._get_gaze_point(timestamp_ns)
            left_hand_center = None
            right_hand_center = None
            
            if enable_hand_tracking:
                left_hand_center, right_hand_center = self._get_hand_centers(frame)
            
            # 生成裁剪窗口视频帧（在绘制追踪点之前）
            crop_frame = None
            if enable_crop_video:
                crop_frame = self._create_crop_windows_frame(
                    frame.copy(), gaze_point, left_hand_center, right_hand_center, 
                    crop_width, crop_height
                )
            
            # 在主视频帧上绘制追踪点
            if gaze_point is not None:
                x, y = gaze_point
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), self.gaze_radius, self.gaze_color, -1)
            
            if enable_hand_tracking:
                # 绘制左手中心（绿色）
                if left_hand_center is not None:
                    x, y = left_hand_center
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), self.hand_radius, self.left_hand_color, -1)
                
                # 绘制右手中心（蓝色）
                if right_hand_center is not None:
                    x, y = right_hand_center
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), self.hand_radius, self.right_hand_color, -1)
            
            # 旋转90度纠正方向
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            out.write(frame)
            
            # 处理并输出裁剪视频帧
            if enable_crop_video and crop_frame is not None:
                crop_frame = cv2.rotate(crop_frame, cv2.ROTATE_90_CLOCKWISE)
                crop_out.write(crop_frame)
        
        out.release()
        if crop_out is not None:
            crop_out.release()
            print(f"✅ 处理完成，输出文件:")
            print(f"   主视频（含追踪点）: {self.output_path}")
            print(f"   裁剪窗口视频: {crop_output_path}")
        else:
            print(f"✅ 处理完成，输出: {self.output_path}")


def main():
    """主函数"""
    vrs_path = "../Data/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_main_recording.vrs"
    mps_dir = "../Data"
    output_path = "output_with_gaze_and_hands.mp4"
    
    try:
        renderer = GazeVideoRenderer(vrs_path, mps_dir, output_path)
        renderer.render_video(start_frame=0, max_frames=100, enable_hand_tracking=True)
    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    main() 