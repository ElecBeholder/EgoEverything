#!/usr/bin/env python3
"""
眼动注视点和手部追踪视频渲染器运行脚本

使用方法:
1. 激活conda环境: conda activate generatedata
2. 运行脚本: python run_gaze_renderer.py

您可以修改下面的参数来自定义处理设置
"""

from gaze_video_renderer import GazeVideoRenderer
import os


def main():
    """主函数"""
    print("=== 眼动注视点和手部追踪视频渲染器 ===")
    
    # ==========================================================================
    # 配置参数 - 您可以根据需要修改这些参数
    # ==========================================================================
    
    # 输入文件路径
    vrs_path = "../Data/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_main_recording.vrs"
    mps_dir = "../Data"  # 包含general_eye_gaze.csv的目录
    
    # 输出文件路径
    output_path = "output_with_gaze_and_hands.mp4"
    
    # 处理参数
    start_frame = 0      # 开始处理的帧数
    max_frames = None     # 最大处理帧数（None表示处理所有帧）
    
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
    crop_size_ratio = 1/2         # 裁剪框大小比例（相对于原视频尺寸）
    
    # ==========================================================================
    
    # 检查输入文件是否存在
    if not os.path.exists(vrs_path):
        print(f"❌ VRS文件不存在: {vrs_path}")
        return
    
    if not os.path.exists(os.path.join(mps_dir, "general_eye_gaze.csv")):
        print(f"❌ 眼动数据文件不存在: {os.path.join(mps_dir, 'general_eye_gaze.csv')}")
        return
    
    try:
        print(f"📁 VRS文件: {vrs_path}")
        print(f"👁  眼动数据目录: {mps_dir}")
        print(f"📤 输出文件: {output_path}")
        print(f"🎯 处理帧数: {start_frame} - {start_frame + max_frames if max_frames else '结尾'}")
        print(f"⚪ 注视点设置: 半径={gaze_radius}px, 颜色={gaze_color}")
        print(f"🖐  手部追踪: {'启用' if enable_hand_tracking else '禁用'}")
        if enable_hand_tracking:
            print(f"   左手: 半径={hand_radius}px, 颜色={left_hand_color}")
            print(f"   右手: 半径={hand_radius}px, 颜色={right_hand_color}")
        print(f"🔲  裁剪窗口视频: {'启用' if enable_crop_video else '禁用'}")
        if enable_crop_video:
            print(f"   裁剪框大小: {crop_size_ratio:.1%} 原视频尺寸")
        print()
        
        # 创建渲染器并处理视频
        renderer = GazeVideoRenderer(vrs_path, mps_dir, output_path)
        renderer.render_video(
            start_frame=start_frame,
            max_frames=max_frames,
            gaze_radius=gaze_radius,
            gaze_color=gaze_color,
            enable_hand_tracking=enable_hand_tracking,
            hand_radius=hand_radius,
            left_hand_color=left_hand_color,
            right_hand_color=right_hand_color,
            enable_crop_video=enable_crop_video,
            crop_size_ratio=crop_size_ratio
        )
        
        print("🎉 处理完成！")
        print(f"📹 输出视频: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 