import cv2
import os
import re
import shutil

# 定义源目录
source_dir = '/F/video/color'

# 定义目标根目录
target_root_dir = '/path/to/target'

# 创建一个正则表达式来匹配和提取文件名中的信息
pattern = re.compile(r'out_color_frame-([FL])-(\d{2})-([a-e])-(\d{2})-.*\.mp4')

def process_video_to_frames(video_path, target_dir):
    """
    将视频文件处理成帧，并保存到目标目录
    """
    # 使用OpenCV读取视频
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(target_dir, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_number += 1
    cap.release()

# 遍历源目录中的所有文件
for filename in os.listdir(source_dir):
    match = pattern.match(filename)
    if match:
        # 提取视角、动作编号、角色和遍数
        view, action, character, iteration = match.groups()

        # 构建目标目录路径
        target_dir = os.path.join(target_root_dir, action, f'{character}-{iteration}', view)
        os.makedirs(target_dir, exist_ok=True)

        # 构建源文件的完整路径
        src_file = os.path.join(source_dir, filename)

        # 处理视频文件，将其转换为帧并保存到目标目录
        process_video_to_frames(src_file, target_dir)
        print(f'Processed and saved frames from: {src_file} to {target_dir}')

print('All videos have been processed.')

#测试更改