# !rm -r /home/aistudio/work/output/L/frames/color_frames/*
# !rm -r /home/aistudio/work/output/L/frames/depth_frames/*
# !rm -r /home/aistudio/work/output/F/frames/color_frames/*
# !rm -r /home/aistudio/work/output/F/frames/depth_frames/*

import cv2
import os

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建之。"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# 基础路径
base_output_path = '/home/aistudio/work/output/'
base_color_path = '/home/aistudio/work/{type}/color/'
base_depth_path = '/home/aistudio/work/{type}/depth/'

# 获取用户输入的后缀
input_suffix = input("Please enter the last digit of the prefix: ")

# 分类处理的类型
types = ['F', 'L']

# 存储路径信息
paths_info = {}

for t in types:
    # 生成前缀、路径等
    prefix = f'color_frame-{t}-01-a-0' + input_suffix
    color_path = base_color_path.format(type=t)
    depth_path = base_depth_path.format(type=t)
    out_path = os.path.join(base_output_path, f'{t}/')

    # 列出目录中所有文件，并筛选出符合前缀的文件名
    files = os.listdir(color_path)
    file_names = [file for file in files if file.startswith(prefix)]
    
    # 确保输出目录存在
    color_frames_dir = os.path.join(out_path, 'frames/color_frames/')
    depth_frames_dir = os.path.join(out_path, 'frames/depth_frames/')
    ensure_dir_exists(color_frames_dir)
    ensure_dir_exists(depth_frames_dir)
    ensure_dir_exists(out_path)
    
    # 这里我们只处理颜色视频路径的示例，深度路径处理类似
    # 假设每个文件名对应一个视频帧，生成视频路径需要具体的文件处理逻辑
    color_video_paths = [os.path.join(color_path, file_name) for file_name in file_names]
    depth_video_paths = [os.path.join(depth_path, file_name.replace('color', 'depth')) for file_name in file_names]  # 假设深度文件名只是前缀不同

    # 存储路径信息以便后续使用
    paths_info[t] = {
        'color_frames_dir': color_frames_dir,
        'depth_frames_dir': depth_frames_dir,
        'color_video_paths': color_video_paths,
        'depth_video_paths': depth_video_paths,
    }

# # 这里只演示如何访问和使用存储的路径信息
# for t, info in paths_info.items():
#     print(f"Processing {t} files...")
#     # 示例：打印出每种类型的颜色视频路径
#     for path in info['color_video_paths']:
#         print(f"Color video path for {t}: {path}")
def extract_frames(color_video_path, depth_video_path, color_frames_dir, depth_frames_dir):
    """
    从两个视频中提取每一帧，并保存到对应的目录。
    """
    color_cap = cv2.VideoCapture(color_video_path)
    depth_cap = cv2.VideoCapture(depth_video_path)
    frame_count = 0

    while True:
        ret_color, color_frame = color_cap.read()
        ret_depth, depth_frame = depth_cap.read()
        
        if not ret_color or not ret_depth:
            break

        color_frame_path = os.path.join(color_frames_dir, f'frame_{frame_count}.jpg')
        depth_frame_path = os.path.join(depth_frames_dir, f'frame_{frame_count}.jpg')
        
        cv2.imwrite(color_frame_path, color_frame)
        cv2.imwrite(depth_frame_path, depth_frame)
        
        frame_count += 1

    color_cap.release()
    depth_cap.release()
    return frame_count

def extract_frames_from_paths(color_video_paths, depth_video_paths, color_frames_dir, depth_frames_dir):
    """
    根据多个视频文件路径提取帧，并保存到指定目录。
    """
    total_frame_count = 0
    for color_video_path, depth_video_path in zip(color_video_paths, depth_video_paths):
        # 对于每对颜色和深度视频路径，调用 extract_frames 函数
        frame_count = extract_frames(color_video_path, depth_video_path, color_frames_dir, depth_frames_dir)
        total_frame_count += frame_count
    return total_frame_count

# 对每种类型（F和L）调用上述函数
for t, info in paths_info.items():
    print(f"Processing {t} files...")
    color_video_paths = info['color_video_paths']
    depth_video_paths = info['depth_video_paths']
    color_frames_dir = info['color_frames_dir']
    depth_frames_dir = info['depth_frames_dir']
    
    # 确保目标目录存在
    ensure_dir_exists(color_frames_dir)
    ensure_dir_exists(depth_frames_dir)
    
    # 提取帧
    total_frames = extract_frames_from_paths(color_video_paths, depth_video_paths, color_frames_dir, depth_frames_dir)
    print(f"Total frames extracted for {t}: {total_frames}")

import os

# 设定四种帧目录的基础路径
base_dirs = {
    'F': {
        'color': 'work/output/F/frames/color_frames/',
        'depth': 'work/output/F/frames/depth_frames',
    },
    'L': {
        'color': 'work/output/L/frames/color_frames',
        'depth': 'work/output/L/frames/depth_frames',
    }
}

# 函数：获取并排序指定目录下的帧文件路径
def get_sorted_frame_paths(frame_dir):
    frame_files = os.listdir(frame_dir)
    sorted_files = sorted(
        [file for file in frame_files if file.startswith('frame_') and file.split('_')[1].split('.')[0].isdigit()],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    return [os.path.join(frame_dir, filename) for filename in sorted_files]

# # 循环遍历每种类型的颜色和深度帧目录，获取并排序帧文件路径
# for type_key, dirs in base_dirs.items():
#     color_frame_paths = get_sorted_frame_paths(dirs['color'])
#     depth_frame_paths = get_sorted_frame_paths(dirs['depth'])

#     print(f'Total color frames for {type_key}: {len(color_frame_paths)}')
#     print(f'Total depth frames for {type_key}: {len(depth_frame_paths)}')
#     # 打印前几个路径查看
#     print(f"Color frame paths for {type_key}:")
#     print(color_frame_paths[:5])  # 打印前5个路径作为示例
#     print(f"Depth frame paths for {type_key}:")
#     print(depth_frame_paths[:5])  # 打印前5个路径作为示例

sorted_frame_paths_info = {}
for type_key, dirs in base_dirs.items():
    color_frame_paths = get_sorted_frame_paths(dirs['color'])
    depth_frame_paths = get_sorted_frame_paths(dirs['depth'])
    sorted_frame_paths_info[type_key] = {
        'color': color_frame_paths,
        'depth': depth_frame_paths,
    }

    print(f'Total color frames for {type_key}: {len(color_frame_paths)}')
    print(f'Total depth frames for {type_key}: {len(depth_frame_paths)}')
    # 打印前几个路径查看
    print(f"Color frame paths for {type_key}:")
    print(color_frame_paths[:5])  # 打印前5个路径作为示例
    print(f"Depth frame paths for {type_key}:")
    print(depth_frame_paths[:5])  # 打印前5个路径作为示例

F_color_frame_paths = sorted_frame_paths_info['F']['color']
L_color_frame_paths = sorted_frame_paths_info['L']['color']
print(L_color_frame_paths[:5])

## import cv2
from matplotlib import pyplot as plt

def select_start_end_frame(frame_paths):
    """
    正序浏览帧并选择开始帧，然后逆序浏览帧并选择结束帧。
    """
    # 选择开始帧
    start_frame_index = select_start_frame(frame_paths)
    
    # 选择结束帧
    end_frame_index = select_end_frame(frame_paths)
    
    return start_frame_index, end_frame_index

def select_start_frame(frame_paths):
    """
    正序浏览帧并选择开始帧。

    :param frame_paths: 帧的路径列表
    """
    # 获取用户输入并转换为整数
    try:
        start_frame_index = int(input("Enter the starting frame index: "))
    except ValueError:
        print("Invalid input. Starting from the first frame.")
        start_frame_index = 0

    # 确保输入的索引在有效范围内
    if start_frame_index < 0 or start_frame_index >= len(frame_paths):
        print(f"Input is out of bounds. Must be between 0 and {len(frame_paths) - 1}. Starting from the first frame.")
        start_frame_index = 0

    for index, frame_path in enumerate(frame_paths[start_frame_index:], start=start_frame_index):
        frame = cv2.imread(frame_path)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        command = input(f"Frame {index + 1}/{len(frame_paths)}. Press Enter to continue, 'q' to select this frame and quit: ")
        if command == 'q':
            return frame_path

def select_end_frame(frame_paths):
    """
    逆序显示帧，按 'q' 选择结束帧
    """
    for index, frame_path in enumerate(reversed(frame_paths)):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load image at {frame_path}")
            continue

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)

        real_index = len(frame_paths) - index - 1  # 计算实际的帧索引
        key = input(f"Frame {real_index + 1}/{len(frame_paths)}. Press Enter to continue or 'q' to select this frame: ")
        plt.close()

        if key == 'q':
            return frame_path

# 示例路径
# frame_paths   # 这里只是示例路径，请替换为您真实的路径列表

# 选择开始和结束帧
F_start_frame_path = select_start_frame(F_color_frame_paths)
F_end_frame_path = select_end_frame(F_color_frame_paths)
# L_start_frame_path = select_start_frame(L_color_frame_paths)
# L_end_frame_path = select_start_frame(L_color_frame_paths)

print(f"Selected start frame: {F_start_frame_path}")
print(f"Selected end frame: {F_end_frame_path}")

# 假设 start_frame_path 和 end_frame_path 是您已经有的选择
F_start_frame_index = int(input("Enter the new start frame index: "))
F_end_frame_index = int(input("Enter the new end frame index: "))

# 确保输入的索引在有效范围内
if 0 <= start_frame_index < len(F_color_frame_paths) and 0 <= end_frame_index < len(F_color_frame_paths):
    start_frame_path = F_color_frame_paths[start_frame_index]
    end_frame_path = F_color_frame_paths[end_frame_index]
    print(f"F Start frame set to: {F_start_frame_path}")
    print(f"F End frame set to: {F_end_frame_path}")
else:
    print("Error: Entered index is out of range.")

# 计算开始和结束帧之间的帧数，包括这两帧
F_total_frames = end_frame_index - start_frame_index + 1

print(f"F Total frames between start and end: {F_total_frames}")

def get_new_start_frame_index(start_frame_path, total_frames):
    # 提取帧编号，假设路径格式为 '.../frame_X.jpg'，其中X是帧编号
    frame_number = int(start_frame_path.split('_')[-1].split('.')[0])
    
    # 计算新的帧编号
    new_frame_index = frame_number - total_frames
    
    return new_frame_index
def select_L_start_frame(frame_paths, start_frame, total_frames):
    """
    正序浏览帧并选择开始帧。

    :param frame_paths: 帧的路径列表
    :param start_frame: 用户定义的开始帧
    :param total_frames: 用户定义的帧总数
    """
    
    start_frame_index = get_new_start_frame_index(start_frame, total_frames)

    # 将 start_frame 和 total_frames 相加来确定新的开始索引
    #start_frame_index = start_frame + total_frames

    # 确保输入的索引在有效范围内
    if start_frame_index < 0 or start_frame_index >= len(frame_paths):
        print(f"index is out of bounds. Must be between 0 and {len(frame_paths) - 1}. Starting from the last frame.")
        return select_start_frame(frame_paths)

    for index, frame_path in enumerate(frame_paths[start_frame_index:], start=start_frame_index):
        frame = cv2.imread(frame_path)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        command = input(f"Frame {index + 1}/{len(frame_paths)}. Press Enter to continue, 'q' to select this frame and quit: ")
        if command == 'q':
            return frame_path

def get_new_frame_index(start_frame_path, total_frames):
    # 提取帧编号，假设路径格式为 '.../frame_X.jpg'，其中X是帧编号
    frame_number = int(start_frame_path.split('_')[-1].split('.')[0])
    
    # 计算新的帧编号
    new_frame_index = frame_number + total_frames
    
    return new_frame_index
def select_L_end_frame(frame_paths, start_frame, total_frames):
    """
    正序浏览帧并选择开始帧。

    :param frame_paths: 帧的路径列表
    :param start_frame: 用户定义的开始帧
    :param total_frames: 用户定义的帧总数
    """
    global condition
    start_frame_index = get_new_frame_index(start_frame, total_frames)

    # 将 start_frame 和 total_frames 相加来确定新的开始索引
    #start_frame_index = start_frame + total_frames

    # 确保输入的索引在有效范围内
    if start_frame_index < 0 or start_frame_index >= len(frame_paths):
        print(f"index is out of bounds. Must be between 0 and {len(frame_paths) - 1}. Starting from the last frame.")
        condition = True
        return select_end_frame(frame_paths)

    for index, frame_path in enumerate(frame_paths[start_frame_index:], start=start_frame_index):
        frame = cv2.imread(frame_path)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        command = input(f"Frame {index + 1}/{len(frame_paths)}. Press Enter to continue, 'q' to select this frame and quit: ")
        if command == 'q':
            return frame_path
L_start_frame_path = select_start_frame(L_color_frame_paths)
L_end_frame_path = select_L_end_frame(L_color_frame_paths, L_start_frame_path, F_total_frames)
if condition:
    print('index is out of bounds.Please reselect the start frames.')
    L_start_frame_path = select_L_start_frame(L_color_frame_paths,L_end_frame_path,F_total_frames)


print(f"Selected L start frame: {L_start_frame_path}")
print(f"Selected L end frame: {L_end_frame_path}")

def get_frame_index(frame_path):
    # 提取帧编号，假设路径格式为 '.../frame_X.jpg'，其中X是帧编号
    frame_number = int(frame_path.split('_')[-1].split('.')[0])
    return frame_number
L_total_frames = get_frame_index(L_end_frame_path) - get_frame_index(L_start_frame_path) + 1
print(f"Selected F start frame: {F_start_frame_path}")
print(f"Selected F end frame: {F_end_frame_path}")
print(f"F Total frames between start and end: {F_total_frames}")
print(f"L Total frames between start and end: {L_total_frames}")
