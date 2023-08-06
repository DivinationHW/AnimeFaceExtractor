import os
import cv2
from PIL import Image
import imagehash
from tqdm import tqdm
import time

# 用户交互输入参数
def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (默认: {default_value}): ")
    return user_input.strip() if user_input else default_value

# 输入参数
input_path = get_user_input("请输入动漫视频文件或文件夹路径", "demo.mkv")
difference_threshold = int(get_user_input("请输入调整帧差别大小的阈值，范围0~20，回车默认10", "10"))
output_frames_directory = get_user_input("请输入输出帧序列的文件夹名称", "output_frames")
enable_face_detection = get_user_input("是否启用人脸检测? (0=关闭/1=开启)", "0") == "1"

if enable_face_detection:
    print("提醒: 这个人脸检测器不一定准确，（动漫脸检测谁来谁麻）。建议直接查看减少重复帧后的输出图片文件夹。")
    face_recognition_thickness = int(get_user_input("请输入人脸识别红框的粗细，可用10这个粗细程度来检查为啥级联分类器觉得这张有人脸（设置为0表示禁用红框）", "0"))
    output_faces_directory = get_user_input("请输入输出人脸的文件夹名称", "detected_faces")
else:
    face_recognition_thickness = None
    output_faces_directory = None

# 计算两个帧之间的差异哈希
def calculate_frame_difference(frame1, frame2):
    hash1 = imagehash.average_hash(Image.fromarray(frame1))
    hash2 = imagehash.average_hash(Image.fromarray(frame2))
    return hash1 - hash2

# 处理输入路径下的视频文件或文件夹
def compact_video(input_path, threshold=10, face_detection=True, recognition_thickness=2):
    if os.path.isdir(input_path):  # 输入路径是文件夹
        for file in tqdm(os.listdir(input_path), desc="处理文件夹中的视频文件"):
            if file.lower().endswith((".mkv", ".mp4", ".avi")):
                input_file = os.path.join(input_path, file)
                output_frames_dir = os.path.join(output_frames_directory, os.path.splitext(file)[0])
                output_faces_dir = os.path.join(output_faces_directory, os.path.splitext(file)[0] + "_faces")
                compact_video_file(input_file, output_frames_dir, threshold, face_detection, recognition_thickness, output_faces_dir)
    elif os.path.isfile(input_path):  # 输入路径是单个视频文件
        output_frames_dir = os.path.join(output_frames_directory, os.path.splitext(os.path.basename(input_path))[0])
        output_faces_dir = os.path.join(output_faces_directory, os.path.splitext(os.path.basename(input_path))[0] + "_faces")
        compact_video_file(input_path, output_frames_dir, threshold, face_detection, recognition_thickness, output_faces_dir)
    else:
        print("输入路径不是有效的文件或文件夹路径")

# 处理单个视频文件
def compact_video_file(input_file, output_frames_directory, threshold=10, face_detection=True, recognition_thickness=2, output_faces_directory=None):
    if not os.path.isfile(input_file):
        raise FileNotFoundError("输入文件不存在")

    cap = cv2.VideoCapture(input_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, prev_frame = cap.read()
    frame_count = 1

    if not os.path.exists(output_frames_directory):
        os.makedirs(output_frames_directory)

    with tqdm(total=total_frames, desc="处理视频帧") as pbar:
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            if threshold == 0:
                frame_filename = os.path.join(output_frames_directory, f"frame_{frame_count:04d}.png")
                cv2.imwrite(frame_filename, curr_frame)
            else:
                diff = calculate_frame_difference(prev_frame, curr_frame)
                if diff > threshold:
                    frame_filename = os.path.join(output_frames_directory, f"frame_{frame_count:04d}.png")
                    cv2.imwrite(frame_filename, curr_frame)

            prev_frame = curr_frame
            frame_count += 1
            pbar.update(1)

    cap.release()

    if face_detection and output_faces_directory:
        for frame_file in tqdm(os.listdir(output_frames_directory), desc="检测并绘制人脸"):
            full_frame_path = os.path.join(output_frames_directory, frame_file)
            detect_and_draw_faces(full_frame_path, recognition_thickness, output_faces_directory)

# 检测并绘制人脸红框
def detect_and_draw_faces(image_file, recognition_thickness=2, output_faces_directory=None, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(image_file):
        raise FileNotFoundError("输入图片不存在")

    if not os.path.isfile(cascade_file):
        raise RuntimeError("人脸检测器文件不存在")

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    if len(faces) > 0:  # 仅当检测到人脸时保存图像
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), recognition_thickness)

        if output_faces_directory:
            if not os.path.exists(output_faces_directory):
                os.makedirs(output_faces_directory)
            output_file = os.path.join(output_faces_directory, os.path.basename(image_file))
            cv2.imwrite(output_file, image)

if __name__ == "__main__":
    start_time = time.time()
    compact_video(input_path, difference_threshold, enable_face_detection, face_recognition_thickness)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n整个程序的运行时间: {total_time:.2f} 秒")
