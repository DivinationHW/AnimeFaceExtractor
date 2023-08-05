import cv2
import os
import numpy as np
import subprocess
from tqdm import tqdm
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity

# 请求用户输入配置参数
VIDEO_PATH = input('请输入视频文件的路径（回车使用默认参数："demo.mkv"）：') or 'demo.mkv'
OUTPUT_FOLDER = input('请输入输出文件夹（回车使用默认参数："output"）：') or 'output'
CASCADE_PATH = input('请输入级联分类器的路径（回车使用默认参数："lbpcascade_animeface.xml"）：') or 'lbpcascade_animeface.xml'
JPEG_QUALITY = int(input('请输入输出图像的JPEG质量（0-100，回车使用默认参数：90）：') or 90)
DETECTION_MODE = int(input('请输入人脸检测模式（0：直接在每一帧上进行人脸检测；1：计算余弦相似度，跳过相似的帧；2：使用图像哈希比较，跳过相似的帧。回车使用默认参数：0）：') or 0)

if DETECTION_MODE == 0:
    SKIP_FRAMES = int(input('请输入在检测到人脸后要跳过的帧数（回车使用默认参数：0）：') or 0)
elif DETECTION_MODE == 1:
    COSINE_THRESHOLD = float(input('请为检测模式1输入余弦相似度阈值（回车使用默认参数：0.95）：') or 0.95)
elif DETECTION_MODE == 2:
    HASH_DIFF_THRESHOLD = int(input('请为检测模式2输入哈希差异阈值（回车使用默认参数：5）：') or 5)

# 如果输出文件夹不存在，创建它
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 加载级联分类器
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# 初始化一些变量
frame_id = 0
skip_counter = 0
last_hash = None
last_vector = None

# 计算视频的总帧数
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# 读取视频
cap = cv2.VideoCapture(VIDEO_PATH)

# 检查是否成功打开
if not cap.isOpened():
    print(f'无法打开视频：{VIDEO_PATH}')
    exit()

# 创建一个进度条
pbar = tqdm(total=total_frames, desc='处理帧中')

# 遍历每一帧
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果读取失败，说明已经读到了视频的结尾
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    # 如果检测到人物，根据DETECTION_MODE进行处理
    if len(faces) > 0:
        if DETECTION_MODE == 0:
            if skip_counter > 0:
                skip_counter -= 1
                continue
            skip_counter = SKIP_FRAMES
        elif DETECTION_MODE == 1:
            vector = np.array(frame).flatten()
            if last_vector is not None and cosine_similarity([vector], [last_vector])[0][0] > COSINE_THRESHOLD:
                continue
            last_vector = vector
        elif DETECTION_MODE == 2:
            hash = imagehash.phash(Image.fromarray(frame))
            if last_hash is not None and abs(hash - last_hash) < HASH_DIFF_THRESHOLD:
                continue
            last_hash = hash

        # 保存帧为图片
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'output_frame_{frame_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        frame_id += 1

    # 更新进度条
    pbar.update(1)

# 释放视频
cap.release()

# 关闭进度条
pbar.close()
