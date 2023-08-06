import cv2
import os
import numpy as np
import subprocess
import shutil
from tqdm import tqdm
from PIL import Image
import imagehash
from time import time
from sklearn.metrics.pairwise import cosine_similarity

# 请求用户输入配置参数
VIDEO_PATH = input('请输入视频文件的路径（如："demo.mkv"，回车使用默认参数："demo.mkv"）：') or 'demo.mkv'
OUTPUT_FOLDER = input('请输入输出文件夹路径（如："output"，回车使用默认参数："output"）：') or 'output'
CASCADE_PATH = input('请输入级联分类器的路径（用于人脸检测，回车使用默认参数："lbpcascade_animeface.xml"）：') or 'lbpcascade_animeface.xml'
JPEG_QUALITY = int(input('请输入输出图像的JPEG质量（0-100，数字越大，输出图像质量越高，文件大小也会增加。回车使用默认参数：100）：') or 100)
DETECTION_MODE = int(input('请输入人脸检测模式（0：直接在每一帧上进行人脸检测；1：计算余弦相似度，跳过相似的帧；2：使用图像哈希比较，跳过相似的帧。回车使用默认参数：0）：') or 0)
USE_FFMPEG = int(input('是否使用ffmpeg提取关键帧进行人脸检测（0：否，1：是。使用关键帧可以减少处理时间，若有ffmpeg环境，建议开启，下载地址：https://www.ffmpeg.org/。回车使用默认参数：0）：') or 0)
DRAW_RED_BOX = int(input('是否在检测到的人像上画红色方框（0：否，1：是。若你觉得，啊？这图片哪来的人脸，可以尝试打开此项看看级联分类器是如何识别人脸的。回车使用默认参数：0）：') or 0)
RED_BOX_THICKNESS = int(input('请输入红色方框的粗细（仅当上一项为1时有效，回车使用默认参数：10）：') or 10)

if DETECTION_MODE == 0:
    SKIP_FRAMES = int(input('请输入在检测到人脸后要跳过的帧数（此参数用于在检测到人脸后跳过指定数量的帧，减少处理时间，若开启ffmpeg关键帧抓取，建议使用默认参数。回车使用默认参数：0）：') or 0)
elif DETECTION_MODE == 1:
    COSINE_THRESHOLD = float(input('请为检测模式1输入余弦相似度阈值（此参数用于判断连续帧之间的相似度，如果相似度高于此值，则跳过帧。值越接近1，相似度要求越严格。回车使用默认参数：0.95）：') or 0.95)
elif DETECTION_MODE == 2:
    HASH_DIFF_THRESHOLD = int(input('请为检测模式2输入哈希差异阈值（此参数用于判断连续帧之间的哈希差异，如果差异小于此值，则跳过帧。值越小，相似度要求越严格。回车使用默认参数：5）：') or 5)

# 创建时间戳后缀的临时文件夹名称
timestamp = int(time())
TEMP_FOLDER = f"temp_keyframes_{timestamp}"

# 是否在任务执行后删除关键帧临时文件夹（0：删除，1：保留。保留临时文件夹可以用于分析关键帧提取情况。回车使用默认参数：0）
KEEP_TEMP_FOLDER = int(input(f'是否在任务执行后保留关键帧临时文件夹（0：否，1：是。临时文件夹路径为："{TEMP_FOLDER}"。回车使用默认参数：0）：') or 0)

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

# 是否使用ffmpeg提取关键帧
if USE_FFMPEG == 1:
    # 创建临时文件夹
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    # 使用ffmpeg提取关键帧，保存为png格式，文件名以关键帧的时间戳命名
    command = f'ffmpeg -i "{VIDEO_PATH}" -vf "select=eq(pict_type\\,I)" -vsync vfr -q:v 2 "{TEMP_FOLDER}/keyframe_%03d.png"'
    subprocess.call(command, shell=True)

    # 获取关键帧文件列表
    keyframes = sorted(os.listdir(TEMP_FOLDER))

    # 创建一个进度条
    pbar = tqdm(total=len(keyframes), desc='处理帧中')

    # 遍历每一个关键帧
    for keyframe in keyframes:
        # 读取关键帧
        frame = cv2.imread(os.path.join(TEMP_FOLDER, keyframe))

        # 如果读取失败，说明已经读到了视频的结尾
        if frame is None:
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

            # 在图像上为每个检测到的人像添加红色的边框
            if DRAW_RED_BOX == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), RED_BOX_THICKNESS)

            # 保存帧为图片
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'output_frame_{frame_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            frame_id += 1

        # 更新进度条
        pbar.update(1)

    # 是否删除临时文件夹
    if KEEP_TEMP_FOLDER == 0:
        shutil.rmtree(TEMP_FOLDER)

    # 关闭进度条
    pbar.close()

else:
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

            # 在图像上为每个检测到的人像添加红色的边框
            if DRAW_RED_BOX == 1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), RED_BOX_THICKNESS)

            # 保存帧为图片
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'output_frame_{frame_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            frame_id += 1

        # 更新进度条
        pbar.update(1)

    # 释放视频
    cap.release()

    # 关闭进度条
    pbar.close()
