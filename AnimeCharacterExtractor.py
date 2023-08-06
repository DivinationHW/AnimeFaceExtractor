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

def process_video(VIDEO_PATH, TEMP_FOLDER, OUTPUT_FOLDER):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    frame_id = 0
    skip_counter = 0
    last_hash = None
    last_vector = None

    if USE_FFMPEG == 1:
        if not os.path.exists(TEMP_FOLDER):
            os.makedirs(TEMP_FOLDER)

        command = f'ffmpeg -i "{VIDEO_PATH}" -vf "select=eq(pict_type\\,I)" -vsync vfr -q:v 2 "{TEMP_FOLDER}/keyframe_%03d.png"'
        subprocess.call(command, shell=True)

        keyframes = sorted(os.listdir(TEMP_FOLDER))
        pbar = tqdm(total=len(keyframes), desc='处理帧中')

        for keyframe in keyframes:
            frame = cv2.imread(os.path.join(TEMP_FOLDER, keyframe))
            if frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)

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

                if DRAW_RED_BOX == 1:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), RED_BOX_THICKNESS)

                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'output_frame_{frame_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_id += 1

            pbar.update(1)

        if KEEP_TEMP_FOLDER == 0:
            shutil.rmtree(TEMP_FOLDER)

        pbar.close()

    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f'无法打开视频：{VIDEO_PATH}')
            exit()

        pbar = tqdm(total=total_frames, desc='处理帧中')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray)

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

                if DRAW_RED_BOX == 1:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), RED_BOX_THICKNESS)

                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'output_frame_{frame_id}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_id += 1

            pbar.update(1)

        cap.release()
        pbar.close()

INPUT_PATH = input('请输入视频文件或文件夹的路径（如："demo.mkv"或"videos_folder"，回车使用默认参数："demo.mkv"）：') or 'demo.mkv'
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

KEEP_TEMP_FOLDER = int(input(f'是否在任务执行后保留关键帧临时文件夹（0：否，1：是。回车使用默认参数：0）：') or 0)

timestamp = int(time())
TEMP_ROOT_FOLDER = f"temp_keyframes_{timestamp}"

if os.path.isdir(INPUT_PATH):
    videos = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.endswith(('.mp4', '.mkv', '.avi', '.flv', '.wmv', '.mov', '.mpeg', '.webm'))]
    for video_path in videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        TEMP_FOLDER = os.path.join(TEMP_ROOT_FOLDER, video_name)
        video_output_folder = os.path.join(OUTPUT_FOLDER, video_name)
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        process_video(video_path, TEMP_FOLDER, video_output_folder)
else:
    TEMP_FOLDER = os.path.join(TEMP_ROOT_FOLDER, os.path.splitext(os.path.basename(INPUT_PATH))[0])
    process_video(INPUT_PATH, TEMP_FOLDER, OUTPUT_FOLDER)

if KEEP_TEMP_FOLDER == 0:
    shutil.rmtree(TEMP_ROOT_FOLDER)
