import os
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import mediapipe as mp


DEBUG = input('Запустить малую версию для проверки гипотез (Yes/No): ')
#пУТИ 
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "UFC101"

if DEBUG == 'No':
    OUTPUT_PATH = BASE_DIR / "keypoints_data" / "mediapipe"
else:
    OUTPUT_PATH = BASE_DIR / "keypoints_data" / "DEBUG" / "mediapipe"

SEQUENCE_LENGTH = 30

FRAME_SKIP = 2

mp_pose = mp.solutions.pose

def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ])

        #Нормазилация позиции
        left_hip = keypoints[23]
        right_hip = keypoints[24]

        hip_center = (
            left_hip + right_hip
        ) / 2

        keypoints = keypoints - hip_center
        return keypoints.flatten()

    return np.zeros(33 * 3)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1

            if frame_idx % FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (224, 224))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    cap.release()
    
    return sequence

#Только для проверки гипотез
classes = ['Archery', 'BenchPress', 'Biking', 
           'PlayingGuitar', 'PlayingPiano', 'LongJump', 
           'Mixing', 'PizzaTossing', 'PlayingDaf', 'CliffDiving']
splits = ["train", "val", "test"]

for split in splits:
    csv_path = DATA_PATH / f"{split}.csv"

    if not csv_path.exists():
        print(f"CSV не найден: {csv_path}")
        continue

    df = pd.read_csv(csv_path, sep=',')
    metadata = []

    if DEBUG == 'Yes':
        df = df[df['label'].isin(classes)].copy()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):

        clip_name = row["clip_name"]
        clip_path = row["clip_path"].lstrip("/\\")
        video_path = DATA_PATH / clip_path
        label = row["label"]

    
        if not video_path.exists():
            print(f"\nФайл не найден: {video_path}")
            continue

        save_dir = os.path.join(
            OUTPUT_PATH,
            split,
            label
        )

        os.makedirs(save_dir, exist_ok=True)

        # Process video

        sequence = process_video(video_path)
        if len(sequence) == 0:
            print(f"\nПустое видео: {video_path}")
            continue

        for i in range(0, len(sequence), SEQUENCE_LENGTH):

            chunk = sequence[i:i + SEQUENCE_LENGTH]

            # если chunk меньше нужной длины дополняем нулями
            while len(chunk) < SEQUENCE_LENGTH:
                chunk.append(np.zeros(33 * 3))

            chunk = np.array(chunk)

            zero_frames = np.sum(
                np.all(chunk == 0, axis=1)
            )

            if zero_frames > 20:
                continue

            # проверка shape
            if chunk.shape != (SEQUENCE_LENGTH, 99):
                print(f"\nОшибка shape: {chunk.shape}")
                continue

            # проверка NaN
            if np.isnan(chunk).any():
                print(f"\nNaN найден: {video_path}")
                continue

            # имя файла
            video_name = os.path.splitext(clip_name)[0]
            save_name = f"{video_name}_chunk{i}.npy"
            save_path = os.path.join(save_dir, save_name)


            np.save(save_path, chunk)

            # metadata
            metadata.append({
                "npy_path": save_path,
                "label": label,
                "split": split
            })
    
    metadata_df = pd.DataFrame(metadata)

    metadata_save_path = os.path.join(
        OUTPUT_PATH,
        f"{split}_metadata.csv"
    )

    metadata_df.to_csv(metadata_save_path, index=False)
    print(f"\nMetadata сохранен: {metadata_save_path}")
print("\nDONE")