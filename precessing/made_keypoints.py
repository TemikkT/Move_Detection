import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp


# === ПУТИ ===
DATA_PATH = "../data/sport_data"
OUTPUT_PATH = "../keypoints_data"

SEQUENCE_LENGTH = 30

mp_pose = mp.solutions.pose

def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(33 * 3)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_pose.Pose() as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)
            keypoints = extract_keypoints(results)

            sequence.append(keypoints)

    cap.release()
    return sequence

# === ГЛАВНЫЙ ЦИКЛ ===

splits = ["train", "val", "test"]

for split in splits:
    split_path = os.path.join(DATA_PATH, split)

    if not os.path.exists(split_path):
        continue

    for action in os.listdir(split_path):
        action_path = os.path.join(split_path, action)

        if not os.path.isdir(action_path):
            continue
        if action not in ["PushUps", "Punch", "WalkingWithDog"]:
            continue

        save_dir = os.path.join(OUTPUT_PATH, split, action)
        os.makedirs(save_dir, exist_ok=True)

        videos = os.listdir(action_path)

        for video in tqdm(videos, desc=f"{split} | {action}"):
            video_path = os.path.join(action_path, video)

            sequence = process_video(video_path)

            # разбиваем на чанки
            for i in range(0, len(sequence) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
                chunk = sequence[i:i + SEQUENCE_LENGTH]

                save_path = os.path.join(
                    save_dir,
                    f"{video}_{i}.npy"
                )

                np.save(save_path, chunk)