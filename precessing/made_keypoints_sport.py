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
    """Извлечение ключевых точек позы (33 точки x 3 координаты)"""
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(33 * 3)

def process_video(video_path):
    """Обработка одного видео: возвращает список кадров с ключевыми точками"""
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

# === ОСНОВНОЙ ЦИКЛ ===
# Перебираем все классы (поддиректории) в DATA_PATH
for action in os.listdir(DATA_PATH):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        continue   # пропускаем файлы, если они есть

    # Создаём выходную директорию для этого класса
    save_dir = os.path.join(OUTPUT_PATH, action)
    os.makedirs(save_dir, exist_ok=True)

    # Получаем список видео в директории класса
    videos = [f for f in os.listdir(action_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video in tqdm(videos, desc=f"Класс: {action}"):
        video_path = os.path.join(action_path, video)
        sequence = process_video(video_path)

        # Нарезаем на чанки длины SEQUENCE_LENGTH
        for i in range(0, len(sequence) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            chunk = sequence[i:i + SEQUENCE_LENGTH]
            # Сохраняем чанк как .npy файл
            save_path = os.path.join(save_dir, f"{os.path.splitext(video)[0]}_{i}.npy")
            np.save(save_path, chunk)

print("Готово! Ключевые точки сохранены в:", OUTPUT_PATH)