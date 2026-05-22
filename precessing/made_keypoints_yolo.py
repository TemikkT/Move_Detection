import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO


DEBUG = input('Запустить малую версию для проверки гипотез (Yes/No): ')

object_detect = input('Запустить детектор объектов в кадре как новый признак (Yes/No): ')
velocity_ = input('Запустить метод velocity для признаков (Yes/No): ')
smoothing_ = input('Запустить метод smoothing для признаков (Yes/No): ')


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "UFC101"
if DEBUG == 'No':
    OUTPUT_PATH = BASE_DIR / "keypoints_data" / "yolo"
else:
    OUTPUT_PATH = BASE_DIR / "keypoints_data" / "DEBUG" / "yolo"
SEQUENCE_LENGTH = 30

model = YOLO('yolov8n-pose.pt') 
detect_model = YOLO('yolov8n.pt') #Модель для детекта объектов в кадре
OBJECT_CLASSES = {
    0: "person",
    1: "bicycle",
    17: "Dog",
    18: "horse",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    44: "Knife",
    79: "hair drier",
    80: "toothbrush",
    67: "keyboard"
}
OBJECT_FEATURE_SIZE = len(OBJECT_CLASSES)



colors = {
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'blue': (255, 0, 0)
}

def extract_object_features(detect_results, person_box):
    object_vector = np.zeros(OBJECT_FEATURE_SIZE)
    if detect_results.boxes is None:
        return object_vector

    boxes = detect_results.boxes.xyxy.cpu().numpy()
    classes = detect_results.boxes.cls.cpu().numpy()
    confs = detect_results.boxes.conf.cpu().numpy()
    px1, py1, px2, py2 = person_box

    margin = 30     # увеличиваем область вокруг человека
    px1 -= margin
    py1 -= margin
    px2 += margin
    py2 += margin
    class_ids = list(OBJECT_CLASSES.keys())

    for box, cls, conf in zip(boxes, classes, confs): #перебор всех параметров каждого объекта в кадре
        if conf < 0.5: # Если уверенность ниже 0.5 отбрасываем
            continue

        cls = int(cls) # Если объекта нет в нашем списке, тоже отбрасываем
        if cls not in OBJECT_CLASSES:
            continue

        x1, y1, x2, y2 = box # создаём бокс из координат объекта
        center_x = (x1 + x2) / 2 # находим центры
        center_y = (y1 + y2) / 2

        if (
            px1 <= center_x <= px2 and # объект рядом с человеком
            py1 <= center_y <= py2
        ):
            vector_idx = class_ids.index(cls)
            object_vector[vector_idx] = 1
    return object_vector


def temporal_smoothing(sequence, window_size=3):
    sequence = np.array(sequence)
    smoothed_sequence = []
    half_window = window_size // 2

    for i in range(len(sequence)):
        start = max(0, i - half_window)
        end = min(len(sequence), i + half_window + 1)

        window = sequence[start:end]
        smoothed_frame = np.mean(window, axis=0)
        smoothed_sequence.append(smoothed_frame)

    return smoothed_sequence


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Ошибка, не удалось загрузить видео')

    sequence = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.resize(frame, (224, 224))

        if object_detect == 'Yes':
            results = model(frame, verbose=False)[0] # Человек
            detect_results = detect_model(frame, verbose=False)[0] # Объекты
        else:
            results = model(frame, verbose=False)[0]



        if object_detect == 'Yes' and detect_results is None:
            sequence.append(np.zeros(51 + OBJECT_FEATURE_SIZE))
            continue


        if results.keypoints is None and object_detect == 'Yes':
            sequence.append(np.zeros(51 + OBJECT_FEATURE_SIZE))
            continue
        elif results.keypoints is None and object_detect == 'No':
            sequence.append(np.zeros(17 * 3))
            continue



        keypoints = results.keypoints.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()



        if len(boxes) == 0 and object_detect == 'Yes':
            sequence.append(np.zeros(51 + OBJECT_FEATURE_SIZE))
            continue
        elif len(boxes) == 0 and object_detect == 'No':
            sequence.append(np.zeros(17 * 3))
            continue

        '''
        Реализация Largest bbox selection, 
        для адеваткого сравнения с MediaPipe, 
        тоже будем брать лишь 1 человека в кадре. 
        Берём человека с наибольшим б-боксом
        '''

        areas = []

        for box in boxes:
            x1, y1, x2, y2 = box

            area = (x2 - x1) * (y2- y1)

            areas.append(area)

        largest_idx = np.argmax(areas)

        person_box = boxes[largest_idx]

        person_keypoints = keypoints[largest_idx]
        #нормализация
        person_keypoints[:, 0] /= 224
        person_keypoints[:, 1] /= 224

        person_keypoints = person_keypoints.flatten()

        if object_detect == 'Yes':
            object_features = extract_object_features(detect_results, person_box)
            combined_features = np.concatenate([person_keypoints, object_features])

            sequence.append(combined_features)

        else:
            sequence.append(person_keypoints)


    if velocity_ == 'Yes':
        cap.release()
        velocity_sequence = []

        for i in range(len(sequence)):
            current_pose = sequence[i]
            if i == 0:
                velocity = np.zeros_like(
                    current_pose
                )
            else:
                velocity = np.zeros_like(
                    current_pose
                )
                for kp in range(17):
                    x_idx = kp * 3
                    y_idx = kp * 3 + 1
                    velocity[x_idx] = (current_pose[x_idx] - sequence[i - 1][x_idx])
                    velocity[y_idx] = (current_pose[y_idx] - sequence[i - 1][y_idx])

            combined = np.concatenate([
                current_pose,
                velocity
            ])
            velocity_sequence.append(
                combined
            )
        return velocity_sequence
    
    if smoothing_ == 'Yes':
        cap.release()
        sequence = temporal_smoothing(
            sequence,
            window_size=3
        )
        return sequence
    
    else:
        cap.release()

        return sequence

def create_chunks(sequence):
    chunks = []

    for i in range(0, len(sequence), SEQUENCE_LENGTH):
        chunk = sequence[i:i + SEQUENCE_LENGTH]

        while len(chunk) < SEQUENCE_LENGTH:
            chunk.append(np.zeros(Chunk_shape))

        chunk = np.array(chunk)

        if chunk.shape != (SEQUENCE_LENGTH, Chunk_shape):
            continue
        if np.all(chunk == 0):
            continue
        chunks.append(chunk)

    return chunks

BASE_FEATURES = 51

if object_detect == 'Yes':
    BASE_FEATURES += OBJECT_FEATURE_SIZE

if velocity_ == 'Yes':
    Chunk_shape = BASE_FEATURES * 2
else:
    Chunk_shape = BASE_FEATURES


classes = ['Archery', 'Biking', 
           'PlayingGuitar', 'PlayingPiano', 'LongJump', 
           'Mixing', 'PizzaTossing', 'PlayingDaf',
           'Typing', 'TennisSwing', 'Skiing', 'SkateBoarding', 
           'Surfing','Basketball', 'BaskerballDunk', 'BaseballPitch']

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

        save_dir = OUTPUT_PATH / split

        os.makedirs(save_dir, exist_ok=True)

        # Process video

        sequence = process_video(video_path)
        chunks = create_chunks(sequence)

        for idx, chunk in enumerate(chunks):

            label_dir = save_dir / label

            label_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            save_name = (
                f"{clip_name}_chunk{idx}.npy"
            )

            save_path = os.path.join(label_dir, save_name)
            np.save(save_path, chunk)

            metadata.append({
                "npy_path": str(save_path),
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