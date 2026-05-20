import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO


DEBUG = input('Запустить малую версию для проверки гипотез (Yes/No): ')

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "UFC101"
if DEBUG == 'No':
    OUTPUT_PATH = BASE_DIR / "keypoints_data" / "yolo"
else:
    OUTPUT_PATH = BASE_DIR / "keypoints_data" / "DEBUG" / "yolo"
SEQUENCE_LENGTH = 30

FRAME_SKIP = 2

model = YOLO('yolov8n-pose.pt')

colors = {
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'blue': (255, 0, 0)
}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Ошибка, не удалось загрузить видео')

    sequence = []

    frame_idx = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break
        
        frame_idx += 1

        if frame_idx % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (224, 224))

        results = model(frame, verbose=False)[0]

        if results.keypoints is None:
            sequence.append(
                np.zeros(17 * 3)
            )

            continue

        keypoints = results.keypoints.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            sequence.append(
                np.zeros(17 * 3)
            )

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

        person_keypoints = keypoints[largest_idx]
        #нормализация
        person_keypoints[:, 0] /= 224
        person_keypoints[:, 1] /= 224

        person_keypoints = person_keypoints.flatten()


        sequence.append(person_keypoints)

    cap.release()

    return sequence

def create_chunks(sequence):
    chunks = []

    for i in range(0, len(sequence), SEQUENCE_LENGTH):
        chunk = sequence[i:i + SEQUENCE_LENGTH]

        while len(chunk) < SEQUENCE_LENGTH:
            chunk.append(np.zeros(51))
        chunk = np.array(chunk)

        if chunk.shape != (SEQUENCE_LENGTH, 51):
            continue
        if np.all(chunk == 0):
            continue
        chunks.append(chunk)

    return chunks


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