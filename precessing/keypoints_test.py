import numpy as np
from pathlib import Path
import os

select_direct = input('Введите dir (mediapipe / yolo): ')

dataset = input("Введите split (train / val / test): ")


folder = Path(f"../keypoints_data/{select_direct}/{dataset}")

total_files = 0
for dirpath, dirnames, filenames in os.walk(folder):
    for filename in filenames:

        if not filename.endswith(".npy"):
            continue

        full_path = os.path.join(dirpath, filename)

        data = np.load(full_path)

        total_files += 1

        print("\n========================")
        print(f"Файл: {filename}")
        print(f"Shape: {data.shape}")
        print(f"Min: {np.min(data)}")
        print(f"Max: {np.max(data)}")
        print(f"NaN count: {np.isnan(data).sum()}")

print(f"\nВсего файлов: {total_files}")