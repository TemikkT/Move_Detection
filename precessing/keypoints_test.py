import numpy as np
from pathlib import Path
import os

dataset = input("Название директории с датасетом")

folder = Path(f'../keypoints_data/{dataset}')

for dirpath, dirname, filenames in os.walk(folder):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)

        data = np.load(full_path)

        print(f'Действие: {dirname}, значения: {data.shape}')
        print(f'Минимум {np.min(data)}, Максимум{np.max(data)}')