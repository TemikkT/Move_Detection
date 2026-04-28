import cv2
import time
import torch
import numpy as np
import os

from camera.video_stream import VideoStream
from detection.pose_detector import PoseDetector

# === МОДЕЛЬ ===
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        return self.fc(x)

# === НАСТРОЙКИ ===
input_size = 99
hidden_size = 128

labels = sorted(os.listdir("keypoints_data/sport_data"))
num_classes = len(labels)

SEQUENCE_LENGTH = 30
TARGET_FPS = 25  # 🔥 ограничение FPS

# === ЗАГРУЗКА МОДЕЛИ ===
model = LSTMModel(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("models/model_sport.pth", map_location="cpu"))
model.eval()

# === KEYPOINTS ===
def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(33 * 3)

# === MAIN ===
def main():
    camera = VideoStream()
    detector = PoseDetector()

    buffer = []
    pred_history = []

    prev_time = 0

    while True:
        start_time = time.time()

        frame = camera.get_frame()
        if frame is None:
            break

        # === DETECTION ===
        results = detector.process(frame)

        # === KEYPOINTS ===
        keypoints = extract_keypoints(results)
        buffer.append(keypoints)

        if len(buffer) > SEQUENCE_LENGTH:
            buffer.pop(0)

        action_text = "..."

        # === ПРЕДСКАЗАНИЕ ===
        if len(buffer) == SEQUENCE_LENGTH:
            input_data = torch.tensor([buffer], dtype=torch.float32)

            with torch.no_grad():
                outputs = model(input_data)
                pred = torch.argmax(outputs, dim=1).item()

            # === СГЛАЖИВАНИЕ ===
            pred_history.append(pred)
            if len(pred_history) > 10:
                pred_history.pop(0)

            pred = max(set(pred_history), key=pred_history.count)
            action_text = labels[pred]

        # === РИСУЕМ СКЕЛЕТ ===
        frame = detector.draw(frame, results)

        # === ТЕКСТ ===
        cv2.putText(frame, action_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # === FPS ===
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # === ПОКАЗ ===
        cv2.imshow("Action Detector", frame)

        # === ОГРАНИЧЕНИЕ FPS ===
        elapsed = time.time() - start_time
        delay = max(0, (1 / TARGET_FPS) - elapsed)
        time.sleep(delay)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()