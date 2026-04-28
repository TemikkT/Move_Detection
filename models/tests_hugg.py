import cv2
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# Загружаем модель
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"

processor = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

model.eval()

# Камера
cap = cv2.VideoCapture(0)

buffer = []
BUFFER_SIZE = 16  # модель ждёт ~16 кадров

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    buffer.append(frame_rgb)

    # держим только последние N кадров
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    # Когда накопили буфер → предсказание
    if len(buffer) == BUFFER_SIZE:
        inputs = processor(list(buffer), return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()

        label = model.config.id2label[predicted_class]

        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    cv2.imshow("HF Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()