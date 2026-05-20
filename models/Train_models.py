import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score

import GRU
import LSTM
import Transformer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print(device)

class ActionDataset(Dataset):
    def __init__(self, data_path, label_enc=None):
        self.data = pd.read_csv(data_path, sep=',')

        if label_enc is None:
            self.LEncoder = LabelEncoder()
            self.data['label_encod'] = self.LEncoder.fit_transform(self.data['label'])

        else:
            self.LEncoder = label_enc
            self.data['label_encod'] = self.LEncoder.transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        array = np.load(row['npy_path'])
        label_array = row['label_encod']

        x = torch.tensor(array, dtype=torch.float32)
        y = torch.tensor(label_array, dtype=torch.long)

        return x, y

which_Data = input('Выберите данные для обучения - (mediapipe / yolo): ')

train_dataset = ActionDataset(f"C:/Users/user/Move_Detection/keypoints_data/{which_Data}/train_metadata.csv")
val_dataset   = ActionDataset(f"C:/Users/user/Move_Detection/keypoints_data/{which_Data}/test_metadata.csv", label_enc=train_dataset.LEncoder)
test_dataset  = ActionDataset(f"C:/Users/user/Move_Detection/keypoints_data/{which_Data}/val_metadata.csv", label_enc=train_dataset.LEncoder)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

which_model = input('Выберите модель для обучения - (LSTM / GRU / Temporal_CNN / Transformer): ')

if which_Data == 'mediapipe':
    input_size = 99
else:
    input_size = 51


if which_model == 'LSTM':
    model = LSTM.LSTMModel(input_size=input_size).to(device)
elif which_model == 'GRU':
     model = GRU.GRUModel(input_size=input_size).to(device)
elif which_model == 'Transformer':
     model = Transformer.TransformerModel(input_size=input_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

def train_epoch(model, loader):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for x, y in progress_bar:

        x = x.to(device)
        y = y.to(device)


        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == y).sum().item()

        total += y.size(0)
        accuracy = correct / total

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{accuracy:.4f}"
        })

    epoch_loss = total_loss / len(loader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader):

    model.eval()

    total_loss = 0

    preds = []
    targets = []

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for x, y in progress_bar:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(outputs, y)

            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            preds.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                y.cpu().numpy()
            )

    accuracy = accuracy_score(targets, preds)

    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy

EPOCHS = 40

best_acc = 0


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_epoch(
        model,
        train_loader
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_acc)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            f"best_{which_model}_model_{which_Data}.pth"
        )

        

        print("Best model saved!")