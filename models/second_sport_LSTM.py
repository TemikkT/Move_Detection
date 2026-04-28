import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch.optim as optim
from sklearn.metrics import accuracy_score

class ActionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


def load_data(data_path):
    data = []
    labels = []
    label_map = {}

    actions = sorted(os.listdir(data_path))

    for idx, action in enumerate(actions):
        label_map[action] = idx
        action_path = os.path.join(data_path, action)

        for file in os.listdir(action_path):
            if file.endswith(".npy"):
                data.append(os.path.join(action_path, file))
                labels.append(idx)

    return data, labels, label_map


# === ЗАГРУЗКА ===
data, labels, label_map = load_data("C:/Users/Artyom/Move_Detection/keypoints_data/sport_data")

# === SPLIT ===
X_train, X_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# === DATASETS ===
train_dataset = ActionDataset(X_train, y_train)
val_dataset   = ActionDataset(X_val, y_val)
test_dataset  = ActionDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        return self.fc(x)
    

input_size = 99
hidden_size = 128
num_classes = len(label_map)

model = LSTMModel(input_size, hidden_size, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1).cpu()

            preds.extend(predicted.numpy())
            targets.extend(y.numpy())

    return accuracy_score(targets, preds)

EPOCHS = 15

for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")


test_acc = evaluate(model, test_loader)
print("Test Accuracy:", test_acc)
torch.save(model.state_dict(), "model_sport.pth")