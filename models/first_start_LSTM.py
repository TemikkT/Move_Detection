import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score


class ActionDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []
        self.label_map = {}

        actions = sorted(os.listdir(data_path))

        for idx, action in enumerate(actions):
            self.label_map[action] = idx
            action_path = os.path.join(data_path, action)

            for file in os.listdir(action_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(action_path, file)
                    self.data.append(file_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
    

train_dataset = ActionDataset("C:/Users/Artyom/Move_Detection/keypoints_data/train")
val_dataset   = ActionDataset("C:/Users/Artyom/Move_Detection/keypoints_data/test")
test_dataset  = ActionDataset("C:/Users/Artyom/Move_Detection/keypoints_data/val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
    
input_size = 99       # если только pose
hidden_size = 128
num_classes = len(train_dataset.label_map)

model = LSTMModel(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for x, y in loader:
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            preds.extend(predicted.numpy())
            targets.extend(y.numpy())

    return accuracy_score(targets, preds)

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {train_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

test_acc = evaluate(model, test_loader)
print("Test Accuracy:", test_acc)
torch.save(model.state_dict(), "model.pth")