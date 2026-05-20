import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_output):
        attention_scores = self.attention(
            lstm_output
        )
        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )
        context_vector = torch.sum(
            attention_weights * lstm_output,
            dim=1
        )
        return context_vector


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=99,
        hidden_size=256,
        num_layers=3,
        num_classes=101,
        dropout=0.4
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(
            hidden_size * 2
        )
        self.attention = Attention(
            hidden_size
        )
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)

        context = self.attention(lstm_out)
        context = self.dropout(context)

        out = self.fc(context)

        return out