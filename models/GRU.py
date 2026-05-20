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

    def forward(self, gru_output):

        attention_scores = self.attention(
            gru_output
        )
        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )
        context_vector = torch.sum(
            attention_weights * gru_output,
            dim=1
        )
        return context_vector


class GRUModel(nn.Module):

    def __init__(
        self,
        input_size=99,
        hidden_size=128,
        num_layers=2,
        num_classes=101,
        dropout=0.3
    ):

        super().__init__()
        self.gru = nn.GRU(

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
        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                256
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)

        gru_out = self.layer_norm(
            gru_out
        )

        context = self.attention(
            gru_out
        )

        context = self.dropout(
            context
        )
        out = self.fc(context)

        return out