import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)
        div_term = torch.exp(

            torch.arange(
                0,
                d_model,
                2
            ) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )
        pe[:, 1::2] = torch.cos(
            position * div_term
        )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):

    def __init__(

        self,
        input_size=99,
        d_model=128,
        num_heads=4,
        num_layers=2,
        num_classes=101,
        dropout=0.3
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        self.positional_encoding = (
            PositionalEncoding(d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=256,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(
            d_model
        )
        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)

        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)

        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        out = self.fc(x)

        return out