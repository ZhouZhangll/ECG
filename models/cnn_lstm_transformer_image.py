from __future__ import absolute_import
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['cnn_lstm_transformer_image']


class Classifier(nn.Module):
    def __init__(self, num_classes=2, input_channels=12):
        dim_hidden = 64
        hidden = [256, 128]
        super().__init__()

        # 2D卷积部分
        self.conv = nn.Sequential(
            # ------------------ stage 1 --------------------
            nn.Conv2d(input_channels, dim_hidden, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(dim_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(dim_hidden, dim_hidden * 2, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(dim_hidden * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Dropout(0.25),

            # ------------------ stage 2 --------------------
            nn.Conv2d(dim_hidden * 2, dim_hidden * 3, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(dim_hidden * 3),
            nn.LeakyReLU(),
            nn.Conv2d(dim_hidden * 3, dim_hidden * 3, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(dim_hidden * 3),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Dropout(0.25),

            # ------------------ stage 3 --------------------
            nn.Conv2d(dim_hidden * 3, dim_hidden * 3, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(dim_hidden * 3),
            nn.LeakyReLU(),
            nn.Conv2d(dim_hidden * 3, dim_hidden * 3, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(dim_hidden * 3),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Dropout(0.25),

            # ------------------ stage 4 --------------------
            nn.Conv2d(dim_hidden * 3, dim_hidden * 4, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(dim_hidden * 4),
            nn.LeakyReLU(),
            nn.Conv2d(dim_hidden * 4, dim_hidden * 4, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(dim_hidden * 4),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.25),
        )

        # LSTM和注意力部分保持不变
        self.lstm1 = nn.LSTM(dim_hidden * 4, hidden[0], batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden[0], hidden[1], batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(2 * hidden[1], 64),
            nn.SELU(),
            nn.Linear(64, num_classes)
        )
        self.dropout = nn.Dropout(0.5)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn

    def forward(self, x):
        # 2D卷积处理
        x = self.conv(x)  # [batch, channels, 1, 1]

        # 调整维度用于LSTM
        x = x.squeeze(-1).squeeze(-1)  # [batch, channels]
        x = x.unsqueeze(1)  # [batch, 1, channels]
        x = x.repeat(1, 10, 1)  # [batch, seq_len=10, channels]

        # LSTM和注意力处理
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        query = self.dropout(x)
        x, _ = self.attention_net(x, query)
        x = self.head(x)
        return x


def cnn_lstm_transformer_image(**kwargs):
    return Classifier(**kwargs)


if __name__ == "__main__":
    x = torch.randn(2, 12, 826,1478)
    model = cnn_lstm_transformer_image()
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be (2, 2)