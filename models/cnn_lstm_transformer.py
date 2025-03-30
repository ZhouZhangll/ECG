from __future__ import absolute_import
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['cnn_lstm_transformer']


class Classifier(nn.Module):
    def __init__(self, num_classes=2,input_size=12):
        dim_hidden=64
        hidden = [256,128]
        super().__init__()
        self.conv=nn.Sequential(
            #------------------stage 1--------------------
            nn.Conv1d(input_size,dim_hidden,24,stride=3),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden,dim_hidden*2,24,stride=1),
            nn.BatchNorm1d(dim_hidden*2),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=2),
            nn.Dropout(0.25),
            #------------------stage 2--------------------
            nn.Conv1d(dim_hidden*2,dim_hidden*3,12,stride=2),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden*3,dim_hidden*3,12,stride=1),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=2),
            nn.Dropout(0.25),
            #------------------stage 3--------------------
            nn.Conv1d(dim_hidden*3,dim_hidden*3,7,stride=2),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden*3,dim_hidden*3,7,stride=1),
            nn.BatchNorm1d(dim_hidden*3),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=2),
            nn.Dropout(0.25),
            #------------------stage 4--------------------
            nn.Conv1d(dim_hidden*3,dim_hidden*4,5,stride=1),
            nn.BatchNorm1d(dim_hidden*4),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden*4,dim_hidden*4,5,stride=1),
            nn.BatchNorm1d(dim_hidden*4),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(2,stride=1),
            nn.Dropout(0.25),
            )
        self.lstm1 =nn.LSTM(dim_hidden*4, hidden[0],
                            batch_first=True, bidirectional=True)
        self.lstm2=nn.LSTM(2 * hidden[0], hidden[1],
                            batch_first=True, bidirectional=True)
        self.head=nn.Sequential(
            nn.Linear(2 * hidden[1], 64),
            nn.SELU(),
            nn.Linear(64, num_classes)
            )
        self.dropout = nn.Dropout(0.5)
    #
    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim = -1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn
    def forward(self, x):
        x=self.conv(x)
        #print(x.shape)
        x=x.permute(0,2,1)
        x,_=self.lstm1(x)
        x,_=self.lstm2(x)
        query = self.dropout(x)
        x, _ = self.attention_net(x, query)
        x = self.head(x)
        return x


def cnn_lstm_transformer(**kwargs):
    return Classifier(**kwargs)


if __name__ == "__main__":
    x = torch.randn(2, 12, 5000)
    model = cnn_lstm_transformer()
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be (2, 2)