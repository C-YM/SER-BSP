import torch
import torch.nn as nn


class CNN_ATT_LSTM_SINGLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 24
        self.channel = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.L, out_channels=self.channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channel),  # 尝试
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(1024, 512, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.classificationHead = nn.Linear(256, 8)

    def forward(self, input):
        conv_input = self.conv1(input)
        del input  # 删除不再需要的张量
        x = self.conv2(conv_input)
        del conv_input  # 删除不再需要的张量
        x = x.squeeze(dim=1)
        x, _ = self.attn(x, x, x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        feature = x.mean(dim=1)  # 256 维 embedding
        task_predict = self.classificationHead(feature)
        return task_predict, feature


if __name__ == "__main__":
    x = torch.rand(32, 24, 150, 1024)
    model = CNN_ATT_LSTM_SINGLE()
    task_predict, feature = model(x)
    print(task_predict.shape, feature.shape)
