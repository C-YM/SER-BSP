import torch
import torch.nn as nn
from torch.autograd import Function

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha * 1.0
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class CNN_LSTM_ATT_SINGLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.GRL = GRL()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(1024, 512, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.classificationHead = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )

    def forward(self, x, alpha):
        feature = self.feature(x)
        feature = feature.squeeze(dim=1)
        feature, _ = self.lstm(feature)
        reverse_feature = feature.mean(dim=1)

        reverse_feature = self.GRL.apply(reverse_feature, alpha)
        domain_predict = self.domain_classifier(reverse_feature)

        # feature, _ = self.lstm(feature)
        feature, _ = self.attn(feature, feature, feature)
        feature = feature.mean(dim=1)

        # reverse_feature = self.GRL.apply(feature, alpha)
        # domain_predict = self.domain_classifier(reverse_feature)

        task_predict = self.classificationHead(feature)

        return task_predict, domain_predict, feature


if __name__ == "__main__":
    x = torch.rand(32, 2, 150, 1024)
    model = CNN_LSTM_ATT_SINGLE()
    x = model(x, 0)
    print(model)
