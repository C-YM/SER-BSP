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


class BSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.GRL = GRL()
        self.lstm = nn.LSTM(1024, 512, num_layers=1, batch_first=True, bidirectional=True)
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

    def forward(self, feature, alpha):
        feature = feature.squeeze(dim=1)
        feature, _ = self.lstm(feature)
        feature = feature.mean(dim=1)
        reverse_feature = self.GRL.apply(feature, alpha)
        domain_predict = self.domain_classifier(reverse_feature)
        task_predict = self.classificationHead(feature)

        return task_predict, domain_predict, feature


if __name__ == "__main__":
    x = torch.rand(32, 1, 150, 1024)
    model = BSP()
    x = model(x, 0)
    print(x)
