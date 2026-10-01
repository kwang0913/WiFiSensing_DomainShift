"""Domain classifier with reversed gradients only on its input."""
import torch
from torch import nn


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight):
        ctx.weight = weight
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.weight * gradient, None


class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim, domains, hidden_dim=64, task_classes=0):
        super().__init__()
        self.task_classes = task_classes
        self.layers = nn.Sequential(nn.Linear(embedding_dim + task_classes, hidden_dim),
                                    # nn.LeakyReLU(0.1),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, domains))

    def forward(self, features, weight=1.0, labels=None):
        features = GradientReverse.apply(features, weight)
        if self.task_classes:
            condition = nn.functional.one_hot(labels, self.task_classes).to(features.dtype)
            features = torch.cat((features, condition), dim=1)
        return self.layers(features)
