"""Conditional adversarial classifier with reversed gradients only on its features."""
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


class AdversarialClassifier(nn.Module):
    def __init__(self, embedding_dim, classes, hidden_dim=64, condition_classes=0):
        super().__init__()
        self.condition_classes = condition_classes
        self.layers = nn.Sequential(nn.Linear(embedding_dim + condition_classes, hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim, classes))

    def forward(self, features, weight=1.0, labels=None):
        features = GradientReverse.apply(features, weight)
        if self.condition_classes:
            condition = nn.functional.one_hot(labels, self.condition_classes).to(features.dtype)
            features = torch.cat((features, condition), dim=1)
        return self.layers(features)
