import torch
import torch.nn as nn
from torchvision import models
from scipy.optimize import minimize
import numpy as np
import torch.nn.functional as F

class MultiBackboneModel(nn.Module):
    """
    Модель с несколькими backbone сетями
    """
    def __init__(self, freeze_backbone=False):
        super(MultiBackboneModel, self).__init__()

        # ResNet50 backbone
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])

        # EfficientNet backbone
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet_features = nn.Sequential(*list(self.efficientnet.children())[:-1])

        # DenseNet backbone
        self.densenet = models.densenet121(pretrained=True)
        self.densenet_features = self.densenet.features

        if freeze_backbone:
            for model in [self.resnet_features, self.efficientnet_features, self.densenet_features]:
                for param in model.parameters():
                    param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.resnet_dim = 2048
        self.efficientnet_dim = 1280
        self.densenet_dim = 1024

        total_dim = self.resnet_dim + self.efficientnet_dim + self.densenet_dim
        self.attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),
            nn.ReLU(),
            nn.Linear(total_dim // 4, 3),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        resnet_feat = self.resnet_features(x)
        resnet_feat = self.adaptive_pool(resnet_feat).flatten(1)

        efficientnet_feat = self.efficientnet_features(x)
        efficientnet_feat = self.adaptive_pool(efficientnet_feat).flatten(1)

        densenet_feat = self.densenet_features(x)
        densenet_feat = self.adaptive_pool(densenet_feat).flatten(1)

        combined_features = torch.cat([resnet_feat, efficientnet_feat, densenet_feat], dim=1)

        attention_weights = self.attention(combined_features)

        weighted_resnet = resnet_feat * attention_weights[:, 0:1]
        weighted_efficientnet = efficientnet_feat * attention_weights[:, 1:2]
        weighted_densenet = densenet_feat * attention_weights[:, 2:3]

        final_features = torch.cat([weighted_resnet, weighted_efficientnet, weighted_densenet], dim=1)

        output = self.classifier(final_features)

        return output.squeeze()
