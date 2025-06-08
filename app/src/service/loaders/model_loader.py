import os, sys
import logging
import torch
import torch.nn as nn
from typing import Dict
import pickle
from typing import Any
from torchvision import transforms
from pathlib import Path

from entities.ml_model.model_architecture import MultiBackboneModel

logger = logging.getLogger(__name__)
MODEL_PATH = "/app/ml_models/nsfw_model"


class ModelLoader:

    _model = None
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    @classmethod
    def init(cls, device: str | torch.device = "cpu") -> None:

        if cls._model is None:
            model = MultiBackboneModel(freeze_backbone=False)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            cls._model = model.to(device).eval()
            cls._device = device
            print("✅ NSFW-модель загружена!")

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls.init()
        return cls._model

