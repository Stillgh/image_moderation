from typing import Optional

import torch
from pydantic import PrivateAttr

from entities.ml_model.inference_input import InferenceInput
from entities.ml_model.ml_model import MLModel
from sqlmodel import Field
from PIL import Image
from pathlib import Path


class ClassificationModel(MLModel, table=True):
    __tablename__ = "ml_models"

    model_type: str = Field(default="classification", const=True)
    _model: Optional[torch.nn.Module] = PrivateAttr(default=None)

    def predict(self, model, img_path, transform, thr=0.5, device="cpu"):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        is_nsfw = prob > thr
        return int(is_nsfw), prob

