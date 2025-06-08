import uuid
from sqlmodel import SQLModel, Field
from abc import ABC, abstractmethod
from .inference_input import InferenceInput


class MLModel(SQLModel, ABC):

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str
    model_type: str
    prediction_cost: float = Field(ge=0.0)

    @abstractmethod
    def predict(self, data_input: InferenceInput):
        pass
