import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlmodel import SQLModel, Field

from entities.task.prediction_request import PredictionRequest
from entities.task.prediction_result import PredictionResult


class PredictionTask(PredictionRequest, PredictionResult, SQLModel, table=True):

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)


class PredictionDTO(BaseModel):
    id: uuid.UUID
    user_email: str
    model_name: str
    inference_input: str
    result: str
    is_success: bool
    cost: float
    request_timestamp: datetime
    result_timestamp: datetime
