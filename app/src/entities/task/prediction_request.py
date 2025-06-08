import uuid
from datetime import datetime
from sqlmodel import SQLModel, Field


class PredictionRequest(SQLModel):
    user_id: uuid.UUID = Field(foreign_key="user.id")
    model_id: uuid.UUID = Field(foreign_key="ml_models.id")
    user_email: str
    inference_input: str
    user_balance_before_task: float
    request_timestamp: datetime
