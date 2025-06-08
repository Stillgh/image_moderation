from sqlmodel import SQLModel
from datetime import datetime


class PredictionResult(SQLModel):
    result: str
    is_success: bool
    balance_withdrawal: float
    result_timestamp: datetime
