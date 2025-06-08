from datetime import datetime
from sqlmodel import SQLModel, Field
import uuid


class BalanceHistory(SQLModel, table=True):

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id")
    amount_before_change: float
    amount_change: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
