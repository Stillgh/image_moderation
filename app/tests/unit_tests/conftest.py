import pytest
from sqlalchemy import create_engine, StaticPool
from sqlmodel import SQLModel, Session
from entities.ml_model.classification_model import ClassificationModel
from entities.ml_model.inference_input import InferenceInput
from entities.task.prediction_request import PredictionRequest
from entities.task.prediction_task import PredictionTask
from entities.user.user import User
from entities.user.balance_history import BalanceHistory

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
SQLModel.metadata.create_all(test_engine)


@pytest.fixture(name="session")
def session_fixture():
    with Session(test_engine) as session:
        yield session


def override_get_session():
    with Session(test_engine) as session:
        yield session
