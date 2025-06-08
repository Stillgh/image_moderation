import uuid
from datetime import datetime

from entities.task.prediction_task import PredictionTask
from entities.user.user import User, UserSignUp
from entities.user.user_role import UserRole
from service.crud.user_service import verify_password
from service.mappers.prediction_mapper import prediction_task_to_dto
from service.mappers.user_mapper import user_to_user_dto, user_signup_dto_to_user


def test_user_to_user_dto():
    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        name="John",
        surname="Doe",
        hashed_password="dummy_hash",
        balance=100.0,
        role=UserRole.USER,
        disabled=False
    )

    user_dto = user_to_user_dto(user)

    assert user_dto.email == user.email
    assert user_dto.name == user.name
    assert user_dto.surname == user.surname
    assert user_dto.balance == user.balance
    assert user_dto.role == user.role


def test_user_signup_dto_to_user():
    password = "secure_password"
    signup_dto = UserSignUp(
        email="newuser@example.com",
        name="Alice",
        surname="Smith",
        password=password
    )

    user = user_signup_dto_to_user(signup_dto)

    assert user.id is not None
    assert isinstance(user.id, uuid.UUID)

    assert user.email == signup_dto.email
    assert user.name == signup_dto.name
    assert user.surname == signup_dto.surname
    assert user.balance == 0.0
    assert user.role == UserRole.USER

    assert user.hashed_password != password
    assert verify_password(password, user.hashed_password) is True


def test_prediction_task_to_dto():
    task_id = uuid.uuid4()
    now = datetime.now()
    task = PredictionTask(
        id=task_id,
        user_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        inference_input="Test inference input",
        user_balance_before_task=100.0,
        request_timestamp=now,
        result="Test result",
        is_success=True,
        balance_withdrawal=50.0,
        result_timestamp=now
    )
    user_email = "user@example.com"
    model_name = "Test Model Name"

    dto = prediction_task_to_dto(task, user_email, model_name)

    assert dto.id == task_id
    assert dto.user_email == user_email
    assert dto.model_name == model_name
    assert dto.inference_input == "Test inference input"
    assert dto.result == "Test result"
    assert dto.is_success is True
    assert dto.cost == 50.0
    assert dto.request_timestamp == now
    assert dto.result_timestamp == now