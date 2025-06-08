import uuid
from datetime import datetime

from sqlmodel import Session

from app.src.service.crud.model_service import (
    create_model,
    get_all_models,
    get_model_by_id,
    create_and_save_default_model,
    get_default_model,
    get_model_by_name,
    make_prediction,
    prepare_and_save_task,
    save_task,
    get_all_prediction_history,
    get_prediction_task_by_id,
    get_prediction_histories_by_user,
    get_prediction_histories_by_model,
    validate_input,
)
import io
from PIL import Image



from config.constants import DEFAULT_MODEL_NAME

from entities.ml_model.classification_model import ClassificationModel
from entities.ml_model.inference_input import InferenceInput
from entities.task.prediction_request import PredictionRequest
from entities.task.prediction_task import PredictionTask
from entities.user.user import User
from service.loaders.model_loader import ModelLoader

model_loader = ModelLoader()


def _fake_jpeg(width: int = 10, height: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="JPEG")
    return buf.getvalue()

def test_create_model(session: Session):
    model = ClassificationModel(name="TestModel", model_type="classification", prediction_cost=50.0)
    create_model(model, session)

    models = get_all_models(session)
    assert len(models) == 1
    fetched = get_model_by_id(model.id, session)
    assert fetched is not None
    assert fetched.name == "TestModel"


def test_create_and_get_default_model(session: Session):
    default_model = create_and_save_default_model()
    assert default_model.name == DEFAULT_MODEL_NAME
    assert default_model.model_type == "classification"
    assert default_model.prediction_cost == 100.0

    create_model(default_model, session)
    fetched = get_default_model(session)
    assert fetched is not None
    assert fetched.name == DEFAULT_MODEL_NAME


def test_get_model_by_name(session: Session):
    if not get_model_by_name(DEFAULT_MODEL_NAME, session):
        model = ClassificationModel(name=DEFAULT_MODEL_NAME, model_type="classification", prediction_cost=75.0)
        create_model(model, session)
    fetched = get_model_by_name(DEFAULT_MODEL_NAME, session)
    assert fetched is not None


def test_save_task(session: Session):
    task = PredictionTask(
        user_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        user_email="dummy@mail.ru",
        inference_input="dummy input",
        user_balance_before_task=150.0,
        request_timestamp=datetime.now(),
        result="neutral",
        is_success=True,
        balance_withdrawal=50.0,
        result_timestamp=datetime.now()
    )
    saved = save_task(task, session)
    assert saved.id is not None

    fetched = get_prediction_task_by_id(saved.id, session)
    assert fetched is not None
    assert fetched.result == "neutral"


def test_prepare_and_save_task(session: Session):
    user_id = uuid.uuid4()
    model_id = uuid.uuid4()
    now = datetime.now()
    request = PredictionRequest(
        user_id=user_id,
        model_id=model_id,
        user_email="dummy@mail.ru",
        inference_input="dummy input",
        user_balance_before_task=200.0,
        request_timestamp=now
    )

    task_id = uuid.uuid4()
    result_str = "positive"
    is_success = True
    cost = 100.0

    task = prepare_and_save_task(request, result_str, is_success, cost, task_id, session)

    fetched = get_prediction_task_by_id(task.id, session)
    assert fetched is not None
    assert fetched.result == result_str
    assert fetched.is_success == is_success
    assert fetched.balance_withdrawal == cost
    assert isinstance(fetched.result_timestamp, datetime)


def test_get_all_prediction_history(session: Session):
    task1 = PredictionTask(
        user_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        user_email="dummy@mail.ru",
        inference_input="input1",
        user_balance_before_task=100.0,
        request_timestamp=datetime.now(),
        result="positive",
        is_success=True,
        balance_withdrawal=20.0,
        result_timestamp=datetime.now()
    )
    task2 = PredictionTask(
        user_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        user_email="dummy@mail.ru",
        inference_input="input2",
        user_balance_before_task=200.0,
        request_timestamp=datetime.now(),
        result="neutral",
        is_success=False,
        balance_withdrawal=30.0,
        result_timestamp=datetime.now()
    )
    save_task(task1, session)
    save_task(task2, session)

    tasks = get_all_prediction_history(session)
    assert len(tasks) >= 2


def test_get_prediction_task_by_id(session: Session):
    task = PredictionTask(
        user_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        user_email="dummy@mail.ru",
        inference_input="input",
        user_balance_before_task=120.0,
        request_timestamp=datetime.now(),
        result="neutral",
        is_success=True,
        balance_withdrawal=15.0,
        result_timestamp=datetime.now()
    )
    save_task(task, session)
    fetched = get_prediction_task_by_id(task.id, session)
    assert fetched is not None
    assert fetched.id == task.id


def test_get_prediction_histories_by_user(session: Session):
    user_id = uuid.uuid4()
    task1 = PredictionTask(
        user_id=user_id,
        model_id=uuid.uuid4(),
        user_email="dummy@mail.ru",
        inference_input="user input 1",
        user_balance_before_task=100.0,
        request_timestamp=datetime(2023, 1, 2, 12, 0, 0),
        result="positive",
        is_success=True,
        balance_withdrawal=10.0,
        result_timestamp=datetime(2023, 1, 2, 12, 30, 0)
    )
    task2 = PredictionTask(
        user_id=user_id,
        model_id=uuid.uuid4(),
        user_email="dummy@mail.ru",
        inference_input="user input 2",
        user_balance_before_task=110.0,
        request_timestamp=datetime(2023, 1, 1, 12, 0, 0),
        result="neutral",
        is_success=False,
        balance_withdrawal=20.0,
        result_timestamp=datetime(2023, 1, 1, 12, 30, 0)
    )
    save_task(task1, session)
    save_task(task2, session)

    tasks = get_prediction_histories_by_user(user_id, session)
    assert len(tasks) == 2
    # The tasks should be ordered by request_timestamp descending.
    assert tasks[0].request_timestamp >= tasks[1].request_timestamp


def test_get_prediction_histories_by_model(session: Session):
    model_id = uuid.uuid4()
    task1 = PredictionTask(
        user_id=uuid.uuid4(),
        model_id=model_id,
        user_email="dummy@mail.ru",
        inference_input="model input 1",
        user_balance_before_task=90.0,
        request_timestamp=datetime.now(),
        result="positive",
        is_success=True,
        balance_withdrawal=10.0,
        result_timestamp=datetime(2023, 2, 1, 12, 0, 0)
    )
    task2 = PredictionTask(
        user_id=uuid.uuid4(),
        model_id=model_id,
        user_email="dummy@mail.ru",
        inference_input="model input 2",
        user_balance_before_task=95.0,
        request_timestamp=datetime.now(),
        result="neutral",
        is_success=False,
        balance_withdrawal=5.0,
        result_timestamp=datetime(2023, 2, 2, 12, 0, 0)
    )
    save_task(task1, session)
    save_task(task2, session)

    tasks = get_prediction_histories_by_model(model_id, session)
    assert len(tasks) == 2
    assert tasks[0].result_timestamp >= tasks[1].result_timestamp


def test_validate_input():
    assert validate_input("valid input") is True
    assert validate_input("bad") is False
    assert validate_input("") is False
    assert not validate_input(None)
