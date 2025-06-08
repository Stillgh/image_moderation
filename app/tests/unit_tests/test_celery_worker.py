import uuid
from datetime import datetime

import pytest

from app.tests.unit_tests.conftest import override_get_session
from celery_worker import perform_prediction
from exceptions.model_exception import ModelException


def test_perform_prediction_success(monkeypatch):
    # Dictionary to record whether each mocked function was called.
    call_log = {
        "get_model_by_name": False,
        "make_prediction": False,
        "prepare_and_save_task": False,
        "withdraw_balance": False,
    }

    # Fake implementations that update our call_log.

    def fake_make_prediction(inference_input):
        call_log["make_prediction"] = True
        return "dummy_prediction_result"

    def fake_prepare_and_save_task(prediction_request, result, success, cost, task_id, session):
        call_log["prepare_and_save_task"] = True

    def fake_withdraw_balance(user_id, cost, session):
        call_log["withdraw_balance"] = True

    monkeypatch.setattr("celery_worker.make_prediction", fake_make_prediction)
    monkeypatch.setattr("celery_worker.prepare_and_save_task", fake_prepare_and_save_task)
    monkeypatch.setattr("celery_worker.withdraw_balance", fake_withdraw_balance)
    monkeypatch.setattr("celery_worker.get_session", override_get_session)

    prediction_request = {
        "inference_input": "dummy data",
        "user_id": uuid.uuid4(),
        "model_id": uuid.uuid4(),
        "user_email": "dummy@mail.ru",
        "user_balance_before_task": 500.0,
        "request_timestamp": datetime.now()
    }
    task_id = uuid.uuid4()
    model_name = "dummy_model"

    result = perform_prediction(prediction_request, task_id, model_name)

    assert call_log["get_model_by_name"] is True
    assert call_log["make_prediction"] is True
    assert call_log["prepare_and_save_task"] is True
    assert call_log["withdraw_balance"] is True
    assert result == 'Prediction succeeded'


def test_perform_prediction_failure(monkeypatch):
    call_log = {
        "get_model_by_name": False,
        "make_prediction": False,
        "prepare_and_save_task": False,
        "withdraw_balance": False,
    }


    def fake_make_prediction(inference_input):
        call_log["make_prediction"] = True
        raise Exception("prediction error")

    def fake_prepare_and_save_task(prediction_request, result, success, cost, task_id, session):
        call_log["prepare_and_save_task"] = True

    def fake_withdraw_balance(user_id, cost, session):
        call_log["withdraw_balance"] = True

    monkeypatch.setattr("celery_worker.make_prediction", fake_make_prediction)
    monkeypatch.setattr("celery_worker.prepare_and_save_task", fake_prepare_and_save_task)
    monkeypatch.setattr("celery_worker.withdraw_balance", fake_withdraw_balance)
    monkeypatch.setattr("celery_worker.get_session", override_get_session)

    prediction_request = {
        "inference_input": "dummy data",
        "user_id": uuid.uuid4(),
        "model_id": uuid.uuid4(),
        "user_email": "dummy@mail.ru",
        "user_balance_before_task": 500.0,
        "request_timestamp": datetime.now()
    }
    task_id = uuid.uuid4()
    model_name = "dummy_model"

    with pytest.raises(ModelException) as excinfo:
        perform_prediction(prediction_request, task_id, model_name)

    assert "Error during model prediction" in str(excinfo.value)
    assert call_log["prepare_and_save_task"] is True
    assert call_log["withdraw_balance"] is False