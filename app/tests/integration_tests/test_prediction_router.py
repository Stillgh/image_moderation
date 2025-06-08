import uuid
from time import sleep

import pytest

from celery_worker import celery, perform_prediction

import io
import uuid
import pytest
from PIL import Image


def _fake_jpeg(width: int = 10, height: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def test_create_prediction_image(admin_client, celery_worker_fixture, celery_app):
    files = {
        "image": ("test.jpg", _fake_jpeg(), "image/jpeg"),
    }
    data = {
        "model_name": "test_model",
    }

    response = admin_client.post("/prediction/predict", files=files, data=data)
    assert response.status_code == 200, response.text

    payload = response.json()
    task_id = payload.get("task_id")
    assert task_id, "task_id отсутствует в ответе"
    try:
        uuid.UUID(task_id)
    except ValueError:
        pytest.fail("task_id не является валидным UUID")

    res = celery_app.AsyncResult(task_id)
    assert res.get(timeout=10) == "Prediction succeeded"


def test_create_prediction_no_money(patched_client, celery_worker_fixture):
    files = {
        "image": ("test.jpg", _fake_jpeg(), "image/jpeg"),
    }
    data = {
        "model_name": "test_model",
    }

    response = patched_client.post("/prediction/predict", files=files, data=data)

    assert response.status_code == 400, "Insufficient balance" in response.text


def test_create_prediction_invalid_input(admin_client, celery_worker_fixture):
    files = {
        "image": ("test.dsa", _fake_jpeg(), "image/jpeg"),
    }
    data = {
        "model_name": "test_model",
    }

    response = admin_client.post("/prediction/predict", files=files, data=data)

    assert response.status_code == 400, "Image format should be image/jpeg, image/png" in response.text


def test_create_prediction_exception(monkeypatch, admin_client):
    # Override apply_async only for this test
    def fake_apply_async(*args, **kwargs):
        raise Exception("Simulated task exception")

    monkeypatch.setattr(perform_prediction, "apply_async", fake_apply_async)

    files = {
        "image": ("test.dsa", _fake_jpeg(), "image/jpeg"),
    }
    data = {
        "model_name": "test_model",
    }

    response = admin_client.post("/prediction/predict", files=files, data=data)

    assert response.status_code == 500
    assert "Task error" in response.text


def test_get_prediction(admin_client, celery_worker_fixture):
    # make prediction at first
    files = {
        "image": ("test.dsa", _fake_jpeg(), "image/jpeg"),
    }
    data = {
        "model_name": "test_model",
    }

    make_pred_resp = admin_client.post("/prediction/predict", files=files, data=data)

    prediction_data = make_pred_resp.json()
    try:
        uuid.UUID(prediction_data["task_id"])
    except ValueError:
        pytest.fail("Returned task_id is not a valid UUID")
    result = celery.AsyncResult(prediction_data["task_id"])
    assert result.get(timeout=10) == "Prediction succeeded"

    # get prediction
    payload = {
        "inference_input": "this is valid input",
        "task_id": prediction_data['task_id']
    }
    response = admin_client.post("/prediction/prediction_result", json=payload)
    assert response.status_code == 200, response
    data = response.json()
    assert data['id'] == prediction_data['task_id'], data['user_email'] == 'admin@example.com'
    assert data['inference_input'] == payload['inference_input']


def test_get_prediction_task_not_ready(admin_client, celery_worker_fixture):
    payload = {
        "inference_input": "this is valid input",
        "task_id": str(uuid.uuid4())
    }
    response = admin_client.post("/prediction/prediction_result", json=payload)
    assert response.status_code == 202, response.json()[
                                            'detail'] == "Prediction task is still processing. Please try again later."


def test_show_prediction_history(admin_client, celery_worker_fixture):
    response = admin_client.get("/prediction/history")

    assert response.status_code == 200
    assert "predictions" in response.context

    prev_len = len(response.context['predictions'])

    files = {
        "image": ("test.dsa", _fake_jpeg(), "image/jpeg"),
    }
    data = {
        "model_name": "test_model",
    }

    make_prediction_response = admin_client.post("/prediction/predict", files=files, data=data)

    prediction_data = make_prediction_response.json()
    first_pred_id = prediction_data["task_id"]

    make_prediction_response = admin_client.post("/prediction/predict", files=files, data=data)
    prediction_data = make_prediction_response.json()
    sec_pred_id = prediction_data["task_id"]
    sleep(5)
    response = admin_client.get("/prediction/history")

    assert response.status_code == 200
    assert len(response.context["predictions"]) == prev_len + 2
    assert first_pred_id in [str(pred.id) for pred in response.context['predictions']]
    assert sec_pred_id in [str(pred.id) for pred in response.context['predictions']]
