import logging
import time
import uuid
import tempfile
from datetime import datetime
from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Body, File, UploadFile, Form
from sqlmodel import Session
from starlette import status
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from celery_worker import perform_prediction
from config.metrics import PREDICT_REQUEST_COUNT, PREDICT_SUCCESS_REQUEST_LATENCY, \
    FAILED_PREDICTION_REQUEST_COUNT, PREDICT_FAILED_REQUEST_LATENCY, \
    record_duration
from database.database import get_session
from entities.auth.auth_entities import TokenData
from entities.task.prediction_request import PredictionRequest
from entities.task.prediction_task import PredictionDTO
from routes.home_router import templates
from service.auth.auth_service import get_current_active_user, authenticate_cookie
from service.crud.model_service import (
    get_model_by_name,
    get_default_model,
    validate_input,
    get_prediction_task_by_id,
    get_model_by_id,
    get_prediction_histories_by_user
)
from service.mappers.prediction_mapper import prediction_task_to_dto

prediction_router = APIRouter(prefix="/prediction", tags=["Prediction"])
logger = logging.getLogger(__name__)


@prediction_router.post("/predict")
async def create_prediction(
        token: Annotated[TokenData, Depends(authenticate_cookie)],
        session: Session = Depends(get_session),
        image: UploadFile = File(...),
        model_name: str = Form(...)
):
    start_time = time.time()
    PREDICT_REQUEST_COUNT.inc()
    logger.info("Received prediction request")

    if not validate_input(image.content_type):
        FAILED_PREDICTION_REQUEST_COUNT.inc()
        record_duration(PREDICT_FAILED_REQUEST_LATENCY, start_time)
        logger.warning(f"Invalid image format, image format: {image.content_type}")
        raise HTTPException(status_code=400, detail="Image format should be image/jpeg, image/png")

    model = get_model_by_name(model_name, session) if model_name else get_default_model(session)
    if not model:
        model = get_default_model(session)

    user = await get_current_active_user(token, session)
    logger.info(f"User authenticated, user_id: {user.id}")

    if user.balance < model.prediction_cost:
        logger.warning(
            f"Insufficient balance for prediction, user_id: {user.id}, required: {model.prediction_cost}, available: {user.balance}"
        )
        FAILED_PREDICTION_REQUEST_COUNT.inc()
        record_duration(PREDICT_FAILED_REQUEST_LATENCY, start_time)
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient balance. Required: {model.prediction_cost}, Available: {user.balance}"
        )

    UPLOAD_DIR = Path("/app/uploads")
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}.jpg"
    with tmp_path.open("wb") as f:
        f.write(await image.read())

    prediction_request = PredictionRequest(
        user_id=user.id,
        model_id=model.id,
        user_email=user.email,
        inference_input=str(tmp_path),
        user_balance_before_task=user.balance,
        request_timestamp=datetime.now()
    )

    task_id = uuid.uuid4()

    try:
        perform_prediction.apply_async(
            args=[prediction_request.dict(), task_id],
            task_id=str(task_id),
            queue="prediction"
        )
        logger.info(f"Celery task dispatched, task_id: {task_id}, user_id: {user.id}")
    except Exception as exc:
        FAILED_PREDICTION_REQUEST_COUNT.inc()
        record_duration(PREDICT_FAILED_REQUEST_LATENCY, start_time)
        logger.error(f"Failed to dispatch Celery task, error: {exc}")
        raise HTTPException(status_code=500, detail=f"Task error: {exc}")

    record_duration(PREDICT_SUCCESS_REQUEST_LATENCY, start_time)

    return {"task_id": task_id}


@prediction_router.post("/prediction_result", response_model=PredictionDTO)
async def get_prediction(
        token: Annotated[TokenData, Depends(authenticate_cookie)],
        session: Session = Depends(get_session),
        task_id: uuid.UUID = Body(..., embed=True)
):
    logger.info(f"Fetching prediction result, task_id: {task_id}")
    user = await get_current_active_user(token, session)
    task = get_prediction_task_by_id(task_id, session)

    if not task:
        logger.info(f"Prediction task is still processing, task_id: {task_id}")
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"detail": "Prediction task is still processing. Please try again later."}
        )

    logger.info(f"Prediction task found, task_id: {task_id}, user_id: {user.id}")
    model = get_model_by_id(task.model_id, session)
    return prediction_task_to_dto(task, user.email, model.name if model else "unknown")


@prediction_router.get("/history", response_class=HTMLResponse)
async def show_prediction_history(
        request: Request,
        token: Annotated[TokenData, Depends(authenticate_cookie)],
        session: Session = Depends(get_session)
):
    user = await get_current_active_user(token, session)
    logger.info(f"Fetching prediction history for user, user_id: {user.id}")
    predictions = get_prediction_histories_by_user(user.id, session)
    logger.info(f"Prediction history fetched, user_id: {user.id}, count: {len(predictions)}")
    prediction_dtos = [
        prediction_task_to_dto(task, user.email, get_model_by_id(task.model_id, session).name)
        for task in predictions
    ]
    return templates.TemplateResponse("prediction_history.html", {"request": request, "predictions": prediction_dtos})
