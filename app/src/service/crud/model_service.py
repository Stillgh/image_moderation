import logging
from datetime import datetime
import uuid
from typing import List

from sqlmodel import Session, select, delete

from config.constants import DEFAULT_MODEL_NAME
from entities.ml_model.inference_input import InferenceInput
from entities.task.prediction_request import PredictionRequest
from entities.task.prediction_result import PredictionResult
from entities.task.prediction_task import PredictionTask
from entities.ml_model.ml_model import MLModel

from entities.ml_model.classification_model import ClassificationModel
from service.loaders.model_loader import ModelLoader

model_loader = ModelLoader()
logger = logging.getLogger(__name__)


def create_model(new_model: ClassificationModel, session: Session) -> None:
    logger.info(f"Creating {new_model.name} model in database")
    session.add(new_model)
    session.commit()
    session.refresh(new_model)
    logger.info(f"{new_model.name} model was created")


def get_all_models(session: Session) -> List[MLModel]:
    logger.info("Getting all models")
    return session.query(ClassificationModel).all()


def get_model_by_id(id: uuid, session: Session) -> ClassificationModel:
    logger.info(f"Getting model by id {id}")

    statement = select(ClassificationModel).where(ClassificationModel.id == id)
    result = session.exec(statement).first()
    logger.info(f"Model with id {id} was fetched")

    return result


def create_and_save_default_model():
    return ClassificationModel(name=DEFAULT_MODEL_NAME, model_type='classification', prediction_cost=100.0)


def get_default_model(session: Session):
    return get_model_by_name(DEFAULT_MODEL_NAME, session)


def get_model_by_name(name: str, session: Session) -> ClassificationModel:
    logger.info(f"Getting {name} model from database")

    statement = select(ClassificationModel) \
        .where(ClassificationModel.name == name)

    result = session.exec(statement).first()

    logger.info(f"{name} model was fetched from database")

    return result


def make_prediction(inference_input: InferenceInput) -> str:
    logger.info(f"Making prediction")
    model = ClassificationModel()
    res = model.predict(ModelLoader.get_model(), inference_input.data, ModelLoader._transform)
    logger.info(f"Prediction made")

    return "Image allowed" if res[0] else "Image is not allowed"


def prepare_and_save_task(request: PredictionRequest, result: str, is_success: bool, cost: float,
                          task_id: uuid, session: Session) -> PredictionTask:
    logger.info(f"Preparing and saving task {task_id}")

    pred_result = PredictionResult(
        result=result,
        is_success=is_success,
        balance_withdrawal=cost,
        result_timestamp=datetime.now()
    )

    task = PredictionTask(
        id=task_id,
        user_id=request.user_id,
        model_id=request.model_id,
        user_email=request.user_email,
        inference_input=request.inference_input,
        user_balance_before_task=request.user_balance_before_task,
        request_timestamp=request.request_timestamp,
        result=pred_result.result,
        is_success=pred_result.is_success,
        balance_withdrawal=pred_result.balance_withdrawal,
        result_timestamp=pred_result.result_timestamp
    )
    task = save_task(task, session)
    logger.info(f"Task was saved task {task.id}")

    return task


def save_task(task: PredictionTask, session: Session) -> PredictionTask:
    try:
        session.add(task)
        session.commit()
        session.refresh(task)
    except Exception as e:
        logger.error(f"Error creating prediction task with id {task.id}: {e}")
        session.rollback()
    return task


def get_all_prediction_history(session: Session) -> List[PredictionTask]:
    logger.info("Getting all prediction history")
    return session.query(PredictionTask).all()


def get_prediction_task_by_id(task_id: uuid, session: Session) -> PredictionTask:
    logger.info(f"Getting prediction task by id {task_id}")
    statement = select(PredictionTask).where(PredictionTask.id == task_id)
    result = session.exec(statement).first()
    return result


def get_all_prediction_histories(session: Session) -> List[PredictionTask]:
    statement = select(PredictionTask) \
        .order_by(PredictionTask.request_timestamp.desc())

    result = session.exec(statement).all()
    return result


def get_prediction_histories_by_user(user_id: uuid.UUID, session: Session) -> List[PredictionTask]:
    statement = select(PredictionTask) \
        .where(PredictionTask.user_id == user_id) \
        .order_by(PredictionTask.request_timestamp.desc())

    result = session.exec(statement).all()
    return result


def remove_prediction_histories_by_user(user_id: uuid.UUID, session: Session) -> int:
    statement = delete(PredictionTask).where(PredictionTask.user_id == user_id)
    result = session.exec(statement)
    session.commit()
    return result.rowcount


def get_prediction_histories_by_model(model_id: uuid.UUID, session: Session) -> List[PredictionTask]:
    statement = select(PredictionTask) \
        .where(PredictionTask.model_id == model_id) \
        .order_by(PredictionTask.request_timestamp.desc())

    result = session.exec(statement).all()
    return result


def validate_input(content_type: str) -> bool:
    return content_type in {"image/jpeg", "image/png"}
