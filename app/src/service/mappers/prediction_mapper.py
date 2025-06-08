from entities.task.prediction_task import PredictionTask, PredictionDTO


def prediction_task_to_dto(
        task: PredictionTask,
        user_email: str,
        model_name: str
) -> PredictionDTO:
    return PredictionDTO(
        id=task.id,
        user_email=user_email,
        model_name=model_name,
        inference_input=task.inference_input,
        result=task.result,
        is_success=task.is_success,
        cost=task.balance_withdrawal,
        request_timestamp=task.request_timestamp,
        result_timestamp=task.result_timestamp
    )
