import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlmodel import Session
from starlette.requests import Request
from starlette.responses import HTMLResponse

from database.database import get_session
from entities.auth.auth_entities import TokenData
from entities.user.user_role import UserRole
from routes.home_router import templates
from service.auth.auth_service import authenticate_cookie, get_current_active_user
from service.crud.model_service import get_model_by_id, get_all_prediction_histories
from service.crud.user_service import get_all_users, add_balance
from service.mappers.prediction_mapper import prediction_task_to_dto
from service.mappers.user_mapper import user_to_user_dto

admin_router = APIRouter(prefix="/admin", tags=["Admin"])
logger = logging.getLogger(__name__)


@admin_router.get("/", response_class=HTMLResponse)
async def admin_panel(
        request: Request,
        token: Annotated[TokenData, Depends(authenticate_cookie)],
        session: Session = Depends(get_session)
):
    admin_user = await get_current_active_user(token, session)
    if admin_user.role != UserRole.ADMIN:
        return templates.TemplateResponse("admin_required.html", {"request": request})
    return templates.TemplateResponse("admin.html", {"request": request})


@admin_router.get("/users", response_class=HTMLResponse)
async def get_users(
        request: Request,
        token: Annotated[TokenData, Depends(authenticate_cookie)],
        session: Session = Depends(get_session)):
    user = await get_current_active_user(token, session)

    if not user or user.role != UserRole.ADMIN:
        return templates.TemplateResponse("admin_required.html", {"request": request})
    all_users = list(map(user_to_user_dto, get_all_users(session)))
    return templates.TemplateResponse("admin_users.html", {"request": request, "users": all_users})


@admin_router.get("/admin_required", response_class=HTMLResponse)
async def admin_required(request: Request,
                         token: Annotated[TokenData, Depends(authenticate_cookie)],
                         session: Annotated[Session, Depends(get_session)]):
    user = await get_current_active_user(token, session)
    if not user or user.role != UserRole.ADMIN:
        return templates.TemplateResponse("admin_required.html", {"request": request})
    return templates.TemplateResponse("admin.html", {"request": request})


@admin_router.post("/users/add_balance", response_class=HTMLResponse)
async def admin_add_balance(
        request: Request,
        token: Annotated[TokenData, Depends(authenticate_cookie)],
        session: Session = Depends(get_session)
):
    logger.info("Received request to add balance by admin")
    admin_user = await get_current_active_user(token, session)
    if admin_user.role != UserRole.ADMIN:
        logger.warning(f"Unauthorized add balance attempt by user: {admin_user.email}")
        return HTMLResponse("Admin privileges required", status_code=403)

    form = await request.form()
    email = form.get("email")
    amount_str = form.get("amount")
    logger.info(f"Add balance by admin request details - email: {email}, amount: {amount_str}")

    try:
        amount = float(amount_str)
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid amount provided: {amount_str} - {str(e)}")
        return HTMLResponse("Invalid amount provided", status_code=400)

    try:
        add_balance(email, amount, session)
        logger.info(f"Successfully added {amount} to user: {email}")
    except Exception as e:
        logger.error(f"Error adding balance to user {email}: {str(e)}")
        return HTMLResponse(f"Error: {str(e)}", status_code=400)

    return HTMLResponse(f"Successfully added {amount} to {email}", status_code=200)


@admin_router.get("/prediction_history_all", response_class=HTMLResponse)
async def show_prediction_history(
    request: Request,
    token: Annotated[TokenData, Depends(authenticate_cookie)],
    session: Session = Depends(get_session)
):
    logger.info(f"Fetching prediction history for all users")
    predictions = get_all_prediction_histories(session)
    logger.info(f"Prediction history fetched, count: {len(predictions)}")
    prediction_dtos = [
        prediction_task_to_dto(task, task.user_email, get_model_by_id(task.model_id, session).name)
        for task in predictions
    ]
    return templates.TemplateResponse("prediction_history.html", {"request": request, "predictions": prediction_dtos})
