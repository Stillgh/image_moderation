import logging

from fastapi import APIRouter, Depends, HTTPException, Form, Body
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import EmailStr
from sqlmodel import Session
from typing import Annotated

from starlette import status
from starlette.requests import Request
from starlette.responses import RedirectResponse, HTMLResponse

from config.auth_config import get_auth_settings
from database.database import get_session
from entities.auth.auth_entities import TokenData
from entities.user.user import UserSignUp, UserDTO
from routes.home_router import templates
from service.auth.auth_service import get_current_active_user, authenticate_cookie
from service.auth.jwt_service import create_access_token
from service.crud.user_service import (
    create_user,
    add_balance,
    withdraw_balance,
    get_balance_histories,
    get_user_by_email,
    find_and_verify_user
)
from service.mappers.user_mapper import user_signup_dto_to_user

user_router = APIRouter(prefix="/users", tags=["User"])
logger = logging.getLogger(__name__)


@user_router.post("/login")
async def login(
    user_login: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
) -> RedirectResponse:
    logger.info(f"Login attempt for username: {user_login.username}")
    auth_settings = get_auth_settings()
    user = find_and_verify_user(user_login.username, user_login.password, session)
    logger.info(f"User verified for login: {user_login.username}")
    response = RedirectResponse(url="/home", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key=auth_settings.COOKIE_NAME,
        value=f"Bearer {create_access_token(user)}",
        httponly=True
    )
    logger.info(f"Login cookie set for user: {user.email}")
    return response


@user_router.post("/token")
async def get_token(
    user_login: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
) -> dict:
    logger.info(f"Token request for username: {user_login.username}")
    auth_settings = get_auth_settings()
    user = find_and_verify_user(user_login.username, user_login.password, session)
    token = create_access_token(user)
    logger.info(f"Token created for user: {user.email}")
    return {auth_settings.COOKIE_NAME: token, "token_type": "bearer"}


@user_router.post("/signup")
async def create_new_user(
    username: str = Form(...),
    surname: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session)
) -> RedirectResponse:
    logger.info(f"Signup attempt for email: {email}")
    user_data = UserSignUp(name=username, surname=surname, email=email, password=password)
    user = get_user_by_email(user_data.email, session)
    auth_settings = get_auth_settings()
    if user:
        logger.warning(f"User with email {user_data.email} already exists")
        raise HTTPException(status_code=409, detail=f"User with email {user_data.email} already exists")
    try:
        user = user_signup_dto_to_user(user_data)
        create_user(user, session)
        logger.info(f"User created successfully: {user.email}")
        response = RedirectResponse(url="/home", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key=auth_settings.COOKIE_NAME,
            value=f"Bearer {create_access_token(user)}",
            httponly=True
        )
        logger.info(f"Signup cookie set for user: {user.email}")
        return response
    except Exception as e:
        logger.error(f"Error during signup for {email}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post("/balance/add")
async def add_user_balance(
    token: Annotated[TokenData, Depends(authenticate_cookie)],
    session: Session = Depends(get_session),
    amount: float = Body(..., embed=True)
) -> dict:
    try:
        user = await get_current_active_user(token, session)
        logger.info(f"Adding balance {amount} for user: {user.email}")
        add_balance(user.email, amount, session)
        new_balance = float(user.balance)
        logger.info(f"New balance for {user.email} is {new_balance}")
        return {
            "message": f"Successfully added {amount} to balance for user {user.email}",
            "new_balance": new_balance
        }
    except Exception as e:
        logger.error(f"Error adding balance for user: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@user_router.post("/balance/withdraw")
async def withdraw_user_balance(
    token: Annotated[TokenData, Depends(authenticate_cookie)],
    session: Session = Depends(get_session),
    amount: float = Body(..., embed=True)
):
    try:
        user = await get_current_active_user(token, session)
        logger.info(f"Attempting to withdraw {amount} for user: {user.email}")
        withdraw_balance(user.id, amount, session)
        new_balance = float(user.balance)
        logger.info(f"New balance after withdrawal for {user.email} is {new_balance}")
        return {
            "message": f"Successfully withdrew {amount} from balance for user {user.email}",
            "new_balance": new_balance
        }
    except Exception as e:
        logger.error(f"Error withdrawing balance for user {user.email}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@user_router.get("/balance/current", response_model=float)
async def get_user_current_balance(
    token: Annotated[TokenData, Depends(authenticate_cookie)],
    session: Session = Depends(get_session)
):
    user = await get_current_active_user(token, session)
    logger.info(f"Fetching current balance for user {user.email}, balance: {user.balance}")
    return user.balance


@user_router.get("/balance/history", response_class=HTMLResponse)
async def balance_history(
    request: Request,
    token: Annotated[TokenData, Depends(authenticate_cookie)],
    session: Session = Depends(get_session)
):
    user = await get_current_active_user(token, session)
    logger.info(f"Fetching balance history for user: {user.email}")
    history = get_balance_histories(user.id, session)
    logger.info(f"Balance history fetched for user {user.email}, records: {len(history)}")
    return templates.TemplateResponse("balance_history.html", {"request": request, "history": history})


@user_router.get("/myinfo", response_model=UserDTO)
async def my_info(
    request: Request,
    token: Annotated[TokenData, Depends(authenticate_cookie)],
    session: Session = Depends(get_session)
):
    user = await get_current_active_user(token, session)
    logger.info(f"Fetching info for user: {user.email}")
    return templates.TemplateResponse("myinfo.html", {"request": request, "user": user})
