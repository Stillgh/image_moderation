import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends
from jwt import InvalidTokenError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlmodel import Session
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates

from config.auth_config import get_auth_settings
from database.database import get_session
from entities.auth.auth_entities import TokenData
from service.auth.auth_service import authenticate_cookie, get_current_active_user

home_router = APIRouter(tags=["Home"])
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger = logging.getLogger(__name__)


@home_router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    auth_settings = get_auth_settings()
    token = request.cookies.get(auth_settings.COOKIE_NAME)
    try:
        if token:
            user = await authenticate_cookie(token)
        else:
            user = None
    except InvalidTokenError as e:
        user = None

    context = {
        "user": user.username if user else "",
        "request": request
    }
    return templates.TemplateResponse("index.html", context)


@home_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@home_router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@home_router.get("/home", response_class=HTMLResponse)
async def home_page(
        request: Request,
        token: TokenData = Depends(authenticate_cookie),
        session: Session = Depends(get_session)
):
    logger.info(f"Handling root endpoint {request.state.request_id}")
    user = await get_current_active_user(token, session)
    return templates.TemplateResponse("home.html", {"request": request, "user": user})


@home_router.get("/logout", response_class=HTMLResponse)
async def login_get():
    auth_settings = get_auth_settings()
    response = RedirectResponse(url="/")
    response.delete_cookie(auth_settings.COOKIE_NAME)
    return response


@home_router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@home_router.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
