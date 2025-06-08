import logging
import os
import threading
import uuid
import logging.config

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import RedirectResponse

from config.auth_config import get_auth_settings
from config.logging_config import LOGGING_CONFIG
from database.database import get_session
from database.tables_initiator import init_db
from routes.admin_router import admin_router
from routes.prediction_router import prediction_router
from routes.user_router import user_router
from routes.home_router import home_router
from service.auth.auth_service import get_current_active_user
from service.auth.jwt_service import verify_token
from service.loaders.model_loader import ModelLoader
from tg_api.tg_api import TgBot
from entities.ml_model.model_architecture import MultiBackboneModel

app = FastAPI()
tg_bot = TgBot()

app.include_router(user_router)
app.include_router(home_router)
app.include_router(prediction_router)
app.include_router(admin_router)
logger = logging.getLogger(__name__)

ALLOWED_PATHS = ["/", "/metrics", "/health", "/docs", "/openapi.json", "/users/token", "/login", "/signup", "/users/signup", "/users/login"]


@app.middleware("http")
async def restrict_access_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    logger.info(f"Received request: {request_id} ")
    if request.url.path not in ALLOWED_PATHS:
        auth_settings = get_auth_settings()
        token = request.cookies.get(auth_settings.COOKIE_NAME)
        try:
            if not token:
                return RedirectResponse(url="/")
            token_data = verify_token(token)
            user = await get_current_active_user(token_data, next(get_session()))
            if not user:
                response = RedirectResponse(url="/")
                response.headers["X-Request-ID"] = request_id
                response.delete_cookie(key=auth_settings.COOKIE_NAME)
                return response
        except Exception as e:
            response = RedirectResponse(url="/")
            response.headers["X-Request-ID"] = request_id
            response.delete_cookie(key=auth_settings.COOKIE_NAME)
            return response
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.on_event("startup")
def on_startup():
    logging.config.dictConfig(LOGGING_CONFIG)
    load_dotenv()
    init_db()
    ModelLoader().init()
    tg_bot.setup()
   bot_thread = threading.Thread(target=tg_bot.start_polling, daemon=True)
   bot_thread.start()


if __name__ == "__main__":
    HOST = os.getenv('APP_HOST')
    PORT = int(os.getenv('APP_PORT'))

    uvicorn.run('main:app', host=HOST, port=PORT, reload=True)
