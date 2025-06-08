import logging
from datetime import datetime, timezone, timedelta
import jwt
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError

from config.auth_config import get_auth_settings
from entities.auth.auth_entities import TokenData
from entities.user.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")
logger = logging.getLogger(__name__)


def create_access_token(user: User):
    logger.info(f"Creating access token for user: {user.email if hasattr(user, 'email') else user.id}")
    auth_settings = get_auth_settings()
    access_token_expires = timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = __create_access_token(
        data={"sub": str(user.id), "username": user.name},
        secret_key=auth_settings.SECRET_KEY,
        algorithm=auth_settings.ALGORITHM,
        expires_delta=access_token_expires
    )
    logger.info(f"Access token created for user: {user.email if hasattr(user, 'email') else user.id}")
    return token


def __create_access_token(data: dict, secret_key: str, algorithm: str, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    logger.info(f"JWT token encoded with expiration at {expire}")
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    try:
        auth_settings = get_auth_settings()
        token = token.removeprefix('Bearer ')
        payload = jwt.decode(token, auth_settings.SECRET_KEY, algorithms=[auth_settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.error("Token verification failed: 'sub' (user_id) is missing")
            raise InvalidTokenError("Missing user id in token")
        token_data = TokenData(user_id=user_id, username=payload.get("username"))
        logger.info(f"Token verified for user_id: {user_id}, username: {payload.get('username')}")
    except InvalidTokenError as e:
        logger.error(f"Token verification failed: {e}")
        raise InvalidTokenError(e)
    return token_data
