from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    ALGORITHM: Optional[str] = None
    SECRET_KEY: Optional[str] = None
    ACCESS_TOKEN_EXPIRE_MINUTES: Optional[int] = None
    COOKIE_NAME: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
