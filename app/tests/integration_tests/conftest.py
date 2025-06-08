import os
import pytest
from celery.contrib.testing.worker import start_worker
from starlette.testclient import TestClient

from celery_worker import celery
from config.auth_config import get_auth_settings
from config.constants import DEFAULT_MODEL_NAME
from entities.user.user import UserSignUp
from entities.user.user_role import UserRole
from main import app
from service.auth.jwt_service import create_access_token
from service.crud.model_service import create_and_save_default_model, get_model_by_name
from service.crud.user_service import get_user_by_email, create_user
from service.mappers.user_mapper import user_signup_dto_to_user

import database.database
from sqlalchemy import StaticPool
from sqlmodel import create_engine, Session, SQLModel
from entities.auth.auth_entities import TokenData
from entities.user.user import User
from entities.user.balance_history import BalanceHistory

# os.environ["COOKIE_NAME"] = "user_token"
# os.environ["SECRET_KEY"] = '703d9589a9a161cecfce6fd1d3fd6d69293eefaeebc1e5244f49928f690aa189'
# os.environ["ALGORITHM"] = 'HS256'
# os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = '30'
# os.environ["CELERY_BROKER_URL"] = "memory://"
# os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
SQLModel.metadata.create_all(test_engine)


def override_get_session():
    with Session(test_engine) as session:
        yield session


app.dependency_overrides[database.database.get_session] = override_get_session


@pytest.fixture(name="client")
def client_fixture():
    return TestClient(app)


@pytest.fixture(autouse=True, scope="session")
def create_default_user():
    with Session(test_engine) as session:
        if not get_user_by_email("default@example.com", session):
            user_data = UserSignUp(
                name="Default",
                surname="User",
                email="default@example.com",
                password="defaultpassword"
            )
            user = user_signup_dto_to_user(user_data)
            create_user(user, session)
            session.commit()
    yield


@pytest.fixture
def default_user():
    with Session(test_engine) as session:
        user = get_user_by_email("default@example.com", session)
        yield user


@pytest.fixture
def token(default_user):
    return create_access_token(default_user)


@pytest.fixture
def patched_client(client, token, monkeypatch):
    monkeypatch.setattr("main.get_session", override_get_session)
    client.cookies.set(get_auth_settings().COOKIE_NAME, f"Bearer {token}")
    return client


@pytest.fixture(autouse=True, scope="session")
def create_admin():
    with Session(test_engine) as session:
        if not get_user_by_email("admin@example.com", session):
            user_data = UserSignUp(
                name="Admin",
                surname="User",
                email="admin@example.com",
                password="defaultpassword"
            )
            user = user_signup_dto_to_user(user_data)
            user.role = UserRole.ADMIN
            user.balance = 10_000
            create_user(user, session)
            session.commit()
    yield


@pytest.fixture
def admin():
    with Session(test_engine) as session:
        user = get_user_by_email("admin@example.com", session)
        yield user


@pytest.fixture
def admin_token(admin):
    return create_access_token(admin)


@pytest.fixture
def admin_client(admin_token, monkeypatch):
    adm_client = TestClient(app)
    monkeypatch.setattr("main.get_session", override_get_session)
    adm_client.cookies.set(get_auth_settings().COOKIE_NAME, f"Bearer {admin_token}")
    return adm_client


@pytest.fixture(autouse=True, scope="session")
def create_default_model():
    with Session(test_engine) as session:
        if not get_model_by_name(DEFAULT_MODEL_NAME, session):
            model = create_and_save_default_model()
            session.add(model)
            session.commit()
            session.refresh(model)
    yield


@pytest.fixture(autouse=True)
def patch_session_in_celery(monkeypatch):
    monkeypatch.setattr("celery_worker.get_session", override_get_session)


@pytest.fixture(scope="session", autouse=True)
def celery_worker_fixture():
    with start_worker(celery, loglevel="INFO", perform_ping_check=False, queues=["prediction"]) as worker:
        yield worker
