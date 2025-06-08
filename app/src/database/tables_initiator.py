from sqlmodel import SQLModel, Session

from database.database import get_engine

import uuid

from fastapi import HTTPException
from entities.user.user import User
from entities.user.user_role import UserRole
from service.auth.auth_service import get_password_hash
from service.crud.model_service import create_and_save_default_model
from service.crud.user_service import get_user_by_email, create_user


def create_admin_user(session: Session) -> None:
    admin_email = "admin@example.com"

    admin = get_user_by_email(admin_email, session)
    if admin:
        return

    admin = User(
        id=uuid.uuid4(),
        email=admin_email,
        name="Admin",
        surname="Admin",
        hashed_password=get_password_hash("admin123"),
        balance=100_000_000.0,
        role=UserRole.ADMIN
    )

    try:
        create_user(admin, session)
        print("Admin user created successfully")
    except Exception as e:
        print(f"Error creating admin user: {e}")
        raise HTTPException(status_code=500, detail="Could not create admin user")


def create_default_model(session: Session):
    model = create_and_save_default_model()
    session.add(model)
    session.commit()
    session.refresh(model)


def init_db():
    SQLModel.metadata.drop_all(get_engine())
    SQLModel.metadata.create_all(get_engine())
    with Session(get_engine()) as session:
        create_admin_user(session)
        create_default_model(session)
