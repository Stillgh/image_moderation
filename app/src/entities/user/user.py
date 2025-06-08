import uuid

from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field

from entities.user.user_role import UserRole


class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str
    surname: str
    hashed_password: str
    balance: float = Field(default=0.0)
    role: UserRole = Field(default='USER')
    disabled: bool = Field(default=False)


class UserDTO(BaseModel):
    email: str
    name: str
    surname: str
    balance: float
    role: UserRole


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserSignUp(BaseModel):
    email: EmailStr
    name: str
    surname: str
    password: str
