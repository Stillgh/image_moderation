import uuid

from entities.user.user import UserDTO, User, UserSignUp
from entities.user.user_role import UserRole
from service.crud.user_service import hash_password


def user_to_user_dto(user: User) -> UserDTO:
    return UserDTO(
        email=user.email,
        name=user.name,
        surname=user.surname,
        balance=user.balance,
        role=user.role
    )


def user_signup_dto_to_user(user_signup_dto: UserSignUp) -> User:
    return User(
        id=uuid.uuid4(),
        email=user_signup_dto.email,
        name=user_signup_dto.name,
        surname=user_signup_dto.surname,
        hashed_password=hash_password(user_signup_dto.password),
        balance=0.0,
        role=UserRole.USER
    )
