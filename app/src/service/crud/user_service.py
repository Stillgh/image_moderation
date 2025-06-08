import logging
import uuid
from typing import List
import bcrypt
from fastapi import HTTPException

from entities.user.user import User
from sqlmodel import Session, select

from entities.user.balance_history import BalanceHistory

logger = logging.getLogger(__name__)


def get_all_users(session: Session) -> List[User]:
    return session.query(User).all()


def get_user_by_id(id: uuid.UUID, session: Session) -> User:
    return session.get(User, id)


def get_user_by_email(email: str, session: Session) -> User:
    statement = select(User).where(User.email == email)
    user = session.exec(statement).first()
    return user


def create_user(new_user: User, session: Session) -> None:
    session.add(new_user)
    session.commit()
    session.refresh(new_user)


def add_balance(email: str, amount: float, session: Session) -> None:
    if amount <= 0:
        raise Exception("Amount must be positive")

    user = get_user_by_email(email, session)
    if not user:
        raise Exception("User not found")

    balance_history = BalanceHistory(user_id=user.id, amount_before_change=user.balance, amount_change=amount)

    user.balance += amount
    session.add(balance_history)
    session.commit()
    session.refresh(user)


def withdraw_balance(user_id: uuid.UUID, amount: float, session: Session) -> None:
    logger.info(f"Withdrawing {amount} from user {user_id}")
    if amount <= 0:
        logger.error(f"Amount must be positive {amount}, user {user_id}")
        raise Exception("Amount must be positive")

    user = get_user_by_id(user_id, session)
    if not user:
        logger.error(f"User {user_id} not found")
        raise HTTPException(status_code=404, detail=str("User not found"))

    if user.balance < amount:
        logger.error(f"Insufissient balance: {user.balance}, amount: {amount}, user: {user_id}")
        raise HTTPException(status_code=400, detail=str("Insufficient balance"))

    balance_history = BalanceHistory(user_id=user.id, amount_before_change=user.balance, amount_change=-amount)

    user.balance -= amount
    session.add(balance_history)
    session.commit()
    session.refresh(user)
    logger.info(f"Amount: {amount} withdrawed from user: {user_id}")


def get_balance_histories(user_id: uuid.UUID, session: Session) -> List[BalanceHistory]:

    statement = select(BalanceHistory) \
        .where(BalanceHistory.user_id == user_id) \
        .order_by(BalanceHistory.timestamp.desc())

    result = session.exec(statement).all()
    return result


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def find_and_verify_user(email: str, password: str, session: Session) -> User:
    user = get_user_by_email(email, session)
    if not user or not verify_password(password, user.hashed_password):
        logger.error(f"Invalid credentials for user {email}")

        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user
