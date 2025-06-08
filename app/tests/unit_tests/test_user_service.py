import uuid
import pytest
from datetime import datetime

from fastapi import HTTPException
from sqlmodel import Session, select

from app.src.service.crud.user_service import (
    get_all_users,
    get_user_by_id,
    get_user_by_email,
    create_user,
    add_balance,
    withdraw_balance,
    get_balance_histories,
    hash_password,
    verify_password,
    find_and_verify_user,
)

from entities.user.user import User
from entities.user.balance_history import BalanceHistory


def test_create_and_get_user(session: Session):
    user = User(
        email="test@example.com",
        name="John",
        surname="Doe",
        hashed_password=hash_password("secret"),
    )
    create_user(user, session)

    users = get_all_users(session)
    assert len(users) == 1
    assert users[0].email == "test@example.com"

    fetched_user = get_user_by_id(user.id, session)
    assert fetched_user is not None
    assert fetched_user.id == user.id

    fetched_by_email = get_user_by_email("test@example.com", session)
    assert fetched_by_email is not None
    assert fetched_by_email.email == "test@example.com"


def test_add_balance_success(session: Session):
    user = User(
        email="balance@example.com",
        name="Alice",
        surname="Smith",
        hashed_password=hash_password("secret"),
        balance=100.0,
    )
    create_user(user, session)

    add_balance("balance@example.com", 50.0, session)
    refreshed_user = get_user_by_email("balance@example.com", session)
    assert refreshed_user.balance == 150.0

    statement = select(BalanceHistory).where(BalanceHistory.user_id == user.id)
    histories = session.exec(statement).all()
    assert len(histories) == 1
    history = histories[0]
    assert history.amount_before_change == 100.0
    assert history.amount_change == 50.0


def test_add_balance_negative_amount(session: Session):
    user = User(
        email="negative@example.com",
        name="Negative",
        surname="Tester",
        hashed_password=hash_password("secret"),
        balance=100.0,
    )
    create_user(user, session)

    with pytest.raises(Exception, match="Amount must be positive"):
        add_balance("negative@example.com", -10.0, session)


def test_add_balance_user_not_found(session: Session):
    with pytest.raises(Exception, match="User not found"):
        add_balance("nonexistent@example.com", 10.0, session)


def test_withdraw_balance_success(session: Session):
    user = User(
        email="withdraw@example.com",
        name="Withdraw",
        surname="Tester",
        hashed_password=hash_password("secret"),
        balance=100.0,
    )
    create_user(user, session)

    withdraw_balance(user.id, 40.0, session)
    refreshed_user = get_user_by_id(user.id, session)
    assert refreshed_user.balance == 60.0

    statement = select(BalanceHistory).where(BalanceHistory.user_id == user.id)
    histories = session.exec(statement).all()
    assert len(histories) == 1
    history = histories[0]
    assert history.amount_before_change == 100.0
    assert history.amount_change == -40.0


def test_withdraw_balance_negative_amount(session: Session):
    user = User(
        email="withdraw_negative@example.com",
        name="WithdrawNeg",
        surname="Tester",
        hashed_password=hash_password("secret"),
        balance=100.0,
    )
    create_user(user, session)

    with pytest.raises(Exception, match="Amount must be positive"):
        withdraw_balance(user.id, -20.0, session)


def test_withdraw_balance_user_not_found(session: Session):
    fake_user_id = uuid.uuid4()
    with pytest.raises(HTTPException) as exc_info:
        withdraw_balance(fake_user_id, 10.0, session)
    assert exc_info.value.status_code == 404
    assert "User not found" in exc_info.value.detail


def test_withdraw_balance_insufficient_funds(session: Session):
    user = User(
        email="insufficient@example.com",
        name="Insufficient",
        surname="Tester",
        hashed_password=hash_password("secret"),
        balance=30.0,
    )
    create_user(user, session)

    with pytest.raises(HTTPException) as exc_info:
        withdraw_balance(user.id, 50.0, session)
    assert exc_info.value.status_code == 400
    assert "Insufficient balance" in exc_info.value.detail


def test_get_balance_histories_order(session: Session):
    user = User(
        email="history@example.com",
        name="History",
        surname="Tester",
        hashed_password=hash_password("secret"),
        balance=0.0,
    )
    create_user(user, session)

    history1 = BalanceHistory(
        user_id=user.id,
        amount_before_change=0.0,
        amount_change=100.0,
        timestamp=datetime(2023, 1, 1, 12, 0, 0),
    )
    history2 = BalanceHistory(
        user_id=user.id,
        amount_before_change=100.0,
        amount_change=-50.0,
        timestamp=datetime(2023, 1, 2, 12, 0, 0),
    )
    session.add(history1)
    session.add(history2)
    session.commit()

    histories = get_balance_histories(user.id, session)
    assert len(histories) == 2
    assert histories[0].timestamp > histories[1].timestamp


def test_hash_and_verify_password():
    password = "mysecretpassword"
    hashed = hash_password(password)
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_find_and_verify_user_success(session: Session):
    plain_password = "validpass"
    user = User(
        email="find@example.com",
        name="Find",
        surname="Tester",
        hashed_password=hash_password(plain_password),
        balance=0.0,
    )
    create_user(user, session)

    found_user = find_and_verify_user("find@example.com", plain_password, session)
    assert found_user.id == user.id


def test_find_and_verify_user_invalid_email(session: Session):
    with pytest.raises(HTTPException) as exc_info:
        find_and_verify_user("nonexistent@example.com", "any", session)
    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.detail


def test_find_and_verify_user_invalid_password(session: Session):
    correct_password = "correctpass"
    user = User(
        email="invalidpass@example.com",
        name="Invalid",
        surname="Password",
        hashed_password=hash_password(correct_password),
        balance=0.0,
    )
    create_user(user, session)

    with pytest.raises(HTTPException) as exc_info:
        find_and_verify_user("invalidpass@example.com", "wrongpass", session)
    assert exc_info.value.status_code == 401
    assert "Invalid credentials" in exc_info.value.detail
