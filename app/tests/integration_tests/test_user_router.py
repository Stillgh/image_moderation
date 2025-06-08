from sqlmodel import Session
from starlette import status

from app.tests.integration_tests.conftest import test_engine
from config.auth_config import get_auth_settings
from service.crud.user_service import get_user_by_email


def get_user(email: str):
    with Session(test_engine) as session:
        return get_user_by_email(email, session)


def test_login_success(client):
    data = {
        "username": "default@example.com",
        "password": "defaultpassword"
    }
    response = client.post("/users/login", data=data, follow_redirects=False)
    assert response.status_code == status.HTTP_302_FOUND
    assert response.headers['location'] == '/home'
    assert response.cookies.get(get_auth_settings().COOKIE_NAME).strip('"').startswith("Bearer")


def test_login_fail(client):
    data = {
        "username": "default@example.com",
        "password": "wrongpwd"
    }
    response = client.post("/users/login", data=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_token(patched_client):
    data = {
        "username": "default@example.com",
        "password": "defaultpassword"
    }

    response = patched_client.post("/users/token", data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert get_auth_settings().COOKIE_NAME in json_data
    assert json_data.get("token_type") == "bearer"


def test_signup_success(client):
    data = {
        "username": "New",
        "surname": "User",
        "email": "newuser@example.com",
        "password": "newpassword"
    }
    response = client.post("/users/signup", data=data, follow_redirects=False)
    assert response.status_code == status.HTTP_302_FOUND
    assert response.headers["location"] == "/home"
    cookie_value = response.cookies.get(get_auth_settings().COOKIE_NAME)
    assert cookie_value is not None
    assert cookie_value.strip('"').startswith("Bearer ")


def test_add_balance(patched_client, default_user):
    cur_balance = get_user(default_user.email).balance
    amount = 100.0
    response = patched_client.post("/users/balance/add", json={"amount": amount})
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data
    assert "new_balance" in json_data
    assert json_data["new_balance"] == amount + cur_balance


def test_withdraw_balance(patched_client, default_user):
    # Test withdrawing balance: first add then withdraw.
    add_amount = 150.0
    withdraw_amount = 50.0
    response_add = patched_client.post("/users/balance/add", json={"amount": add_amount})
    assert response_add.status_code == 200
    cur_amount = get_user(default_user.email).balance
    response_withdraw = patched_client.post("/users/balance/withdraw", json={"amount": withdraw_amount})
    assert response_withdraw.status_code == 200
    json_data = response_withdraw.json()
    assert "message" in json_data
    assert "new_balance" in json_data
    expected_balance = cur_amount - withdraw_amount
    assert json_data["new_balance"] == expected_balance


def test_get_current_balance(patched_client):
    response = patched_client.get("/users/balance/current")
    assert response.status_code == 200
    try:
        balance = float(response.json())
    except Exception:
        balance = None
    assert balance is not None


def test_balance_history(patched_client):
    response = patched_client.get("/users/balance/history")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type


def test_my_info(patched_client):
    response = patched_client.get("/users/myinfo")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
    assert "Default" in response.text
    assert "default@example.com" in response.text
