

def test_admin_required_admin(admin_client):
    response = admin_client.get("/admin/admin_required")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
    assert "Welcome, admin" in response.text


def test_admin_required_not_admin(patched_client):
    response = patched_client.get("/admin/admin_required")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type
    assert "Welcome, admin" not in response.text
    assert "To access that page, you should be an admin." in response.text
