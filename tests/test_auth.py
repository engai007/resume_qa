def signup(client, email="jane@x.com", password="secret123", user_type="candidate"):
    return client.post("/api/v1/auth/signup", json={"email": email, "password": password, "user_type": user_type})


def login(client, email="jane@x.com", password="secret123"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def test_signup(client):
    r = signup(client)
    assert r.status_code == 201
    assert r.json() == {"id": 1, "email": "jane@x.com", "user_type": "candidate", "is_active": True}


def test_signup_duplicate_email(client):
    signup(client)
    assert signup(client, email="JANE@x.com").status_code == 409


def test_signup_unknown_user_type(client):
    assert signup(client, user_type="admin").status_code == 422


def test_login_and_me(client):
    signup(client, user_type="recruiter")
    token = login(client).json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["email"] == "jane@x.com" and me.json()["user_type"] == "recruiter"


def test_login_wrong_password(client):
    signup(client)
    assert login(client, password="wrong-pass").status_code == 401


def test_me_needs_valid_token(client):
    assert client.get("/api/v1/auth/me").status_code in (401, 403)
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
