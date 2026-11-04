from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_run_qa_agent():
    r = client.post("/api/v1/agents/qa/run", json={"input": "hello"})
    assert r.status_code == 200
    assert r.json()["agent"] == "qa"
