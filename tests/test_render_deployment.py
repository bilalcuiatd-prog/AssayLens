import base64

from fastapi.testclient import TestClient
from render_app import app


def test_health_without_auth(monkeypatch):
    monkeypatch.delenv("ASSAYLENS_SITE_USERNAME", raising=False)
    monkeypatch.delenv("ASSAYLENS_SITE_PASSWORD", raising=False)
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/").status_code == 503


def test_authentication(monkeypatch):
    monkeypatch.setenv("ASSAYLENS_SITE_USERNAME", "scientist")
    monkeypatch.setenv("ASSAYLENS_SITE_PASSWORD", "test-secret")
    with TestClient(app) as c:
        assert c.get("/").status_code == 401
        wrong = base64.b64encode(b"scientist:wrong").decode()
        assert c.get("/", headers={"Authorization": "Basic " + wrong}).status_code == 401
        ok = base64.b64encode(b"scientist:test-secret").decode()
        assert c.get("/", headers={"Authorization": "Basic " + ok}).status_code == 200
        assert c.get("/assets/assaylens-logo-original.png", headers={"Authorization": "Basic " + ok}).status_code == 200
