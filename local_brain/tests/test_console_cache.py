"""The mutable HTML entrypoint must not retain an old asset manifest."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("entry", ["react", "legacy-console", "legacy-dashboard", "missing"])
def test_console_entrypoint_is_no_store_for_every_fallback(tmp_path, monkeypatch, entry):
    from local_brain.command_center import server

    react = tmp_path / "dist"
    static = tmp_path / "static"
    static.mkdir()
    monkeypatch.setattr(server, "REACT_DIST", react)
    monkeypatch.setattr(server, "STATIC_DIR", static)
    if entry == "react":
        react.mkdir()
        (react / "index.html").write_text("<html>current-react-shell</html>")
    elif entry == "legacy-console":
        (static / "console.html").write_text("<html>legacy-console-shell</html>")
    elif entry == "legacy-dashboard":
        (static / "dashboard.html").write_text("<html>legacy-dashboard-shell</html>")
    response = TestClient(server.app).get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"].startswith("text/html")
    expected = {
        "react": "current-react-shell",
        "legacy-console": "legacy-console-shell",
        "legacy-dashboard": "legacy-dashboard-shell",
        "missing": "No dashboard found.",
    }
    assert expected[entry] in response.text


def test_entrypoint_reads_current_build_without_conditional_old_shell(tmp_path, monkeypatch):
    from local_brain.command_center import server

    monkeypatch.setattr(server, "REACT_DIST", tmp_path)
    index = tmp_path / "index.html"
    index.write_text('<script src="/assets/old-hash.js"></script>')
    client = TestClient(server.app)
    assert "old-hash" in client.get("/").text
    index.write_text('<script src="/assets/new-hash.js"></script>')
    response = client.get("/", headers={"If-None-Match": "old-shell"})
    assert response.status_code == 200
    assert "new-hash" in response.text
    assert response.headers["cache-control"] == "no-store"
