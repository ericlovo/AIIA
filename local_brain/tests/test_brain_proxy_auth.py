"""Every Command Center call to an authenticated Brain route must carry the API key.

The Brain enforces ``verify_api_key`` on most ``/v1/aiia/*`` routes. A proxy that
omits the header does not fail loudly: the handler turns 401 into an empty list,
a silent ``{"status": "unknown"}``, or a swallowed fire-and-forget. These tests
read both modules and assert the invariant statically, so a new proxy cannot
reintroduce the outage without failing here.
"""

import ast
from pathlib import Path

import pytest

BRAIN = Path(__file__).resolve().parents[1] / "local_api.py"
COMMAND_CENTER = Path(__file__).resolve().parents[1] / "command_center" / "server.py"
HTTP_METHODS = {"get", "post", "delete", "put", "patch", "stream"}


def _decorator_paths(tree):
    """Map each Brain route path to whether it requires the API key."""
    routes = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in HTTP_METHODS or not decorator.args:
                continue
            path = decorator.args[0]
            if not isinstance(path, ast.Constant) or not isinstance(path.value, str):
                continue
            guarded = "verify_api_key" in ast.unparse(decorator)
            routes[path.value] = routes.get(path.value, False) or guarded
    return routes


@pytest.fixture(scope="module")
def keyed_routes():
    routes = _decorator_paths(ast.parse(BRAIN.read_text()))
    keyed = sorted(path for path, guarded in routes.items() if guarded)
    assert "/v1/aiia/memory" in keyed, "expected the memory route to require the key"
    assert "/v1/aiia/remember" in keyed, "expected the remember route to require the key"
    return keyed


def _route_prefix(path: str) -> str:
    """Strip path parameters so /v1/aiia/memory/{id} matches its route."""
    return path.split("{")[0].rstrip("/")


def _brain_calls():
    """Yield (line, url_text, keywords) for every outbound httpx call in the server."""
    tree = ast.parse(COMMAND_CENTER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in HTTP_METHODS or not node.args:
            continue
        # client.stream("POST", url, ...) puts the URL second.
        url_node = (
            node.args[1] if node.func.attr == "stream" and len(node.args) > 1 else node.args[0]
        )
        url = ast.unparse(url_node)
        if "AIIA_BASE_URL" not in url and "localhost:8100" not in url and "AIIA_ASK_URL" not in url:
            continue
        yield node.lineno, url, {kw.arg for kw in node.keywords}


def _targets(url: str, keyed: list[str]) -> bool:
    """True when this call URL resolves to a Brain route that requires the key."""
    if "AIIA_ASK_URL" in url:
        # AIIA_ASK_URL is /v1/aiia/ask; the stream variant appends /stream.
        return True
    for route in keyed:
        if _route_prefix(route) in url:
            return True
    return False


def test_every_authenticated_brain_call_sends_the_key(keyed_routes):
    missing = [
        (line, url)
        for line, url, keywords in _brain_calls()
        if _targets(url, keyed_routes) and "headers" not in keywords
    ]
    assert not missing, "Brain calls missing headers=AIIA_HEADERS: " + "; ".join(
        f"line {line}: {url}" for line, url in missing
    )


def test_shared_client_factory_carries_the_key():
    """get_aiia_client() is the shared pooled client; it must default to authenticated."""
    tree = ast.parse(COMMAND_CENTER.read_text())
    factories = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        and node.name == "get_aiia_client"
    ]
    assert factories, "expected a get_aiia_client factory"
    assert "AIIA_HEADERS" in ast.unparse(factories[0])


def test_headers_constant_is_empty_without_a_configured_key(monkeypatch):
    """Local development without a key must not send a bogus x-api-key header."""
    source = COMMAND_CENTER.read_text()
    start = source.index("AIIA_HEADERS = (")
    snippet = source[start : source.index("\n\n", start)]
    for value, expected in (("", {}), ("secret", {"x-api-key": "secret"})):
        monkeypatch.setenv("LOCAL_BRAIN_API_KEY", value)
        namespace: dict = {"os": __import__("os")}
        exec(snippet, namespace)
        assert namespace["AIIA_HEADERS"] == expected


def test_detects_a_regression(keyed_routes, tmp_path, monkeypatch):
    """The check must actually fail when a proxy forgets the header."""
    broken = tmp_path / "server.py"
    broken.write_text(
        "import httpx\n"
        'AIIA_BASE_URL = "http://localhost:8100"\n'
        "async def leak():\n"
        "    async with httpx.AsyncClient() as client:\n"
        '        await client.get(f"{AIIA_BASE_URL}/v1/aiia/memory?category=wip")\n'
    )
    monkeypatch.setattr(
        "local_brain.tests.test_brain_proxy_auth.COMMAND_CENTER", broken, raising=False
    )
    missing = [
        (line, url)
        for line, url, keywords in _brain_calls()
        if _targets(url, keyed_routes) and "headers" not in keywords
    ]
    assert missing, "the static check failed to notice a proxy without headers"
