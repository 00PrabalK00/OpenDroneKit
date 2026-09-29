"""The browser Hub signs a user in and talks to the API from a different loopback origin.

Two things kept the Hub from working at all: it had no sign-in form (a user had to
curl /auth/login for a token), and the API refused the CORS preflight from the
desktop shell's 127.0.0.1:<ephemeral port> origin. This drives hub.html in a real
headless Chromium over the DevTools protocol, against the real API under uvicorn on a
second loopback port, so the preflight is the browser's own.
"""

from __future__ import annotations

import functools
import json
import subprocess
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("email_validator")
websockets_sync = pytest.importorskip("websockets.sync.client")

from browser_evidence import SANDBOX_FLAGS, STARTUP_FLAGS, chromium  # noqa: E402
from hub_api_helpers import configure, reset_engine  # noqa: E402
from live_api_server import live_api  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "app" / "web"


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        return


@pytest.fixture
def hub_page():
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/hub.html"
    server.shutdown()
    server.server_close()


class _Page:
    def __init__(self, socket):
        self.socket = socket
        self.next_id = 0

    def send(self, method: str, **params):
        self.next_id += 1
        self.socket.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv(timeout=30))
            if message.get("id") == self.next_id:
                return message.get("result", {})

    def js(self, expression: str):
        result = self.send("Runtime.evaluate", expression=expression, awaitPromise=True,
                           returnByValue=True)
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.js(expression)
            if value:
                return value
            time.sleep(0.2)
        shown = self.js("document.querySelector('#hub-status').textContent")
        raise AssertionError(f"timed out waiting for {expression}; status: {shown}")


@pytest.fixture
def browser(tmp_path, hub_page):
    profile = tmp_path / "profile"
    process = subprocess.Popen(
        [str(chromium("the Hub sign-in test")), "--headless=new", *SANDBOX_FLAGS,
         *STARTUP_FLAGS, f"--user-data-dir={profile}", "--remote-debugging-port=0",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while not port_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        port = int(port_file.read_text().splitlines()[0])
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list").read())
        target = next(t for t in targets if t["type"] == "page")
        with websockets_sync.connect(target["webSocketDebuggerUrl"], max_size=None) as socket:
            page = _Page(socket)
            page.send("Page.enable")
            page.send("Page.navigate", url=hub_page)
            page.wait_for("document.readyState === 'complete' && !!document.querySelector('#hub-login-form')")
            yield page
    finally:
        process.kill()
        process.wait(timeout=10)


@pytest.fixture
def api(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    monkeypatch.delenv("ODK_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("ODK_CORS_LOOPBACK", raising=False)
    with live_api() as base:
        yield base
    reset_engine()


def _fill(page, api_url: str, email: str, password: str, organization: str = "") -> None:
    page.js(f"""(() => {{
        const set = (id, v) => {{ const el = document.querySelector(id); el.value = v; }};
        set('#hub-api-url', {json.dumps(api_url)});
        set('#hub-email', {json.dumps(email)});
        set('#hub-password', {json.dumps(password)});
        set('#hub-new-org', {json.dumps(organization)});
        set('#hub-org-id', '');
        return true;
    }})()""")


def test_create_account_then_sign_in_from_the_hub(browser, api):
    page = browser
    _fill(page, api, "pilot@example.com", "longenough1", "Hub Survey Co")
    page.js("document.querySelector('#hub-sign-up').click(), true")
    assert page.wait_for(
        "document.querySelector('#hub-connection-state').textContent === 'connected'")
    assert "Connected to the OpenDroneKit API" in page.js(
        "document.querySelector('#hub-status').textContent")
    assert page.js("document.querySelector('#hub-token').value.length > 20")
    assert page.js("document.querySelector('#hub-password').value") == ""
    assert page.js("document.querySelector('#metric-projects').textContent") == "0"

    # Sign in again from a fresh page with the account just created.
    page.send("Page.reload")
    page.wait_for("document.readyState === 'complete' && !!document.querySelector('#hub-login-form')")
    _fill(page, api, "pilot@example.com", "longenough1")
    page.js("document.querySelector('#hub-login-form').requestSubmit(), true")
    assert page.wait_for(
        "document.querySelector('#hub-connection-state').textContent === 'connected'")
    assert int(page.js("document.querySelector('#hub-org-id').value")) >= 1


def test_a_wrong_password_is_reported_not_swallowed(browser, api):
    page = browser
    _fill(page, api, "pilot@example.com", "longenough1", "Hub Survey Co")
    page.js("document.querySelector('#hub-sign-up').click(), true")
    page.wait_for("document.querySelector('#hub-connection-state').textContent === 'connected'")
    page.send("Page.reload")
    page.wait_for("document.readyState === 'complete' && !!document.querySelector('#hub-login-form')")
    _fill(page, api, "pilot@example.com", "wrong-password")
    page.js("document.querySelector('#hub-login-form').requestSubmit(), true")
    message = page.wait_for(
        "(t => t.includes('Sign-in failed') ? t : '')(document.querySelector('#hub-status').textContent)")
    assert "Incorrect email or password" in message
    assert page.js("document.querySelector('#hub-connection-state').textContent") == "not connected"


def test_the_api_answers_a_loopback_preflight_and_not_a_foreign_one(api):
    def preflight(origin: str):
        request = urllib.request.Request(api + "/auth/login", method="OPTIONS", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization"})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.headers.get("access-control-allow-origin")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("access-control-allow-origin")

    assert preflight("http://127.0.0.1:54321") == (200, "http://127.0.0.1:54321")
    assert preflight("http://localhost:3000") == (200, "http://localhost:3000")
    assert preflight("https://evil.example.com")[1] is None
