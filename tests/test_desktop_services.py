"""What the desktop shell serves besides the page: offline tiles, the Hub, the UI choice.

- The "Offline cache" basemap loaded tiles from the Hub service on port 8000, which the
  launcher never started, so on a disconnected laptop the map was blank. The shell now
  serves the cache from its own loopback origin, and the Api can fill it.
- The toolbar's Hub link opened a page whose REST service nothing started. The shell
  runs it in-process on loopback and publishes where in odk-runtime.json.
- ODK_UI=classic was documented and meant nothing; unknown values fell back silently.

The only fake here is the external tile provider (a local HTTP server standing in for
the internet); everything on the application side is the real code path.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "app" / "web"

# A real 1x1 PNG, different from the transparent "not cached" answer.
TILE = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415408d763f8cfc0000003010100c9fe92ef0000000049454e44ae426082"
)


class _Provider(BaseHTTPRequestHandler):
    """Stands in for the internet tile server: serves every z/x/y as TILE."""

    requests: list[str] = []
    body: bytes = TILE

    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        type(self).requests.append(self.path)
        self.send_response(200)
        body = type(self).body
        self.send_header("Content-Type", "image/jpeg" if body[:2] == b"\xff\xd8" else "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def provider(monkeypatch):
    from core import tile_cache

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setitem(tile_cache.PROVIDERS, "local-test", {
        "url": f"http://127.0.0.1:{server.server_port}/{{z}}/{{x}}/{{y}}.png",
        "max_zoom": 3, "attribution": "local test source",
    })
    _Provider.requests = []
    _Provider.body = TILE
    yield server
    server.shutdown()
    server.server_close()


def _get(url: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def _wait_job(api, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = api.job_status(job_id)["job"]
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


class TestOfflineBasemap:
    def test_cached_area_is_served_by_the_shell_with_the_provider_gone(self, provider, tmp_path):
        from app.api import Api
        from app.session import AppSession
        from app.shell import WEB_ROOT, Shell

        api = Api(AppSession())
        started = api.cache_basemap_area(-10, -10, 10, 10, 0, 1, "local-test")
        assert started["ok"], started
        job = _wait_job(api, started["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["fetched"] == started["tiles"] and job["result"]["failed"] == 0

        status = api.basemap_cache_status()
        assert status["providers"]["local-test"]["tiles"] == started["tiles"]
        assert status["providers"]["local-test"]["zoom_levels"] == [0, 1]

        provider.shutdown()  # from here, nothing can come from "the internet"

        url = Shell._serve(WEB_ROOT, "index.html", runtime={})
        origin = url.rsplit("/", 1)[0]
        code, headers, body = _get(f"{origin}/tiles/0/0/0.png?provider=local-test")
        assert code == 200 and body == TILE
        assert headers.get("X-ODK-Tile") == "cached"

        # Never cached: an honest gap, not a neighbouring tile.
        code, headers, body = _get(f"{origin}/tiles/3/7/7.png?provider=local-test")
        assert code == 404 and headers.get("X-ODK-Tile") == "not-cached" and body != TILE

        # The page itself is still served.
        code, _, body = _get(url)
        assert code == 200 and b"basemap-select" in body

    def test_cached_imagery_is_served_as_what_it_is(self, provider):
        """Imagery providers send JPEG; the cache stores it as <y>.png and both servers
        used to label it image/png. The bytes' real type is what goes on the wire."""
        import io

        from PIL import Image

        from app.api import Api
        from app.session import AppSession
        from app.shell import WEB_ROOT, Shell

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (40, 90, 30)).save(buffer, "JPEG")
        _Provider.body = buffer.getvalue()

        api = Api(AppSession())
        started = api.cache_basemap_area(-10, -10, 10, 10, 0, 0, "local-test")
        assert _wait_job(api, started["job_id"])["status"] == "done"
        provider.shutdown()

        origin = Shell._serve(WEB_ROOT, "index.html", runtime={}).rsplit("/", 1)[0]
        code, headers, body = _get(f"{origin}/tiles/0/0/0.png?provider=local-test")
        assert code == 200 and body == _Provider.body
        assert headers.get("Content-Type") == "image/jpeg"

        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from services.api.main import app as hub_app

        answer = TestClient(hub_app).get("/tiles/0/0/0.png", params={"provider": "local-test"})
        assert answer.status_code == 200 and answer.content == _Provider.body
        assert answer.headers["content-type"] == "image/jpeg"

    def test_caching_refuses_what_it_cannot_honour(self, provider):
        from app.api import Api
        from app.session import AppSession

        api = Api(AppSession())
        too_deep = api.cache_basemap_area(-1, -1, 1, 1, 0, 9, "local-test")
        assert too_deep["ok"] is False and "zoom 3" in too_deep["error"]
        unknown = api.cache_basemap_area(-1, -1, 1, 1, 0, 1, "nope")
        assert unknown["ok"] is False and "Unknown provider" in unknown["error"]
        assert _Provider.requests == []

    def test_the_offline_basemap_points_at_the_page_origin(self):
        source = (WEB / "js" / "map.js").read_text(encoding="utf-8")
        assert "'http://127.0.0.1:8000/tiles/{z}/{x}/{y}.png';" not in source
        assert "location.origin" in source
        assert "/tiles/{z}/{x}/{y}.png?provider=satellite" in source
        # And it falls back to the cache when the online basemap cannot be reached.
        assert "setBasemap('offline')" in source
        assert "onBasemapFallback" in (WEB / "js" / "app.js").read_text(encoding="utf-8")


class TestHubService:
    def test_the_shell_starts_the_hub_and_publishes_where(self, tmp_path):
        pytest.importorskip("uvicorn")
        pytest.importorskip("fastapi")
        import services.api.main  # noqa: F401 - already imported, as anything in-process may
        from app.shell import WEB_ROOT, Shell, _free_port, start_hub_service

        runtime: dict = {}
        page = Shell._serve(WEB_ROOT, "hub.html", runtime=runtime)
        ui_origin = page.rsplit("/", 1)[0]
        start_hub_service(runtime, ui_origin=ui_origin, port=_free_port(0))

        deadline = time.time() + 60
        while runtime.get("hub_status") == "starting" and time.time() < deadline:
            time.sleep(0.1)
        assert runtime["hub_status"] == "running", runtime

        # The page learns the address from the shell...
        code, _, body = _get(f"{ui_origin}/odk-runtime.json")
        published = json.loads(body)
        assert code == 200 and published["hub_api"] == runtime["hub_api"]

        # ...and the service answers a cross-origin request from that page.
        code, headers, body = _get(f"{runtime['hub_api']}/health", {"Origin": ui_origin})
        assert code == 200 and json.loads(body)["status"] in {"ok", "degraded"}
        allowed = {k.lower(): v for k, v in headers.items()}.get("access-control-allow-origin")
        assert allowed == ui_origin

        code, _, body = _get(f"{runtime['hub_api']}/tiles/status")
        assert code == 200 and "total_tiles" in json.loads(body)

        # What the page's Sign in... dialog does: create an account, then find its org.
        register = urllib.request.Request(
            f"{runtime['hub_api']}/auth/register", method="POST",
            data=json.dumps({"email": "pilot@example.org", "password": "correct horse 42",
                             "organization_name": "Field team"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": ui_origin},
        )
        with urllib.request.urlopen(register, timeout=10) as response:
            token = json.loads(response.read())["access_token"]
        code, _, body = _get(f"{runtime['hub_api']}/organizations",
                             {"Authorization": f"Bearer {token}", "Origin": ui_origin})
        assert code == 200 and [o["name"] for o in json.loads(body)] == ["Field team"]

    def test_the_hub_page_reads_the_published_address(self):
        source = (WEB / "js" / "hub.js").read_text(encoding="utf-8")
        assert "odk-runtime.json" in source
        assert "hub_error" in source
        # and can obtain a token itself instead of needing curl first
        assert "'/auth/register'" in source and "'/auth/login'" in source
        assert 'id="hub-signin"' in (WEB / "hub.html").read_text(encoding="utf-8")


class TestUiSelection:
    def test_classic_is_the_default_and_both_are_explicit(self, monkeypatch):
        from app.shell import UI_PAGES, resolve_ui

        monkeypatch.delenv("ODK_UI", raising=False)
        assert UI_PAGES[resolve_ui()] == "index.html"
        monkeypatch.setenv("ODK_UI", "classic")
        assert UI_PAGES[resolve_ui()] == "index.html"
        monkeypatch.setenv("ODK_UI", "Cockpit")
        assert UI_PAGES[resolve_ui()] == "workspace.html"
        assert UI_PAGES[resolve_ui("classic")] == "index.html"  # the flag wins
        monkeypatch.setenv("ODK_UI", "fancy")
        with pytest.raises(ValueError, match="classic, cockpit"):
            resolve_ui()

    def test_the_flag_and_the_variable_agree_on_case_and_blanks(self, monkeypatch):
        """ODK_UI=Cockpit opened the cockpit while --ui Cockpit was refused, and a blank
        ODK_UI (`set ODK_UI= ` in a batch file) was refused as a UI named ''. The flag is
        read the way the variable is now, and a blank variable means unset."""
        from app import shell

        opened = []

        class FakeShell:
            def run(self, **kwargs):
                opened.append(kwargs["ui"])

        monkeypatch.setattr(shell, "Shell", FakeShell)
        monkeypatch.setenv("ODK_UI", "   ")
        assert shell.resolve_ui() == "classic"
        assert shell.main([]) == 0
        assert shell.main(["--ui", "Cockpit"]) == 0
        monkeypatch.setenv("ODK_UI", "COCKPIT")
        assert shell.main([]) == 0
        assert shell.main(["--ui", "classic"]) == 0  # the flag wins over the variable
        assert opened == ["classic", "cockpit", "cockpit", "classic"]
        with pytest.raises(shell.UsageError, match="invalid choice"):
            shell.main(["--ui", "fancy"])

    @pytest.mark.skipif(__import__("os").environ.get("ODK_GUI_TESTS") != "1",
                        reason="opens real WebView2 windows; set ODK_GUI_TESTS=1 on a desktop")
    @pytest.mark.parametrize("how,page", [
        ({}, "index.html"),
        ({"ODK_UI": "classic"}, "index.html"),
        ({"ODK_UI": "cockpit"}, "workspace.html"),
        ({"argv": ["--ui", "cockpit"]}, "workspace.html"),
        ({"ODK_UI": "cockpit", "argv": ["--ui", "classic"]}, "index.html"),
    ], ids=["default", "env-classic", "env-cockpit", "flag-cockpit", "flag-wins"])
    def test_each_documented_way_opens_the_page_it_says(self, tmp_path, how, page):
        """Launch the real entry point under pythonw, the way OpenDroneKit.bat does, and
        ask the WebView2 window itself (DevTools protocol) which page it loaded."""
        import json
        import os
        import socket
        import subprocess
        import sys
        import time
        import urllib.request
        from pathlib import Path

        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if not pythonw.is_file():
            pytest.skip("no pythonw.exe beside this interpreter")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if k != "ODK_UI"}
        env.update(ODK_DATA_DIR=str(tmp_path / "data"), ODK_NO_DIALOG="1", ODK_HUB="0",
                   WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=f"--remote-debugging-port={port}")
        if "ODK_UI" in how:
            env["ODK_UI"] = how["ODK_UI"]
        process = subprocess.Popen([str(pythonw), str(REPO_ROOT / "main.py"), *how.get("argv", [])],
                                   cwd=str(tmp_path), env=env)
        try:
            url = ""
            deadline = time.time() + 90
            while time.time() < deadline and "127.0.0.1" not in url:
                assert process.poll() is None, "the app exited instead of opening a window"
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                        targets = json.loads(r.read().decode("utf-8"))
                    url = next((t["url"] for t in targets if t.get("type") == "page"), "")
                except OSError:
                    pass
                time.sleep(0.5)
            assert url.startswith("http://127.0.0.1:") and url.endswith("/" + page), url
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True)

    def test_the_command_line_refuses_an_unknown_ui(self):
        import subprocess
        import sys

        done = subprocess.run([sys.executable, str(REPO_ROOT / "main.py"), "--ui", "fancy"],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 2 and "invalid choice" in done.stderr

    def test_the_guide_says_what_opens(self):
        guide = (REPO_ROOT / "docs" / "UI_GUIDE.md").read_text(encoding="utf-8")
        assert "python main.py --ui cockpit" in guide
        assert "`python main.py` opens the classic" in guide
        assert "Case is ignored" in guide and "blank `ODK_UI` counts as unset" in guide

    def test_every_doc_names_the_same_default(self):
        """The README quick start said only `python main.py` and described the cockpit as
        marked "sample data in the status bar", a chip removed in favour of a banner."""
        from app.shell import DEFAULT_UI

        assert DEFAULT_UI == "classic"
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        install = (REPO_ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
        for text in (readme, install):
            assert "python main.py --ui cockpit" in text
            assert "sample data** in the status bar" not in text
        assert "ODK_UI=cockpit" in readme and "ODK_UI=cockpit" in install


class TestFirstRender:
    """Both UIs start correctly however the bridge and the page race."""

    def test_the_classic_ui_does_not_depend_on_winning_the_race(self):
        """pywebview fires pywebviewready once. With the network unreachable the page
        loaded late, the event had already fired, and the window sat on "Ready" with
        no map for the whole session. It now checks for the bridge as well."""
        source = (WEB / "js" / "app.js").read_text(encoding="utf-8")
        tail = source.split("let started = false;", 1)[1]
        assert "window.addEventListener('pywebviewready', start);" in tail
        assert "\n  start();\n" in tail
        assert "typeof window.pywebview.api.get_state !== 'function'" in tail

    def test_the_cockpit_treats_an_empty_api_object_as_not_ready(self):
        source = (WEB / "js" / "workspace" / "api.js").read_text(encoding="utf-8")
        assert 'typeof window.pywebview.api.get_state === "function"' in source

    def test_the_cockpit_re_renders_once_connected(self):
        source = (WEB / "js" / "workspace" / "shell.js").read_text(encoding="utf-8")
        connect = source.split("  async connect() {", 1)[1].split("\n  }\n", 1)[0]
        assert "this.open(this.workspaceId);" in connect
        assert "pywebviewready" in connect  # a late bridge still connects

    def test_the_cockpit_chrome_holds_no_invented_status(self):
        source = (WEB / "js" / "workspace" / "shell.js").read_text(encoding="utf-8")
        for literal in ('text: " 3 workers"', 'text: "PK"', "F-118 crack", "Roof Block A v3",
                        'el("span", { class: "mono", text: "EPSG:4326" })'):
            assert literal not in source, literal

    @pytest.mark.skipif(__import__("os").environ.get("ODK_GUI_TESTS") != "1",
                        reason="drives a real WebView2 window; set ODK_GUI_TESTS=1 on a desktop")
    def test_the_real_cockpit_shows_the_real_project_on_first_render(self):
        import webview

        from app.shell import WEB_ROOT, Shell

        shell = Shell()
        shell.api.create_project("Viaduct 3 survey", "", "")
        url = shell._serve(WEB_ROOT, "workspace.html", runtime=shell.runtime)
        window = webview.create_window("first render", url, js_api=shell.api, hidden=True)
        seen = {}

        def look():
            try:
                time.sleep(10)
                seen["nav"] = window.evaluate_js("document.querySelector('.nav-right').innerText")
                seen["status"] = window.evaluate_js("document.querySelector('.statusbar').innerText")
                seen["ws"] = window.evaluate_js("document.querySelector('.workspace').innerText")
            finally:
                window.destroy()

        webview.start(look)
        assert "Viaduct 3 survey" in seen["ws"], seen["ws"][:800]
        assert "No projects yet" not in seen["ws"]
        assert "Viaduct 3 survey" in seen["nav"] and "Viaduct 3 survey" in seen["status"]
        assert "3 workers" not in seen["nav"] and "Aukerman" not in seen["status"]
        assert "00:00:00" not in seen["nav"]

    def test_the_reports_workspace_holds_no_invented_figures(self):
        source = (WEB / "js" / "workspace" / "workspaces.js").read_text(encoding="utf-8")
        reports = source.split('  id: "reports",', 1)[1].split('  id: "developers",', 1)[0]
        code = "\n".join(line for line in reports.splitlines()
                         if not line.lstrip().startswith(("*", "//", "/*")))
        for literal in ('meta: "6"', '"DEMO organisation"', '"brand.png"'):
            assert literal not in code, literal

    @pytest.mark.skipif(__import__("os").environ.get("ODK_GUI_TESTS") != "1",
                        reason="drives a real WebView2 window; set ODK_GUI_TESTS=1 on a desktop")
    def test_the_real_reports_workspace_counts_the_projects_findings(self, tmp_path, monkeypatch):
        """Report Structure said "Inspection Findings 6" and Branding said "DEMO
        organisation" / "brand.png" over every project. Two findings are recorded here,
        one dismissed: the real cockpit must show 1 and leave branding for the operator."""
        import webview

        from app.shell import WEB_ROOT, Shell
        from core.annotations import create_annotation

        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "data"))
        shell = Shell()
        project = shell.api.create_project("Pier 4 reports", "", "")["project"]
        root = Path(shell.api._session.project_root())
        for status in ("open", "dismissed"):
            create_annotation(root, str(project["id"]), "image", "IMG_0001.JPG", "point",
                              {"type": "Point", "coordinates": [10, 20]}, "crack", "high", status)
        url = shell._serve(WEB_ROOT, "workspace.html", runtime=shell.runtime)
        window = webview.create_window("reports", url, js_api=shell.api, hidden=True)
        seen = {}

        def look():
            try:
                time.sleep(8)
                window.evaluate_js(
                    "[...document.querySelectorAll('.nav-item')]"
                    ".find((b) => b.textContent.trim() === 'Reports').click()")
                time.sleep(4)
                seen["structure"] = window.evaluate_js(
                    "document.querySelector('[data-panel-id=\"rp.structure\"]').innerText")
                seen["branding"] = window.evaluate_js(
                    "[...document.querySelectorAll('[data-panel-id=\"rp.branding\"] input')]"
                    ".map((i) => i.value).join('|')")
            finally:
                window.destroy()

        webview.start(look)
        lines = [line.strip() for line in seen["structure"].splitlines() if line.strip()]
        at = lines.index("Inspection Findings")
        assert lines[at + 1] == "1", lines
        assert "6" not in lines
        assert seen["branding"] == "|", seen["branding"]
