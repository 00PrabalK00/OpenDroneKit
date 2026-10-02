"""Installed-build runtime pieces: weights downloader, update check, GPU detection.

Downloads and release checks run against real local servers (file:// and a loopback
HTTP server), so the code under test makes real requests and hashes real bytes.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import gpu, model_download, updates


# -- fixtures -------------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """A registry with one model, served from a temporary models folder."""
    models = tmp_path / "models"
    models.mkdir()
    rows: dict = {}
    monkeypatch.setattr("core.models.models_root", lambda: models)
    monkeypatch.setattr("core.models.load_registry", lambda: {"version": 1, "models": rows})
    monkeypatch.delenv("ODK_MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv(model_download.SOURCES_ENV, raising=False)
    return SimpleNamespace(rows=rows, models=models, tmp=tmp_path)


class _Server:
    def __init__(self, routes: dict[str, tuple[int, bytes]]):
        self.routes = routes
        self.hits: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return

            def do_GET(self):  # noqa: N802
                outer.hits.append(self.path)
                status, body = outer.routes.get(self.path, (404, b'{"message":"Not Found"}'))
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture()
def server():
    created = []

    def make(routes):
        srv = _Server(routes)
        created.append(srv)
        return srv

    yield make
    for srv in created:
        srv.close()


# -- weights downloader ---------------------------------------------------------

class TestModelDownloadVerification:
    def test_no_published_url_is_stated_not_invented(self, registry):
        registry.rows["m"] = {"path": "x/m.onnx", "sha256": "a" * 64}
        info = model_download.availability("m")
        assert info["downloadable"] is False
        assert info["reason"] == model_download.NOT_PUBLISHED
        assert "urls" not in info
        result = model_download.download_model("m")
        assert result["ok"] is False and "not available for download yet" in result["error"]

    def test_matching_sha256_is_installed(self, registry):
        payload = b"onnx-bytes-" * 1000
        src = registry.tmp / "src.onnx"
        src.write_bytes(payload)
        registry.rows["m"] = {"path": "x/m.onnx", "sha256": _sha(payload),
                              "download": {"url": src.as_uri()}}
        result = model_download.download_model("m")
        assert result["ok"], result
        assert (registry.models / "x" / "m.onnx").read_bytes() == payload

    def test_mismatched_sha256_is_refused_and_discarded(self, registry):
        src = registry.tmp / "src.onnx"
        src.write_bytes(b"tampered")
        registry.rows["m"] = {"path": "x/m.onnx", "sha256": _sha(b"original"),
                              "download": {"url": src.as_uri()}}
        result = model_download.download_model("m")
        assert result["ok"] is False
        assert result["error"].startswith("refused:") and "does not match" in result["error"]
        folder = registry.models / "x"
        assert not (folder / "m.onnx").exists()
        assert list(folder.iterdir()) == [], "the rejected bytes must not be left behind"

    def test_no_recorded_digest_refuses_before_downloading(self, registry, server):
        srv = server({"/m.onnx": (200, b"x")})
        registry.rows["m"] = {"path": "m.onnx", "download": {"url": srv.base + "/m.onnx"}}
        info = model_download.availability("m")
        assert info["downloadable"] is False and "no sha256" in info["reason"]
        assert model_download.download_model("m")["ok"] is False
        assert srv.hits == [], "nothing may be fetched when it cannot be verified"

    def test_downloads_over_http(self, registry, server, monkeypatch):
        payload = bytes(range(256)) * 50
        srv = server({"/w.onnx": (200, payload)})
        registry.rows["m"] = {"path": "w.onnx", "sha256": _sha(payload)}
        sources = registry.tmp / "sources.json"
        sources.write_text(json.dumps({"sources": {"m": {"url": srv.base + "/w.onnx"}}}))
        monkeypatch.setenv(model_download.SOURCES_ENV, str(sources))
        result = model_download.download_model("m")
        assert result["ok"], result
        assert (registry.models / "w.onnx").read_bytes() == payload

    def test_disallowed_scheme_is_refused(self, registry):
        registry.rows["m"] = {"path": "m.onnx", "sha256": "a" * 64,
                              "download": {"url": "ftp://example.invalid/m.onnx"}}
        assert "not allowed" in model_download.availability("m")["reason"]

    def test_sidecar_files_each_verified_then_the_set(self, registry):
        from core.semantic_engine import sha256_onnx_model

        graph, data = b"graph" * 10, b"weights" * 100
        ref = registry.tmp / "ref"
        ref.mkdir()
        (ref / "s.onnx").write_bytes(graph)
        (ref / "s.onnx.data").write_bytes(data)
        expected = sha256_onnx_model(ref / "s.onnx")
        files = [{"name": "s.onnx", "url": (ref / "s.onnx").as_uri(), "sha256": _sha(graph)},
                 {"name": "s.onnx.data", "url": (ref / "s.onnx.data").as_uri(), "sha256": _sha(data)}]
        registry.rows["s"] = {"path": "s.onnx", "sha256": expected, "download": {"files": files}}
        result = model_download.download_model("s")
        assert result["ok"], result
        assert (registry.models / "s.onnx.data").read_bytes() == data

        files[1]["sha256"] = _sha(b"something else")
        (registry.models / "s.onnx").unlink()
        (registry.models / "s.onnx.data").unlink()
        bad = model_download.download_model("s")
        assert bad["ok"] is False and "s.onnx.data" in bad["error"]
        assert not (registry.models / "s.onnx").exists()

    def test_the_shipped_registry_publishes_no_invented_urls(self):
        """Weights are not hosted yet: no row may carry a download location."""
        rows = json.loads((Path(__file__).resolve().parents[1] / "models" / "model_registry.json").read_text(encoding="utf-8"))["models"]
        assert not [k for k, r in rows.items() if r.get("download")]
        report = model_download.first_run_report()
        assert not report["downloadable"]


# -- update check ---------------------------------------------------------------

class TestVersionComparison:
    @pytest.mark.parametrize("newer, older", [
        ("0.2.0", "0.1.0"), ("v1.0.0", "0.9.9"), ("1.10.0", "1.9.0"), ("1.0.0", "1.0.0-rc.2"),
        ("1.0.0-rc.2", "1.0.0-rc.1"), ("1.0.0-beta", "1.0.0-alpha"), ("1.0.1", "1.0"),
        ("2.0.0.post1", "2.0.0"), ("1.0.0rc1", "1.0.0b9"),
    ])
    def test_ordering(self, newer, older):
        assert updates.is_newer(newer, older)
        assert not updates.is_newer(older, newer)

    @pytest.mark.parametrize("a, b", [("1.0", "1.0.0"), ("v0.1.0", "0.1.0"), ("1.2.0+build5", "1.2.0")])
    def test_equal_is_not_newer(self, a, b):
        assert not updates.is_newer(a, b) and not updates.is_newer(b, a)

    @pytest.mark.parametrize("bad", ["nightly", "", "v", "1.0.0-weird"])
    def test_non_versions_are_rejected(self, bad):
        with pytest.raises(ValueError):
            updates.parse_version(bad)


def _release(base: str, tag: str, assets: list[dict]) -> bytes:
    return json.dumps({"tag_name": tag, "html_url": f"{base}/r/{tag}", "draft": False,
                       "prerelease": False, "assets": assets}).encode()


class TestUpdateCheck:
    def test_off_by_default_contacts_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        called = []
        monkeypatch.setattr(updates, "_get", lambda *a, **k: called.append(a) or b"{}")
        result = updates.check_for_update(current="0.1.0")
        assert result["checked"] is False and "opt-in" in result["reason"]
        assert called == []

    def test_newer_release_with_digest_is_offered_and_downloaded(self, server, tmp_path):
        installer = b"MZ fake installer bytes" * 100
        name = "OpenDroneKit-0.2.0-win64-setup-UNSIGNED.exe"
        routes = {}
        srv = server(routes)
        routes["/repos/o/r/releases/latest"] = (200, _release(srv.base, "v0.2.0", [
            {"name": name, "browser_download_url": f"{srv.base}/dl/{name}",
             "size": len(installer), "digest": f"sha256:{_sha(installer)}"},
            {"name": "SHA256SUMS.txt", "browser_download_url": f"{srv.base}/dl/SHA256SUMS.txt"}]))
        routes["/dl/SHA256SUMS.txt"] = (200, f"{_sha(installer)}  {name}\n".encode())
        routes[f"/dl/{name}"] = (200, installer)
        offer = updates.check_for_update(current="0.1.0", explicit=True, api_base=srv.base, repo="o/r")
        assert offer["update_available"] is True, offer
        assert offer["sha256"] == _sha(installer) and offer["unsigned"] is True
        got = updates.download_update(offer, dest_dir=tmp_path)
        assert got["ok"] and got["installed"] is False
        assert Path(got["path"]).read_bytes() == installer
        assert "never run automatically" in got["message"]

    def test_tampered_download_is_deleted(self, server, tmp_path):
        name = "OpenDroneKit-0.2.0-win64-setup.exe"
        routes = {}
        srv = server(routes)
        routes["/repos/o/r/releases/latest"] = (200, _release(srv.base, "0.2.0", [
            {"name": name, "browser_download_url": f"{srv.base}/dl/{name}",
             "digest": f"sha256:{_sha(b'published')}"}]))
        routes[f"/dl/{name}"] = (200, b"swapped on the wire")
        offer = updates.check_for_update(current="0.1.0", explicit=True, api_base=srv.base, repo="o/r")
        got = updates.download_update(offer, dest_dir=tmp_path)
        assert got["ok"] is False and "does not match" in got["error"]
        assert list(tmp_path.iterdir()) == []

    def test_release_without_sha256_is_not_offered(self, server):
        routes = {}
        srv = server(routes)
        routes["/repos/o/r/releases/latest"] = (200, _release(srv.base, "9.0.0", [
            {"name": "x-setup.exe", "browser_download_url": f"{srv.base}/x"}]))
        offer = updates.check_for_update(current="0.1.0", explicit=True, api_base=srv.base, repo="o/r")
        assert offer["update_available"] is False and "no sha256" in offer["reason"]

    def test_disagreeing_digests_refuse(self, server):
        routes = {}
        srv = server(routes)
        routes["/repos/o/r/releases/latest"] = (200, _release(srv.base, "9.0.0", [
            {"name": "x-setup.exe", "browser_download_url": f"{srv.base}/x", "digest": "sha256:" + "a" * 64},
            {"name": "SHA256SUMS", "browser_download_url": f"{srv.base}/sums"}]))
        routes["/sums"] = (200, ("b" * 64 + "  x-setup.exe\n").encode())
        offer = updates.check_for_update(current="0.1.0", explicit=True, api_base=srv.base, repo="o/r")
        assert offer["update_available"] is False and "disagrees" in offer["reason"]

    def test_same_or_older_is_up_to_date(self, server):
        routes = {}
        srv = server(routes)
        routes["/repos/o/r/releases/latest"] = (200, _release(srv.base, "v0.1.0", []))
        offer = updates.check_for_update(current="0.1.0", explicit=True, api_base=srv.base, repo="o/r")
        assert offer["update_available"] is False and "up to date" in offer["reason"]

    def test_no_release_yet(self, server):
        srv = server({})
        offer = updates.check_for_update(current="0.1.0", explicit=True, api_base=srv.base, repo="o/r")
        assert offer["checked"] is True and offer["update_available"] is False
        assert "no published release" in offer["reason"]

    def test_unreachable_is_reported_not_raised(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        offer = updates.check_for_update(current="0.1.0", explicit=True,
                                         api_base=f"http://127.0.0.1:{port}", repo="o/r", timeout=2)
        assert offer["checked"] is False and "Could not reach" in offer["reason"]

    def test_nothing_in_the_module_executes_the_installer(self):
        import inspect

        source = inspect.getsource(updates)
        for forbidden in ("subprocess", "os.startfile", "os.system", "Popen", "/VERYSILENT", "/SILENT"):
            assert forbidden not in source

    def test_sha256sums_parsing(self):
        text = f"{'c' * 64}  a.exe\n{'D' * 64} *b.exe\nnot a line\n"
        assert updates.parse_sha256sums(text) == {"a.exe": "c" * 64, "b.exe": "d" * 64}

    def test_opt_in_persists_through_the_api(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path / "data"))
        from app.api import Api

        api = Api()
        assert api.update_settings()["opt_in"] is False
        assert api.update_settings(True)["opt_in"] is True
        assert updates.opted_in() is True
        assert Api().update_settings()["opt_in"] is True


# -- GPU detection --------------------------------------------------------------

def _proc(stdout: str, code: int = 0):
    return SimpleNamespace(stdout=stdout, returncode=code)


class TestGpuDetection:
    def test_nvidia_smi_parsing(self):
        rows = gpu.parse_nvidia_smi("NVIDIA GeForce RTX 5060 Laptop GPU, 581.57, 8151\n")
        assert rows == [{"name": "NVIDIA GeForce RTX 5060 Laptop GPU", "vendor": "NVIDIA",
                         "driver": "581.57", "memory_mb": 8151, "source": "nvidia-smi"}]

    def test_video_controller_parsing(self):
        one = json.dumps({"Name": "Intel(R) Graphics", "DriverVersion": "32.0", "AdapterRAM": 4293918720})
        rows = gpu.parse_video_controllers(one)
        assert rows[0]["vendor"] == "Intel" and rows[0]["memory_is_floor"] is True
        assert gpu.parse_video_controllers("garbage") == []

    def test_present_but_unusable_is_said_plainly(self, monkeypatch):
        monkeypatch.setattr(gpu, "onnxruntime_providers", lambda: {
            "installed": True, "version": "1", "providers": ["CPUExecutionProvider"],
            "gpu_providers": [], "reason": ""})
        monkeypatch.setattr(gpu, "torch_cuda", lambda: {"installed": False, "cuda": False, "device": "", "reason": ""})

        def runner(cmd, timeout):
            if cmd[0] == "nvidia-smi":
                return _proc("NVIDIA RTX A, 1.0, 4096\n")
            return _proc(json.dumps([{"Name": "NVIDIA RTX A"}, {"Name": "Microsoft Basic Display Adapter"}]))

        result = gpu.detect_gpus(runner)
        assert result["gpu_present"] is True and result["gpu_usable_for_inference"] is False
        assert "CPU build" in result["summary"]
        assert [a["name"] for a in result["adapters"]].count("NVIDIA RTX A") == 1

    def test_no_gpu_and_no_tools(self, monkeypatch):
        monkeypatch.setattr(gpu, "onnxruntime_providers", lambda: {
            "installed": False, "version": "", "providers": [], "gpu_providers": [], "reason": "x"})
        monkeypatch.setattr(gpu, "torch_cuda", lambda: {"installed": False, "cuda": False, "device": "", "reason": ""})
        result = gpu.detect_gpus(lambda cmd, timeout: None)
        assert result["gpu_present"] is False and result["adapters"] == []
        assert "CPU" in result["summary"] and result["notes"]

    def test_this_machine_reports_through_capabilities(self, tmp_path, monkeypatch):
        """The real probe, through the real Environment Capabilities call."""
        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path))
        from app.api import Api

        caps = Api().capabilities()["capabilities"]
        detection = caps["gpu_detection"]
        assert set(detection) >= {"adapters", "onnxruntime", "gpu_present",
                                  "gpu_usable_for_inference", "summary"}
        assert caps["gpu_summary"] == detection["summary"]
        assert "CPUExecutionProvider" in detection["onnxruntime"]["providers"]
