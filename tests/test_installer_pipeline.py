"""The installer pipeline's own logic: manifest, signing plan, runtime layout, lock, script.

The full build (download the runtime, install every wheel, compile with Inno Setup,
install, self-test, uninstall) takes many minutes and is run by
tools/installer/smoke_install.py; its evidence is recorded in docs/INSTALLER.md. These
tests pin the parts that decide whether that build is honest and reproducible.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools.installer import build, manifest, sign

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools" / "installer"


def _meta(**over):
    return {"app": "OpenDroneKit", "version": "0.1.0", "python": "3.11.9", "signed": False, **over}


class TestManifest:
    def test_round_trip_and_verification(self, tmp_path):
        (tmp_path / "odk").mkdir()
        (tmp_path / "odk" / "main.py").write_bytes(b"print(1)\n")
        (tmp_path / "runtime.dll").write_bytes(b"\x00" * 100)
        data = manifest.build_manifest(tmp_path, _meta())
        assert data["file_count"] == 2 and data["total_bytes"] == 109
        path = manifest.write_manifest(tmp_path, data)
        loaded = manifest.load_manifest(path)
        assert manifest.verify_tree(tmp_path, loaded) == []
        assert "install_manifest.json" not in loaded["files"], "the manifest cannot list itself"

    def test_changed_and_missing_files_are_reported(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        data = manifest.build_manifest(tmp_path, _meta())
        (tmp_path / "a.py").write_text("A")  # same size, different bytes
        (tmp_path / "b.py").unlink()
        (tmp_path / "log.txt").write_text("created after install")
        assert sorted(manifest.verify_tree(tmp_path, data)) == ["changed: a.py", "missing: b.py"]

    @pytest.mark.parametrize("rel", ["../evil.py", "/abs.py", "C:/x.py"])
    def test_unsafe_paths_are_rejected(self, rel):
        data = {"schema": 1, **_meta(), "files": {rel: {"sha256": "a" * 64, "size": 1}}}
        with pytest.raises(ValueError, match="unsafe"):
            manifest.validate_manifest(data)

    def test_transient_open_failure_is_retried_but_persistent_one_raises(self, tmp_path, monkeypatch):
        target = tmp_path / "f.bin"
        target.write_bytes(b"abc")
        real_open = Path.open
        calls = {"n": 0}

        def flaky(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(22, "Invalid argument")
            return real_open(self, *a, **k)

        monkeypatch.setattr(manifest.time, "sleep", lambda s: None)
        monkeypatch.setattr(Path, "open", flaky)
        import hashlib

        assert manifest.sha256_file(target) == hashlib.sha256(b"abc").hexdigest()

        def broken(self, *a, **k):
            raise OSError(22, "Invalid argument")

        monkeypatch.setattr(Path, "open", broken)
        with pytest.raises(OSError):
            manifest.sha256_file(target)

    def test_directory_listing_retries_transient_errors(self, tmp_path, monkeypatch):
        # A starved machine fails a directory listing with WinError 1450; the walk must
        # retry it rather than abort the build, and still list every file exactly once.
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_bytes(b"a")
        (tmp_path / "b.dll").write_bytes(b"bb")
        real = manifest._list_dir
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(1450, "Insufficient system resources")
            return real(path)

        monkeypatch.setattr(manifest.time, "sleep", lambda s: None)
        monkeypatch.setattr(manifest, "_list_dir", flaky)
        data = manifest.build_manifest(tmp_path, _meta())
        assert sorted(data["files"]) == ["b.dll", "pkg/a.py"]
        assert data["files"]["b.dll"]["size"] == 2

        def always(path):
            raise OSError(1450, "Insufficient system resources")

        monkeypatch.setattr(manifest, "_list_dir", always)
        with pytest.raises(OSError):
            manifest.build_manifest(tmp_path, _meta())

    def test_signed_must_be_a_real_boolean(self):
        data = {"schema": 1, **_meta(signed="no"), "files": {}}
        with pytest.raises(ValueError, match="signed"):
            manifest.validate_manifest(data)

    def test_forbidden_payload_is_caught(self, tmp_path):
        cfg = json.loads((INSTALLER / "config.json").read_text(encoding="utf-8"))
        for name in ("crack.onnx", "w.onnx.data", "best.pt", ".env", "kaggle.json",
                     "google1234abcd.html", "libdirp.dll", "ok.py"):
            (tmp_path / name).write_text("x")
        data = manifest.build_manifest(tmp_path, _meta())
        hits = manifest.forbidden_files(data, cfg["forbidden_payload_globs"])
        assert "ok.py" not in hits and len(hits) == 7

    def test_allowlist_exempts_only_the_named_third_party_files(self):
        cfg = json.loads((INSTALLER / "config.json").read_text(encoding="utf-8"))
        allow = [a["path"] for a in cfg["forbidden_payload_allow"]]
        assert all(a.get("reason") for a in cfg["forbidden_payload_allow"])
        sp = "runtime/Lib/site-packages/"
        paths = [sp + "certifi/cacert.pem", sp + "botocore/cacert.pem",
                 sp + "onnxruntime/datasets/mul_1.onnx",
                 sp + "somepkg/cacert.pem",              # not named: refused
                 sp + "onnxruntime/capi/model.onnx",     # outside datasets/: refused
                 "odk/models/onnx/crack.onnx",           # app weights: refused
                 "odk/server.pem", "odk/main.py"]
        hits = manifest.forbidden_files(paths, cfg["forbidden_payload_globs"], allow)
        assert hits == [sp + "somepkg/cacert.pem", sp + "onnxruntime/capi/model.onnx",
                        "odk/models/onnx/crack.onnx", "odk/server.pem"]


class TestPayloadShape:
    def test_long_paths_are_refused(self):
        ok = "runtime/Lib/site-packages/numpy/__init__.py"
        long = "runtime/Lib/site-packages/share/jupyter/" + "x" * 200 + ".js.map"
        assert manifest.too_long_paths([ok, long], 190) == [long]
        assert manifest.too_long_paths([ok], 190) == []

    def test_limit_keeps_installed_paths_under_max_path(self):
        cfg = json.loads((INSTALLER / "config.json").read_text(encoding="utf-8"))
        default_dir = "C:/Users/a-typical-user/AppData/Local/Programs/OpenDroneKit/"
        assert len(default_dir) + cfg["max_relative_path"] < 260

    def test_prune_removes_only_named_folders_with_reasons(self, tmp_path):
        cfg = json.loads((INSTALLER / "config.json").read_text(encoding="utf-8"))
        assert cfg["payload_prune"] and all(r.get("reason") for r in cfg["payload_prune"])
        jup = tmp_path / "runtime/Lib/site-packages/share/jupyter/labextensions"
        jup.mkdir(parents=True)
        (jup / "chunk.js").write_text("x")
        keep = tmp_path / "runtime/Lib/site-packages/numpy"
        keep.mkdir(parents=True)
        (keep / "__init__.py").write_text("x")
        assert build.prune_payload(tmp_path, cfg["payload_prune"]) == [
            "runtime/Lib/site-packages/share/jupyter"]
        assert not jup.exists() and (keep / "__init__.py").is_file()
        # idempotent: a reused runtime that was already pruned is fine
        assert build.prune_payload(tmp_path, cfg["payload_prune"]) == []

    def test_prune_rule_without_reason_or_escaping_stage_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="without a reason"):
            build.prune_payload(tmp_path, [{"path": "runtime/x"}])
        with pytest.raises(SystemExit, match="unsafe"):
            build.prune_payload(tmp_path, [{"path": "../outside", "reason": "r"}])


class TestSigningPlan:
    def test_no_certificate_means_unsigned_and_says_so(self):
        plan = sign.plan_signing(env={}, signtool="signtool.exe")
        assert plan.will_sign is False and plan.reason.startswith("UNSIGNED")
        assert sign.sign_file(plan, Path("x.exe")) == {"signed": False, "reason": plan.reason}

    def test_azure_trusted_signing_command(self):
        env = {"ODK_AZURE_SIGNING_DLIB": r"C:\ts\Azure.CodeSigning.Dlib.dll",
               "ODK_AZURE_SIGNING_METADATA": r"C:\ts\metadata.json"}
        plan = sign.plan_signing(env=env, signtool="signtool.exe")
        assert plan.mode == "azure"
        cmd = plan.command("setup.exe")
        assert cmd[:2] == ["signtool.exe", "sign"] and cmd[-1] == "setup.exe"
        assert cmd[cmd.index("/dlib") + 1].endswith("Dlib.dll")
        assert cmd[cmd.index("/dmdf") + 1].endswith("metadata.json")
        assert cmd[cmd.index("/fd") + 1] == "SHA256" and "/tr" in cmd
        assert plan.inno_signtool().endswith("$f")

    def test_certificate_thumbprint_command(self):
        plan = sign.plan_signing(env={"ODK_SIGN_CERT_THUMBPRINT": "AB CD EF"}, signtool="st.exe")
        assert plan.mode == "signtool" and plan.args[-2:] == ["/sha1", "ABCDEF"]

    def test_configured_but_no_signtool_is_unsigned(self):
        plan = sign.plan_signing(env={"ODK_SIGN_CERT_THUMBPRINT": "AB"}, signtool="")
        assert plan.will_sign is False and "signtool.exe was not found" in plan.reason

    def test_half_configured_azure_is_unsigned(self):
        plan = sign.plan_signing(env={"ODK_SIGN_MODE": "azure", "ODK_AZURE_SIGNING_DLIB": "x"},
                                 signtool="st.exe")
        assert plan.will_sign is False and "ODK_AZURE_SIGNING_METADATA" in plan.reason


class TestRuntimeLayout:
    def test_pth_enables_site_packages_and_the_app(self):
        lines = build.pth_contents("3.11.9").splitlines()
        assert lines[0] == "python311.zip"
        assert "Lib\\site-packages" in lines and "..\\odk" in lines and "import site" in lines

    def test_python_pin_matches_the_tested_minor_version(self):
        cfg = json.loads((INSTALLER / "config.json").read_text(encoding="utf-8"))
        pinned = tuple(int(p) for p in cfg["python"]["version"].split(".")[:2])
        assert re.fullmatch(r"[0-9a-f]{64}", cfg["python"]["embed_sha256"])
        assert cfg["python"]["embed_url"].startswith("https://www.python.org/ftp/python/")
        assert pinned == sys.version_info[:2], "the suite must run on the runtime's minor version"

    def test_lock_is_exact_and_hashed(self):
        lock = INSTALLER / "requirements-lock.txt"
        lines = [l for l in lock.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
        assert lines
        for line in lines:
            assert re.fullmatch(r"[a-z0-9-]+==[^ ]+ --hash=sha256:[0-9a-f]{64}", line), line
        names = {l.split("==")[0] for l in lines}
        from app.dependencies import read_requirements

        for req in read_requirements():
            assert build.normalize(req.name) in names, f"{req.name} is not in the lock"
        assert "onnxruntime-gpu" not in names and "torch" not in names

    def test_runtime_is_reused_only_for_the_same_embed_and_lock(self, tmp_path, monkeypatch):
        cfg = json.loads((INSTALLER / "config.json").read_text(encoding="utf-8"))
        lock = tmp_path / "lock.txt"
        lock.write_text("numpy==2.3.5 --hash=sha256:" + "a" * 64 + "\n")
        monkeypatch.setattr(build, "LOCK", lock)
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        stamp = build.runtime_stamp(cfg)
        assert not build.runtime_reusable(runtime, stamp), "no stamp, no reuse"
        (runtime / build.STAMP_NAME).write_text(stamp + "\n")
        assert not build.runtime_reusable(runtime, stamp), "no interpreter, no reuse"
        (runtime / "python.exe").write_bytes(b"MZ")
        assert build.runtime_reusable(runtime, stamp)
        lock.write_text("numpy==2.3.4 --hash=sha256:" + "b" * 64 + "\n")
        assert not build.runtime_reusable(runtime, build.runtime_stamp(cfg)), "lock changed"
        other = {**cfg, "python": {**cfg["python"], "embed_sha256": "0" * 64}}
        lock.write_text("numpy==2.3.5 --hash=sha256:" + "a" * 64 + "\n")
        assert not build.runtime_reusable(runtime, build.runtime_stamp(other)), "embed changed"

    def test_distribution_file_names_parse(self):
        for name, want in (("pywebview-6.2.1-py3-none-any.whl", ("pywebview", "6.2.1")),
                           ("open3d-0.19.0-cp311-cp311-win_amd64.whl", ("open3d", "0.19.0")),
                           ("proxy_tools-0.1.0.tar.gz", ("proxy_tools", "0.1.0"))):
            match = build._DIST.match(name)
            assert (match["name"], match["ver"]) == want


class TestInnoScript:
    TEXT = (INSTALLER / "OpenDroneKit.iss").read_text(encoding="utf-8")

    def test_per_user_without_admin(self):
        assert "PrivilegesRequired=lowest" in self.TEXT
        assert "DefaultDirName={autopf}" in self.TEXT

    def test_shortcuts_and_uninstaller(self):
        assert "{autoprograms}\\{#AppName}\\{#AppName}" in self.TEXT
        assert "{autodesktop}\\{#AppName}" in self.TEXT
        assert "{uninstallexe}" in self.TEXT
        assert "--download-models" in self.TEXT

    def test_every_define_is_passed_by_the_build(self):
        used = set(re.findall(r"\{#(\w+)\}", self.TEXT)) | set(re.findall(r"#if (\w+)", self.TEXT))
        passed = set(re.findall(r'/D(\w+)=', (INSTALLER / "build.py").read_text(encoding="utf-8")))
        assert used <= passed, used - passed

    def test_unsigned_is_marked(self):
        assert "InfoBeforeFile={#UnsignedNotice}" in self.TEXT
        src = (INSTALLER / "build.py").read_text(encoding="utf-8")
        assert '"-UNSIGNED"' in src and "(UNSIGNED)" in src

    def test_user_data_is_not_deleted_on_uninstall(self):
        deletes = re.findall(r"Name: \"([^\"]+)\"", self.TEXT.split("[UninstallDelete]")[1].split("[Code]")[0])
        assert deletes == ["{app}\\odk", "{app}\\runtime", "{app}"]


class TestLauncher:
    def test_installed_defaults_keep_user_state_outside_the_install(self, tmp_path):
        env = {"LOCALAPPDATA": str(tmp_path), "SYSTEMROOT": "C:\\Windows", "PATH": ""}
        code = ("import sys, os; sys.argv=['x']; sys.path.insert(0, r'%s'); import launch; "
                "launch._defaults(); print(os.environ['ODK_DATA_DIR']); "
                "print(os.environ['ODK_MODEL_WEIGHTS_DIR']); print(os.environ['ODK_INSTALLED'])"
                % INSTALLER)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, check=True).stdout.split()
        assert out == [str(tmp_path / "OpenDroneKit" / "data"),
                       str(tmp_path / "OpenDroneKit" / "models"), "1"]

    def test_launcher_never_writes_bytecode_into_the_install(self, tmp_path):
        env = {"LOCALAPPDATA": str(tmp_path), "SYSTEMROOT": "C:\\Windows", "PATH": ""}
        code = ("import sys; sys.path.insert(0, r'%s'); import launch; "
                "launch._defaults = lambda: None; launch.ODK = __import__('pathlib').Path(r'%s'); "
                "sys.modules['main'] = type(sys)('main'); sys.modules['main'].main = lambda a: 0; "
                "launch.main([]); print(sys.dont_write_bytecode)" % (INSTALLER, tmp_path))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, check=True).stdout.split()
        assert out == ["True"]


class TestBytecode:
    def test_site_packages_are_recompiled_as_unchecked_hash(self, tmp_path):
        # Timestamp .pyc files were rewritten on first import after Inno Setup changed
        # source mtimes; unchecked-hash ones never are.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("X = 1\n")
        (pkg / "py2_only.py").write_text("print 'python 2'\n")
        out = subprocess.run([sys.executable, "-c", build.BYTECODE_SCRIPT, str(tmp_path)],
                             capture_output=True, text=True, check=True).stdout
        assert "bytecode: 2 compiled, 1 skipped" in out
        tag = sys.implementation.cache_tag
        for name in ("__init__", "mod"):
            header = (pkg / "__pycache__" / f"{name}.{tag}.pyc").read_bytes()[4:8]
            assert int.from_bytes(header, "little") == 0b01, "hash-based, source not checked"


class TestSelfTest:
    def test_runs_headless_in_this_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ODK_DATA_DIR", str(tmp_path))
        from app.selftest import run_self_test

        report = run_self_test(verify_manifest=False)
        failed = [c for c in report["checks"] if not c["ok"]]
        assert report["ok"], failed
        names = {c["check"] for c in report["checks"]}
        assert {"import webview", "construct Api", "api.capabilities",
                "api.check_for_update (opt-in off, no network)"} <= names
