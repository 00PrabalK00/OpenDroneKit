"""Install a built installer into a temporary folder, prove it starts, then uninstall it.

    python tools/installer/smoke_install.py --installer dist/OpenDroneKit-...-setup-UNSIGNED.exe \
        --work E:/odk-build/smoke [--weights E:/odk-weights]

Steps, each recorded in <work>/smoke-report.json:

1. silent per-user install (/CURRENTUSER) into <work>/install, with the desktop icon task;
2. the uninstall entry exists under HKCU, and the Start-menu and desktop shortcuts exist;
3. ``runtime\\python.exe launch.py --self-test`` runs with a scrubbed environment (PATH is
   System32 only; no PYTHONPATH/PYTHONHOME/conda variables), so it can only succeed on
   what the installer put down; the self-test re-hashes every file in the manifest;
4. silent uninstall, then the install folder, shortcuts and uninstall entry are gone.

User state is pointed at <work>/userdata so the run leaves nothing in %LOCALAPPDATA%.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import winreg
from pathlib import Path

APP = "OpenDroneKit"


def known_folder(name: str) -> Path:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"[Environment]::GetFolderPath('{name}')"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return Path(out)


def uninstall_entry(app_id: str) -> dict[str, str] | None:
    key = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{app_id}_is1"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
            values = {}
            for name in ("DisplayName", "DisplayVersion", "UninstallString", "InstallLocation"):
                try:
                    values[name] = str(winreg.QueryValueEx(handle, name)[0])
                except OSError:
                    values[name] = ""
            return values
    except OSError:
        return None


def clean_env(userdata: Path, weights: str) -> dict[str, str]:
    system = os.environ.get("SystemRoot", r"C:\Windows")
    keep = {k: v for k, v in os.environ.items()
            if k.upper() in {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "USERPROFILE", "APPDATA",
                             "LOCALAPPDATA", "TEMP", "TMP", "USERNAME", "COMPUTERNAME",
                             "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "COMSPEC",
                             "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "PATHEXT"}}
    keep["PATH"] = rf"{system}\System32;{system};{system}\System32\WindowsPowerShell\v1.0"
    keep["ODK_DATA_DIR"] = str(userdata / "data")
    keep["ODK_MODEL_WEIGHTS_DIR"] = weights or str(userdata / "models")
    keep["PYTHONIOENCODING"] = "utf-8"
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installer", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--weights", default="", help="ODK_MODEL_WEIGHTS_DIR for the self-test")
    parser.add_argument("--resume", action="store_true",
                        help="keep an install a previous (interrupted) run left in <work>/install "
                             "and continue from the checks; the report says so")
    parser.add_argument("--selftest-timeout", type=int, default=5400,
                        help="seconds the installed self-test may take. A first start on a busy "
                             "machine is slow (antivirus scans the fresh 1.2 GB tree on first "
                             "load, then the manifest re-hashes it); a run that exceeds this is "
                             "recorded as a failed step and the uninstall still runs")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    app_id = json.loads((here / "config.json").read_text(encoding="utf-8"))["inno_app_id"].replace("{{", "{")
    installer = Path(args.installer).resolve()
    work = Path(args.work).resolve()
    target, userdata = work / "install", work / "userdata"
    resume = bool(args.resume and (target / "launch.py").is_file())
    if work.exists() and not resume:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"installer": str(installer), "install_dir": str(target), "steps": [],
                    "resumed": resume}

    def step(name: str, ok: bool, **detail) -> None:
        report["steps"].append({"step": name, "ok": bool(ok), **detail})
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:400]}", flush=True)

    t0 = time.time()
    if resume:
        # The install itself ran in an earlier, interrupted run; its Inno Setup log is the
        # evidence that it finished (Inno writes "Installation process succeeded" last).
        log_text = (work / "install.log").read_text(encoding="utf-8", errors="replace")             if (work / "install.log").is_file() else ""
        step("silent per-user install (earlier run, resumed)",
             "Installation process succeeded" in log_text and (target / "launch.py").is_file(),
             install_log=str(work / "install.log"))
    else:
        proc = subprocess.run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                               "/CURRENTUSER", f"/DIR={target}", "/TASKS=desktopicon",
                               f"/LOG={work / 'install.log'}"])
        step("silent per-user install", proc.returncode == 0 and (target / "launch.py").is_file(),
             returncode=proc.returncode, seconds=round(time.time() - t0, 1))

    entry = uninstall_entry(app_id)
    step("uninstall entry in HKCU (no admin)", entry is not None, entry=entry)
    programs = known_folder("Programs") / APP
    desktop = known_folder("Desktop") / f"{APP}.lnk"
    links = sorted(p.name for p in programs.glob("*.lnk")) if programs.is_dir() else []
    step("Start-menu shortcuts", f"{APP}.lnk" in links and f"Uninstall {APP}.lnk" in links,
         folder=str(programs), shortcuts=links)
    step("desktop shortcut", desktop.is_file(), path=str(desktop))

    env = clean_env(userdata, args.weights)
    try:
        run_installed_checks(target, work, env, args.selftest_timeout, step, report)
    finally:
        # Whatever happened above, the machine is left without the test install.
        uninstall(target, work, app_id, step)

    report["ok"] = all(s["ok"] for s in report["steps"])
    (work / "smoke-report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[smoke] {'ALL PASSED' if report['ok'] else 'FAILED'}; report: {work / 'smoke-report.json'}")
    return 0 if report["ok"] else 1


def run_logged(cmd: list[str], cwd: Path, env: dict[str, str], stdout: Path, stderr: Path,
               timeout: int) -> tuple[int | None, float]:
    """Run with output streamed to files (readable while it runs); None return code on timeout."""
    t0 = time.time()
    with open(stdout, "w", encoding="utf-8") as out, open(stderr, "w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err)
        try:
            code: int | None = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = None
    return code, round(time.time() - t0, 1)


def run_installed_checks(target: Path, work: Path, env: dict[str, str], timeout: int,
                         step, report: dict) -> None:
    python, launcher = str(target / "runtime" / "python.exe"), str(target / "launch.py")
    code, seconds = run_logged([python, launcher, "--self-test"], target / "odk", env,
                               work / "selftest.json", work / "selftest.stderr.txt", timeout)
    text = (work / "selftest.json").read_text(encoding="utf-8")
    try:
        selftest = json.loads(text) if text.strip() else {"ok": False, "checks": []}
    except ValueError:
        selftest = {"ok": False, "checks": []}
    summary = [{"check": c["check"], "ok": c["ok"], "seconds": c.get("seconds"),
                **({"error": c["error"]} if not c["ok"] else {})}
               for c in selftest.get("checks", [])]
    detail: dict = {"returncode": code, "seconds": seconds, "checks": summary}
    if code is None:
        progress = (work / "selftest.stderr.txt").read_text(encoding="utf-8").splitlines()
        detail["timed_out_after"] = timeout
        detail["last_progress"] = progress[-3:]
    step("installed app self-test (scrubbed env)", code == 0 and bool(selftest.get("ok")), **detail)
    report["selftest"] = selftest

    code, seconds = run_logged([python, launcher, "--download-models", "--list"], target / "odk",
                               env, work / "download-models.txt",
                               work / "download-models.stderr.txt", 1800)
    out = (work / "download-models.txt").read_text(encoding="utf-8")
    step("first-run model downloader (--list)", code == 0, returncode=code, seconds=seconds,
         output=out.splitlines()[:20])


def uninstall(target: Path, work: Path, app_id: str, step) -> None:
    programs = known_folder("Programs") / APP
    desktop = known_folder("Desktop") / f"{APP}.lnk"

    uninstaller = target / "unins000.exe"
    if not uninstaller.is_file():
        step("silent uninstall removes everything", False, error=f"{uninstaller} not found")
        return
    t0 = time.time()
    proc = subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                           f"/LOG={work / 'uninstall.log'}"])
    deadline = time.time() + 300
    while time.time() < deadline and (target.exists() or uninstall_entry(app_id) is not None):
        time.sleep(2)
    leftovers = sorted(str(p.relative_to(target)) for p in target.rglob("*")) if target.exists() else []
    step("silent uninstall removes everything", proc.returncode == 0 and not target.exists()
         and uninstall_entry(app_id) is None and not desktop.exists() and not programs.exists(),
         returncode=proc.returncode, seconds=round(time.time() - t0, 1), leftovers=leftovers[:20],
         desktop_shortcut_left=desktop.exists(), start_menu_left=programs.exists())


if __name__ == "__main__":
    sys.exit(main())
