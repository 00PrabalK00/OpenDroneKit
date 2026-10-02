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
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    app_id = json.loads((here / "config.json").read_text(encoding="utf-8"))["inno_app_id"].replace("{{", "{")
    installer = Path(args.installer).resolve()
    work = Path(args.work).resolve()
    if work.exists():
        shutil.rmtree(work)
    target, userdata = work / "install", work / "userdata"
    work.mkdir(parents=True)
    report: dict = {"installer": str(installer), "install_dir": str(target), "steps": []}

    def step(name: str, ok: bool, **detail) -> None:
        report["steps"].append({"step": name, "ok": bool(ok), **detail})
        print(f"[smoke] {'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:400]}", flush=True)

    t0 = time.time()
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
    t0 = time.time()
    test = subprocess.run([str(target / "runtime" / "python.exe"), str(target / "launch.py"), "--self-test"],
                          cwd=str(target / "odk"), env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=1800)
    (work / "selftest.json").write_text(test.stdout, encoding="utf-8")
    (work / "selftest.stderr.txt").write_text(test.stderr, encoding="utf-8")
    try:
        selftest = json.loads(test.stdout)
    except ValueError:
        selftest = {"ok": False, "checks": []}
    summary = [{"check": c["check"], "ok": c["ok"], **({"error": c["error"]} if not c["ok"] else {})}
               for c in selftest.get("checks", [])]
    step("installed app self-test (scrubbed env)", test.returncode == 0 and selftest.get("ok"),
         returncode=test.returncode, seconds=round(time.time() - t0, 1), checks=summary)
    report["selftest"] = selftest

    dl = subprocess.run([str(target / "runtime" / "python.exe"), str(target / "launch.py"),
                         "--download-models", "--list"], cwd=str(target / "odk"), env=env,
                        capture_output=True, text=True, encoding="utf-8", timeout=600)
    (work / "download-models.txt").write_text(dl.stdout + dl.stderr, encoding="utf-8")
    step("first-run model downloader (--list)", dl.returncode == 0, output=dl.stdout.splitlines()[:20])

    uninstaller = target / "unins000.exe"
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

    report["ok"] = all(s["ok"] for s in report["steps"])
    (work / "smoke-report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[smoke] {'ALL PASSED' if report['ok'] else 'FAILED'}; report: {work / 'smoke-report.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
