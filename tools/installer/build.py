"""Build the OpenDroneKit Windows installer, reproducibly.

    python tools/installer/build.py --out E:/odk-build             # build from HEAD
    python tools/installer/build.py --out E:/odk-build --relock    # re-resolve the lock first

Pipeline (each step refuses rather than guesses):

1. **Runtime**: the python.org *embeddable* CPython (version pinned in config.json),
   downloaded once into the cache and checked against the pinned sha256.
2. **Packages**: every wheel named in ``requirements-lock.txt`` -- exact versions and
   sha256 hashes -- downloaded with ``--require-hashes`` and installed into
   ``runtime/Lib/site-packages``. ``--relock`` regenerates the lock from the
   environment running this script (the one the test suite passes in): the closure
   of ``requirements.txt`` at exactly the installed versions, with the GPU build of
   onnxruntime swapped for the CPU build and torch never included.
3. **Payload**: ``git archive HEAD`` (so the build is the commit, not the working tree),
   minus tests and dev metadata. The build fails if any model weight, credential or
   vendor SDK file is in it.
4. **Manifest**: ``install_manifest.json`` lists every shipped file with its sha256.
5. **Installer**: Inno Setup (installed with winget, or choco, only when absent).
6. **Signing**: tools/installer/sign.py. Without a certificate the output is marked
   UNSIGNED in its name, version info, manifest and first wizard page.
7. **Checksums**: ``SHA256SUMS.txt`` beside the installer, in the format the in-app
   update check reads from a GitHub release.

The script must run under the same CPython minor version as the embedded runtime,
because it uses this interpreter's pip to install into the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from tools.installer.manifest import (  # noqa: E402
    build_manifest, forbidden_files, sha256_file, write_manifest,
)
from tools.installer.sign import plan_signing  # noqa: E402

LOCK = HERE / "requirements-lock.txt"
CONFIG = HERE / "config.json"
ICON = REPO / "app" / "web" / "assets" / "brand" / "derived" / "favicon.ico"
_DIST = re.compile(r"^(?P<name>[A-Za-z0-9_.]+?)-(?P<ver>[0-9][^-]*?)(-|\.tar\.gz$|\.zip$)")


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], **kwargs)
    if proc.returncode != 0:
        raise SystemExit(f"[build] command failed ({proc.returncode}): {cmd[0]}")
    return proc


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def app_version() -> str:
    text = (REPO / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# -- 1. runtime ----------------------------------------------------------------

def fetch_embed(cfg: dict, cache: Path) -> Path:
    py = cfg["python"]
    target = cache / Path(py["embed_url"]).name
    if not target.is_file():
        log(f"downloading {py['embed_url']}")
        with urllib.request.urlopen(py["embed_url"], timeout=120) as response:  # noqa: S310
            data = response.read()
        target.write_bytes(data)
    got = sha256_file(target)
    if got != py["embed_sha256"]:
        target.unlink()
        raise SystemExit(f"[build] REFUSED: {target.name} sha256 {got} != pinned {py['embed_sha256']}")
    log(f"embeddable python verified: {target.name} sha256 {got}")
    return target


def pth_contents(version: str) -> str:
    tag = "".join(version.split(".")[:2])
    return "\n".join([f"python{tag}.zip", ".", "Lib\\site-packages", "..\\odk", "import site", ""])


# -- 2. packages ----------------------------------------------------------------

# The runtime ships the CPU build of onnxruntime: the GPU builds need a matching CUDA /
# cuDNN install that an installer cannot assume. The tested env may carry a GPU build.
SUBSTITUTE = {"onnxruntime-gpu": "onnxruntime", "onnxruntime-directml": "onnxruntime",
              "opencv-python-headless": "opencv-python"}
# Never bundled, whatever the env has (torch is optional and multi-GB; see INSTALLER.md).
NEVER = {"torch", "torchvision", "pip", "setuptools", "wheel"}


def env_closure(requirements: Path) -> dict[str, str]:
    """{name: version} for requirements.txt and everything it pulls in, read from the
    environment running this script -- the environment the test suite passes in.

    Pinning what was tested, instead of asking pip to re-resolve against an index, is
    what makes the bundle the same software the suite verified (and avoids resolver
    backtracking across hundreds of candidate versions)."""
    from importlib import metadata

    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    from app.dependencies import ALTERNATIVES, read_requirements

    def dist_for(name: str):
        for candidate in (name, *ALTERNATIVES.get(normalize(name), ())):
            try:
                return metadata.distribution(candidate)
            except metadata.PackageNotFoundError:
                continue
        raise SystemExit(f"[build] {name} is required but not installed in {sys.executable}")

    pins: dict[str, str] = {}
    queue: list[tuple[str, frozenset[str]]] = []
    for line in (requirements.read_text(encoding="utf-8").splitlines()):
        line = line.split("#", 1)[0].strip()
        if line:
            req = Requirement(line)
            queue.append((req.name, frozenset(req.extras)))
    assert {normalize(r.name) for r in read_requirements()} <= {normalize(n) for n, _ in queue}
    seen: set[tuple[str, frozenset[str]]] = set()
    while queue:
        name, extras = queue.pop()
        if (normalize(name), extras) in seen:
            continue
        seen.add((normalize(name), extras))
        dist = dist_for(name)
        key = normalize(dist.metadata["Name"])
        key = SUBSTITUTE.get(key, key)
        if key in NEVER:
            continue
        pins[key] = dist.version
        for spec in dist.requires or []:
            req = Requirement(spec)
            env = default_environment()
            wanted = any(req.marker.evaluate({**env, "extra": e}) for e in extras or {""}) \
                if req.marker else True
            if wanted:
                queue.append((req.name, frozenset(req.extras)))
    return pins


def relock(wheel_dir: Path) -> None:
    """Pin the tested environment's closure of requirements.txt into a hashed lock."""
    with tempfile.TemporaryDirectory(prefix="odk-relock-", dir=str(wheel_dir.parent)) as tmp:
        tmpdir = Path(tmp)
        pins = env_closure(REPO / "requirements.txt")
        pinned = tmpdir / "pinned.txt"
        pinned.write_text("".join(f"{n}=={v}\n" for n, v in sorted(pins.items())), encoding="utf-8")
        log(f"pinning {len(pins)} distributions from {sys.executable}")
        dl = tmpdir / "dl"
        run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--prefer-binary",
             "-r", pinned, "-d", dl])
        lines = []
        for path in sorted(dl.iterdir(), key=lambda p: p.name.lower()):
            match = _DIST.match(path.name)
            if not match:
                raise SystemExit(f"[build] cannot parse distribution file name {path.name}")
            lines.append(f"{normalize(match['name'])}=={match['ver']} --hash=sha256:{sha256_file(path)}")
            wheel_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, wheel_dir / path.name)
    header = ("# Generated by tools/installer/build.py --relock from requirements.txt, constrained\n"
              f"# to the versions in the environment that ran it (Python {sys.version.split()[0]}).\n"
              "# Exact versions and sha256 hashes; installed with pip --require-hashes.\n")
    LOCK.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {LOCK.relative_to(REPO)} ({len(lines)} distributions)")


def install_packages(wheel_dir: Path, site_packages: Path) -> None:
    run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--require-hashes",
         "-r", LOCK, "-d", wheel_dir])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheel_dir,
         "--no-deps", "--require-hashes", "--no-build-isolation", "--no-warn-script-location",
         "--target", site_packages, "-r", LOCK])
    # pip --target writes console-script launchers bound to *this* interpreter's path,
    # which would be wrong on any other machine. The app never uses them.
    shutil.rmtree(site_packages / "bin", ignore_errors=True)


# -- 3. payload -------------------------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()


def stage_payload(dest: Path, excludes: list[str]) -> dict:
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty:
        log("WARNING: the working tree has uncommitted changes; the build uses HEAD only")
    data = subprocess.run(["git", "-c", "safe.directory=*", "archive", "--format=tar", "HEAD"],
                          cwd=REPO, capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = [m for m in tar.getmembers()
                   if not any(m.name == e.rstrip("/") or m.name.startswith(e) for e in excludes)]
        tar.extractall(dest, members=members, filter="data")
    return {"git_commit": commit, "git_dirty": dirty}


# -- 5. Inno Setup ------------------------------------------------------------------

def find_iscc() -> str:
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
    ]
    found = shutil.which("ISCC") or next((str(c) for c in candidates if c.is_file()), "")
    return found


def ensure_iscc(allow_install: bool) -> str:
    iscc = find_iscc()
    if iscc:
        log(f"Inno Setup found: {iscc}")
        return iscc
    if not allow_install:
        raise SystemExit("[build] Inno Setup 6 not found and --no-tool-install was given.")
    if shutil.which("winget"):
        log("Inno Setup 6 is not installed; installing it with winget (user scope)")
        run(["winget", "install", "--id", "JRSoftware.InnoSetup", "-e", "--scope", "user",
             "--silent", "--accept-package-agreements", "--accept-source-agreements",
             "--disable-interactivity"])
    elif shutil.which("choco"):
        log("Inno Setup 6 is not installed; installing it with choco")
        run(["choco", "install", "innosetup", "-y", "--no-progress"])
    else:
        raise SystemExit("[build] Inno Setup 6 is not installed and neither winget nor choco "
                         "is available. Install it from https://jrsoftware.org/isinfo.php")
    iscc = find_iscc()
    if not iscc:
        raise SystemExit("[build] Inno Setup was installed but ISCC.exe was not found.")
    return iscc


UNSIGNED_NOTICE = """\
UNSIGNED BUILD

This OpenDroneKit installer is not code-signed. The project does not have a
code-signing certificate yet, so Windows SmartScreen may warn that the publisher
is unknown.

To check that this is the file the project published, compare its SHA-256 with
SHA256SUMS.txt from the same GitHub release:

    Get-FileHash .\\<installer>.exe -Algorithm SHA256
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the OpenDroneKit Windows installer.")
    parser.add_argument("--out", required=True, help="build root (cache/, stage/, dist/)")
    parser.add_argument("--relock", action="store_true", help="regenerate requirements-lock.txt")
    parser.add_argument("--no-tool-install", action="store_true", help="never install Inno Setup")
    parser.add_argument("--skip-installer", action="store_true", help="stage only; do not run ISCC")
    args = parser.parse_args(argv)

    started = time.time()
    cfg = load_config()
    want = tuple(int(p) for p in cfg["python"]["version"].split(".")[:2])
    if sys.version_info[:2] != want:
        raise SystemExit(f"[build] run this with Python {want[0]}.{want[1]} (the embedded runtime's "
                         f"version); this is {sys.version.split()[0]}")
    out = Path(args.out).resolve()
    cache, stage_root, dist = out / "cache", out / "stage", out / "dist"
    wheel_dir = cache / "wheelhouse"
    for folder in (cache, dist, wheel_dir):
        folder.mkdir(parents=True, exist_ok=True)
    version = app_version()
    plan = plan_signing()
    signed = plan.will_sign
    log(f"OpenDroneKit {version}; signing: {plan.reason}")

    iscc = "" if args.skip_installer else ensure_iscc(not args.no_tool_install)
    embed = fetch_embed(cfg, cache)
    if args.relock or not LOCK.is_file():
        relock(wheel_dir)

    stage = stage_root / cfg["app_name"]
    if stage_root.exists():
        shutil.rmtree(stage_root)
    runtime = stage / "runtime"
    runtime.mkdir(parents=True)
    with zipfile.ZipFile(embed) as archive:
        archive.extractall(runtime)
    tag = "".join(cfg["python"]["version"].split(".")[:2])
    (runtime / f"python{tag}._pth").write_text(pth_contents(cfg["python"]["version"]), encoding="ascii")
    site_packages = runtime / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    install_packages(wheel_dir, site_packages)

    git_meta = stage_payload(stage / "odk", cfg["payload_exclude"])
    shutil.copy2(HERE / "launch.py", stage / "launch.py")
    shutil.copy2(ICON, stage / "OpenDroneKit.ico")
    # Byte-compile with the embedded interpreter itself: this is also the first proof the
    # runtime starts. unchecked-hash .pyc files are never rewritten, so the install folder
    # stays exactly what the manifest lists.
    run([runtime / "python.exe", "-m", "compileall", "-q", "-j", "0",
         "--invalidation-mode", "unchecked-hash", stage / "odk", stage / "launch.py"])
    run([runtime / "python.exe", "-c",
         "import sys, numpy, onnxruntime, rasterio, webview; print(sys.version); "
         "print('runtime imports ok')"])

    packages = [l.split(" ")[0] for l in LOCK.read_text(encoding="utf-8").splitlines()
                if l and not l.startswith("#")]
    meta = {
        "app": cfg["app_name"], "version": version, "python": cfg["python"]["version"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signed": signed, "signing": plan.reason if not signed else plan.mode,
        "embed_sha256": cfg["python"]["embed_sha256"], "lock_sha256": sha256_file(LOCK),
        "packages": packages, **git_meta,
    }
    manifest = build_manifest(stage, meta)
    bad = forbidden_files(manifest, cfg["forbidden_payload_globs"])
    if bad:
        raise SystemExit(f"[build] REFUSED: forbidden files in the payload (weights/secrets/SDKs): {bad[:20]}")
    write_manifest(stage, manifest)
    log(f"manifest: {manifest['file_count']} files, {manifest['total_bytes'] / 1e6:.0f} MB")

    report = {"version": version, "signed": signed, "signing": plan.reason, "stage": str(stage),
              "manifest_files": manifest["file_count"], "payload_bytes": manifest["total_bytes"],
              **git_meta}
    if not args.skip_installer:
        label = "" if signed else "-UNSIGNED"
        base = f"{cfg['app_name']}-{version}-win64-setup{label}"
        notice = out / "UNSIGNED.txt"
        notice.write_text(UNSIGNED_NOTICE, encoding="utf-8")
        cmd = [iscc, "/Q",
               f"/DAppName={cfg['app_name']}", f"/DAppVersion={version}", f"/DAppId={cfg['inno_app_id']}",
               f"/DPublisher={cfg['publisher']}", f"/DAppUrl={cfg['app_url']}",
               f"/DStageDir={stage}", f"/DOutputDir={dist}", f"/DOutputBase={base}",
               f"/DIconFile={ICON}", f"/DLicenseFile={stage / 'odk' / 'LICENSE'}",
               f"/DUnsignedNotice={notice}", f"/DUnsigned={'0' if signed else '1'}",
               f"/DSignLabel={'' if signed else ' (UNSIGNED)'}",
               f"/DSignEnabled={'1' if signed else '0'}"]
        if signed:
            cmd.append(f"/Sodksign={plan.inno_signtool()}")
        cmd.append(HERE / "OpenDroneKit.iss")
        run(cmd)
        installer = dist / f"{base}.exe"
        digest = sha256_file(installer)
        (dist / "SHA256SUMS.txt").write_text(f"{digest}  {installer.name}\n", encoding="utf-8")
        report.update({"installer": str(installer), "installer_sha256": digest,
                       "installer_bytes": installer.stat().st_size})
        if not signed:
            log("=" * 72)
            log(f"UNSIGNED: {installer.name}")
            log(plan.reason)
            log("=" * 72)
    report["seconds"] = round(time.time() - started, 1)
    (out / "build-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
