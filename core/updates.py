"""Opt-in update check against GitHub Releases. Never installs anything by itself.

OpenDroneKit is offline-first, so this module makes **no** network call unless the user
has switched update checks on (``AppSettings.update_check_opt_in``, off by default) or
asks for one explicitly. When it does check:

* it reads ``releases/latest`` of ``00PrabalK00/OpenDroneKit`` from the GitHub API;
* a newer version is reported only when the release publishes a sha256 for the
  installer -- GitHub's own asset ``digest`` field, or a ``SHA256SUMS`` asset -- and the
  two agree when both exist. A release without a digest is reported but not offered;
* ``download_update`` fetches the installer to the user's downloads folder for
  OpenDroneKit, verifies the sha256, and returns the path. Running it is the user's
  decision; nothing here executes it, schedules it, or passes it silent flags.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = "00PrabalK00/OpenDroneKit"
API_BASE = "https://api.github.com"
USER_AGENT = "OpenDroneKit-update-check"
_SHA = re.compile(r"\b([0-9a-fA-F]{64})\b")


# -- versions ------------------------------------------------------------------

_PRE_ORDER = {"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2, "c": 3, "rc": 3, "pre": 3}


def parse_version(text: str) -> tuple[tuple[int, ...], tuple[int, int]]:
    """'v1.2.0-rc.1' -> ((1, 2, 0), (3, 1)). A final release sorts after its pre-releases.

    Raises ValueError for anything that is not a version, so a tag like 'nightly' is
    never mistaken for a newer release.
    """
    raw = str(text or "").strip()
    raw = raw[1:] if raw[:1] in ("v", "V") else raw
    raw = raw.split("+", 1)[0]  # build metadata never orders
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:[-._]?([A-Za-z]+)[-._]?(\d*))?", raw)
    if not match:
        raise ValueError(f"not a version: {text!r}")
    numbers = tuple(int(p) for p in match.group(1).split("."))
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers = numbers[:-1]
    label = (match.group(2) or "").lower()
    if not label:
        pre = (99, 0)
    elif label in _PRE_ORDER:
        pre = (_PRE_ORDER[label], int(match.group(3) or 0))
    elif label in ("post", "final"):
        pre = (100, int(match.group(3) or 0))
    else:
        raise ValueError(f"unknown pre-release label in {text!r}")
    return numbers, pre


def is_newer(candidate: str, current: str) -> bool:
    return parse_version(candidate) > parse_version(current)


def current_version() -> str:
    root = Path(__file__).resolve().parents[1] / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', root.read_text(encoding="utf-8"))
    return match.group(1) if match else "0.0.0"


# -- release parsing -----------------------------------------------------------

def parse_sha256sums(text: str) -> dict[str, str]:
    """`<hex>  <name>` (sha256sum format, optional `*`) -> {name: hex}."""
    sums: dict[str, str] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and _SHA.fullmatch(parts[0]):
            sums[parts[-1].lstrip("*")] = parts[0].lower()
    return sums


def pick_installer(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    exes = [a for a in assets if str(a.get("name", "")).lower().endswith(".exe")]
    for asset in exes:
        if "setup" in str(asset["name"]).lower():
            return asset
    return exes[0] if exes else None


def release_digest(asset: dict[str, Any], sums: dict[str, str] | None) -> tuple[str, str]:
    """(sha256, source) published for an asset, or ('', reason) when none can be trusted."""
    api = str(asset.get("digest") or "")
    api_hex = api.split(":", 1)[1].lower() if api.lower().startswith("sha256:") else ""
    listed = (sums or {}).get(str(asset.get("name", "")), "")
    if api_hex and listed and api_hex != listed:
        return "", (f"the release's SHA256SUMS ({listed}) disagrees with GitHub's asset "
                    f"digest ({api_hex}); refusing")
    if api_hex:
        return api_hex, "github asset digest"
    if listed:
        return listed, "SHA256SUMS"
    return "", "the release publishes no sha256 for its installer; refusing to offer it"


# -- network -------------------------------------------------------------------

def _get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https/configured base
        return response.read()


def opted_in() -> bool:
    try:
        from core.settings import load_app_settings

        return bool(getattr(load_app_settings(), "update_check_opt_in", False))
    except Exception:  # noqa: BLE001
        return False


def check_for_update(*, current: str | None = None, explicit: bool = False,
                     api_base: str = API_BASE, repo: str = REPO,
                     timeout: float = 15.0) -> dict[str, Any]:
    """Compare the latest published release with this build.

    Unless the user opted in, or ``explicit`` says the user just asked, no request is made.
    """
    current = current or current_version()
    if not explicit and not opted_in():
        return {"checked": False, "current": current,
                "reason": "Update checks are off. They are opt-in; nothing was contacted."}
    url = f"{api_base.rstrip('/')}/repos/{repo}/releases/latest"
    try:
        release = json.loads(_get(url, timeout).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"checked": True, "current": current, "update_available": False,
                    "reason": f"{repo} has no published release yet."}
        return {"checked": False, "current": current, "reason": f"GitHub answered HTTP {exc.code}."}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"checked": False, "current": current, "reason": f"Could not reach GitHub: {exc}"}

    tag = str(release.get("tag_name") or "")
    result: dict[str, Any] = {"checked": True, "current": current, "latest": tag,
                              "release_url": release.get("html_url", ""),
                              "update_available": False}
    if release.get("draft") or release.get("prerelease"):
        result["reason"] = f"The latest release {tag} is a draft or pre-release; not offered."
        return result
    try:
        newer = is_newer(tag, current)
    except ValueError as exc:
        result["reason"] = f"Release tag is not a version ({exc}); not offered."
        return result
    if not newer:
        result["reason"] = f"{current} is up to date (latest release {tag})."
        return result

    assets = [a for a in release.get("assets") or [] if isinstance(a, dict)]
    installer = pick_installer(assets)
    if installer is None:
        result["reason"] = f"Release {tag} has no Windows installer asset."
        return result
    sums: dict[str, str] = {}
    sums_asset = next((a for a in assets if str(a.get("name", "")).upper().startswith("SHA256SUMS")), None)
    if sums_asset is not None:
        try:
            sums = parse_sha256sums(_get(str(sums_asset["browser_download_url"]), timeout).decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            result["reason"] = f"Could not read the release's SHA256SUMS: {exc}"
            return result
    digest, source = release_digest(installer, sums)
    if not digest:
        result["reason"] = source
        return result
    result.update({
        "update_available": True,
        "installer_name": installer.get("name", ""),
        "installer_url": installer.get("browser_download_url", ""),
        "installer_size": installer.get("size"),
        "sha256": digest,
        "sha256_source": source,
        "unsigned": "unsigned" in str(installer.get("name", "")).lower(),
        "reason": f"{tag} is available. Download it, then run the installer yourself.",
    })
    return result


def downloads_dir() -> Path:
    from core.paths import data_root

    return data_root() / "updates"


def download_update(update: dict[str, Any], *, dest_dir: Path | None = None,
                    timeout: float = 120.0) -> dict[str, Any]:
    """Fetch the installer an update check offered and verify its sha256. Never runs it."""
    url = str(update.get("installer_url") or "")
    expected = str(update.get("sha256") or "").lower()
    name = Path(str(update.get("installer_name") or "")).name
    if not (update.get("update_available") and url and name and _SHA.fullmatch(expected or "")):
        return {"ok": False, "error": "No verified update was offered; nothing downloaded."}
    folder = Path(dest_dir) if dest_dir else downloads_dir()
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / (name + ".part")
    hasher = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:  # noqa: S310
            for block in iter(lambda: response.read(1024 * 1024), b""):
                hasher.update(block)
                out.write(block)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        return {"ok": False, "error": f"Download failed: {exc}"}
    got = hasher.hexdigest()
    if got != expected:
        partial.unlink(missing_ok=True)
        return {"ok": False, "error": f"Refused: sha256 {got} does not match the published "
                                      f"{expected}. The file was deleted."}
    final = folder / name
    partial.replace(final)
    return {"ok": True, "path": str(final), "sha256": got, "installed": False,
            "message": "Verified. Close OpenDroneKit and run this installer to update; "
                       "it is never run automatically."}
