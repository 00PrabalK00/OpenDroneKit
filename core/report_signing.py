"""Signed, traceable reports: an input manifest and an Ed25519 signature.

What a signed report carries
----------------------------
* **Manifest** (``odk-report-manifest/1``): the SHA-256 of every input the report was
  built from -- images, rasters, data files -- and the digest of every model that
  produced something in it. The manifest is printed in the PDF's appendix and embedded
  in the PDF as the attachment ``odk-manifest.json``, together with the report's
  content (every field and whether it was provided or MISSING) as
  ``odk-report-content.json``.
* **Signature** (``odk-report-signature/1``): Ed25519 over a small canonical-JSON
  statement that names the SHA-256 of the PDF bytes as written (manifest attachment
  included) and the SHA-256 of the manifest. It is appended to the PDF as an
  incremental update (the ``/ODKSignature`` document-info entry), so the signed bytes
  are an untouched prefix of the file -- the same idea as a PDF signature's ByteRange.
  A detached ``signature.json`` next to the PDF signs the final PDF and HTML files.

Keys
----
One key pair per user installation, created on first use under
``<data root>/keys/`` (``core.paths.data_root``; ``ODK_DATA_DIR`` moves it). The
private key never leaves that folder and is never written into a report; the public
key is embedded so anyone can check the signature, and the fingerprint printed on the
report says *which* key signed it. A valid signature by a key this installation does
not know is reported as untrusted, not as verified: anyone can make a key.

What verification does not prove: that the inputs were captured as described, or that
the report meets any standard. It proves the report and its manifest are the ones this
key signed, and -- where the inputs are reachable -- that they still hash the same.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FORMAT = "odk-report-manifest/1"
SIGNATURE_FORMAT = "odk-report-signature/1"
DETACHED_FORMAT = "odk-report-files-signature/1"
MANIFEST_ATTACHMENT = "odk-manifest.json"
CONTENT_ATTACHMENT = "odk-report-content.json"
SIGNATURE_INFO_KEY = "/ODKSignature"
PRIVATE_KEY_NAME = "report_signing_ed25519.pem"
PUBLIC_KEY_NAME = "report_signing_ed25519.pub.pem"
TRUSTED_DIR_NAME = "trusted_report_keys"


class SigningUnavailable(RuntimeError):
    """The signing library is not installed, so a report cannot be signed here."""


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise SigningUnavailable(
            "Report signing needs the 'cryptography' package (pip install -r "
            "requirements.txt); no signature was made.") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def canonical(value: Any) -> bytes:
    """The one byte form a statement is signed in: sorted keys, no whitespace, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Keys ──────────────────────────────────────────────────────────────────────

def keys_dir() -> Path:
    from .paths import data_root

    return data_root() / "keys"


def fingerprint(public_raw: bytes) -> str:
    """A short, stable name for a public key: SHA-256 of its 32 raw bytes, grouped."""
    digest = hashlib.sha256(public_raw).hexdigest()[:32]
    return ":".join(digest[i:i + 4] for i in range(0, 32, 4))


def _public_raw(public_key) -> bytes:
    serialization, _priv, _pub = _crypto()
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def load_or_create_key() -> tuple[Any, dict[str, Any]]:
    """This installation's signing key, created the first time it is needed."""
    serialization, Ed25519PrivateKey, _pub = _crypto()
    folder = keys_dir()
    private_path = folder / PRIVATE_KEY_NAME
    public_path = folder / PUBLIC_KEY_NAME
    created = False
    if private_path.is_file():
        key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{private_path} is not an Ed25519 private key; refusing to "
                             "sign with it.")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        # Exclusive create: two first uses at once must not overwrite each other's key.
        fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(pem)
        created = True
    public_pem = key.public_key().public_bytes(serialization.Encoding.PEM,
                                               serialization.PublicFormat.SubjectPublicKeyInfo)
    if not public_path.is_file() or public_path.read_bytes() != public_pem:
        public_path.write_bytes(public_pem)
    raw = _public_raw(key.public_key())
    return key, {
        "created": created,
        "algorithm": "Ed25519",
        "fingerprint": fingerprint(raw),
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "private_key_path": str(private_path),
        "public_key_path": str(public_path),
    }


def key_info() -> dict[str, Any]:
    """The installation key's public details (creates the key if there is none yet)."""
    _key, info = load_or_create_key()
    return info


def trusted_fingerprints() -> dict[str, str]:
    """fingerprint -> where it came from: this installation's key plus imported keys."""
    serialization, _priv, Ed25519PublicKey = _crypto()
    trusted: dict[str, str] = {}
    own = keys_dir() / PUBLIC_KEY_NAME
    candidates = [(own, "this installation's key")] if own.is_file() else []
    folder = keys_dir() / TRUSTED_DIR_NAME
    if folder.is_dir():
        candidates += [(p, f"trusted key {p.name}") for p in sorted(folder.glob("*.pem"))]
    for path, label in candidates:
        try:
            key = serialization.load_pem_public_key(path.read_bytes())
        except Exception:  # noqa: BLE001 - an unreadable file trusts nothing
            continue
        if isinstance(key, Ed25519PublicKey):
            trusted[fingerprint(_public_raw(key))] = label
    return trusted


def trust_public_key(public_pem_path: str | Path, name: str = "") -> dict[str, Any]:
    """Accept another signer's public key, so reports it signed verify as trusted."""
    serialization, _priv, Ed25519PublicKey = _crypto()
    data = Path(public_pem_path).read_bytes()
    key = serialization.load_pem_public_key(data)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{public_pem_path} is not an Ed25519 public key.")
    fp = fingerprint(_public_raw(key))
    folder = keys_dir() / TRUSTED_DIR_NAME
    folder.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in (name or fp.replace(":", "")) if c.isalnum() or c in "-_")
    target = folder / f"{stem or 'key'}.pem"
    target.write_bytes(data)
    return {"fingerprint": fp, "path": str(target)}


# ── Manifest ──────────────────────────────────────────────────────────────────

def build_manifest(inputs: Iterable[dict[str, Any]], models: Iterable[dict[str, Any]] = (),
                   report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hash every input; record every model digest.

    ``inputs``: ``{"path", "role"}``. A path that does not exist is refused: a manifest
    must not list something the report could not have read.
    ``models``: ``{"name", "path"}`` (hashed here) or ``{"name", "sha256",
    "digest_source"}`` when only a recorded digest exists (the file is not on this
    machine); the entry says which.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in inputs:
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input {path} does not exist or is not a file.")
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        entries.append({
            "role": str(item.get("role") or "input"),
            "name": path.name,
            "path": key,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    model_entries: list[dict[str, Any]] = []
    for model in models:
        name = str(model.get("name") or "")
        path = str(model.get("path") or "")
        if path:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Model file {path} does not exist.")
            model_entries.append({"name": name or Path(path).name, "path": str(Path(path).resolve()),
                                  "sha256": sha256_file(path),
                                  "digest_source": "hashed from the model file"})
        elif model.get("sha256"):
            model_entries.append({"name": name, "path": "", "sha256": str(model["sha256"]),
                                  "digest_source": str(model.get("digest_source") or
                                                       "recorded digest; file not re-hashed")})
        else:
            raise ValueError(f"Model {name!r} has neither a file nor a recorded digest.")
    return {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report": dict(report or {}),
        "hash_algorithm": "SHA-256",
        "inputs": entries,
        "models": model_entries,
    }


# ── Signing ───────────────────────────────────────────────────────────────────

def _sign(statement: dict[str, Any]) -> dict[str, Any]:
    key, info = load_or_create_key()
    signature = key.sign(canonical(statement))
    return {
        "statement": statement,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
        "public_key_b64": info["public_key_b64"],
        "fingerprint": info["fingerprint"],
        "algorithm": "Ed25519",
    }


def embed_attachments(pdf_path: str | Path, attachments: dict[str, bytes]) -> None:
    """Rewrite a PDF with JSON attachments (done before signing, so they are signed)."""
    import pypdf

    pdf_path = Path(pdf_path)
    writer = pypdf.PdfWriter(clone_from=str(pdf_path))
    for name, data in attachments.items():
        writer.add_attachment(name, data)
    tmp = pdf_path.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as handle:
        writer.write(handle)
    os.replace(tmp, pdf_path)


def sign_pdf(pdf_path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Append an Ed25519 signature over the PDF's current bytes and its manifest."""
    import pypdf

    pdf_path = Path(pdf_path)
    prefix = pdf_path.read_bytes()
    statement = {
        "format": SIGNATURE_FORMAT,
        "signed_length": len(prefix),
        "pdf_prefix_sha256": sha256_bytes(prefix),
        "manifest_sha256": sha256_bytes(canonical(manifest)),
        "signed_at": datetime.now(timezone.utc).isoformat(),
    }
    signed = _sign(statement)
    writer = pypdf.PdfWriter(str(pdf_path), incremental=True)
    writer.add_metadata({SIGNATURE_INFO_KEY: json.dumps(signed, sort_keys=True)})
    tmp = pdf_path.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as handle:
        writer.write(handle)
    if not tmp.read_bytes().startswith(prefix):  # pragma: no cover - pypdf contract
        tmp.unlink()
        raise RuntimeError("The signature update rewrote the signed bytes; not signed.")
    os.replace(tmp, pdf_path)
    return signed


def sign_files(folder: str | Path, names: Iterable[str], manifest: dict[str, Any]) -> Path:
    """A detached signature.json over the final files (PDF, HTML) in a report folder."""
    folder = Path(folder)
    files = {name: sha256_file(folder / name) for name in names if (folder / name).is_file()}
    statement = {
        "format": DETACHED_FORMAT,
        "files": files,
        "manifest_sha256": sha256_bytes(canonical(manifest)),
        "signed_at": datetime.now(timezone.utc).isoformat(),
    }
    target = folder / "signature.json"
    target.write_text(json.dumps(_sign(statement), indent=2, sort_keys=True), encoding="utf-8")
    return target


# ── Verification ──────────────────────────────────────────────────────────────

def _check_signature(signed: dict[str, Any]) -> tuple[bool, str]:
    serialization, _priv, Ed25519PublicKey = _crypto()
    from cryptography.exceptions import InvalidSignature

    try:
        raw = base64.b64decode(signed["public_key_b64"])
        public = Ed25519PublicKey.from_public_bytes(raw)
        public.verify(base64.b64decode(signed["signature_b64"]), canonical(signed["statement"]))
    except InvalidSignature:
        return False, "The Ed25519 signature does not match the signed statement."
    except Exception as exc:  # noqa: BLE001 - malformed signature block
        return False, f"The signature block is malformed: {exc}"
    if fingerprint(raw) != signed.get("fingerprint"):
        return False, "The printed key fingerprint does not belong to the embedded key."
    return True, ""


def check_inputs(manifest: dict[str, Any], inputs_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Re-hash each manifest input that can be found here.

    Looked for at its recorded path, then (with ``inputs_root``) by file name under that
    folder -- for a report checked on another machine. Status: ``match``, ``changed``
    or ``not_found`` (cannot be checked here; not a failure of the report).
    """
    by_name: dict[str, list[Path]] = {}
    if inputs_root and Path(inputs_root).is_dir():
        for path in Path(inputs_root).rglob("*"):
            if path.is_file():
                by_name.setdefault(path.name, []).append(path)
    out = []
    for entry in manifest.get("inputs") or []:
        recorded = Path(str(entry.get("path", "")))
        candidates = [recorded] if recorded.is_file() else by_name.get(str(entry.get("name")), [])
        status, where = "not_found", ""
        for candidate in candidates:
            where = str(candidate)
            if sha256_file(candidate) == entry.get("sha256"):
                status = "match"
                break
            status = "changed"
        out.append({"name": entry.get("name"), "role": entry.get("role"), "status": status,
                    "checked_path": where, "sha256": entry.get("sha256")})
    return out


def _attachments(reader) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for name, blobs in (reader.attachments or {}).items():
        if blobs:
            found[name] = blobs[0]
    return found


def verify_pdf(pdf_path: str | Path, *, check_input_files: bool = True,
               inputs_root: str | Path | None = None) -> dict[str, Any]:
    """Check a signed report PDF: signature, untouched bytes, manifest, inputs."""
    import io

    import pypdf

    path = Path(pdf_path)
    problems: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {"path": str(path), "kind": "pdf"}
    data = path.read_bytes()
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        raw = (reader.metadata or {}).get(SIGNATURE_INFO_KEY)
    except Exception as exc:  # noqa: BLE001 - unreadable PDF
        return {**result, "ok": False, "signature_valid": False,
                "problems": [f"Not a readable PDF: {exc}"], "warnings": []}
    if not raw:
        return {**result, "ok": False, "signature_valid": False,
                "problems": ["This PDF carries no OpenDroneKit signature."], "warnings": []}
    try:
        signed = json.loads(str(raw))
        statement = signed["statement"]
    except Exception as exc:  # noqa: BLE001
        return {**result, "ok": False, "signature_valid": False,
                "problems": [f"The signature entry is not readable JSON: {exc}"], "warnings": []}

    valid, why = _check_signature(signed)
    if not valid:
        problems.append(why)
    if statement.get("format") != SIGNATURE_FORMAT:
        problems.append(f"Unknown signature format {statement.get('format')!r}.")
    length = int(statement.get("signed_length") or 0)
    prefix = data[:length]
    if length <= 0 or length > len(data) or sha256_bytes(prefix) != statement.get("pdf_prefix_sha256"):
        problems.append("The signed part of the PDF has been altered since it was signed.")
    tail = data[length:]
    if tail.count(b"%%EOF") != 1 or not tail.rstrip().endswith(b"%%EOF"):
        problems.append("The PDF was changed after signing (content appended after the "
                        "signature update).")

    manifest: dict[str, Any] = {}
    content: dict[str, Any] = {}
    try:
        attachments = _attachments(pypdf.PdfReader(io.BytesIO(prefix)))
        manifest = json.loads(attachments[MANIFEST_ATTACHMENT].decode("utf-8"))
        if CONTENT_ATTACHMENT in attachments:
            content = json.loads(attachments[CONTENT_ATTACHMENT].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"The embedded manifest cannot be read from the signed part: {exc}")
    if manifest and sha256_bytes(canonical(manifest)) != statement.get("manifest_sha256"):
        problems.append("The embedded manifest is not the one that was signed.")

    trusted = trusted_fingerprints()
    signer = signed.get("fingerprint", "")
    signer_trusted = signer in trusted
    if valid and not signer_trusted:
        problems.append(f"The signature is mathematically valid but key {signer} is not "
                        "trusted on this installation (not this installation's key and not "
                        "imported with trust_public_key). Anyone can make a key.")

    inputs: list[dict[str, Any]] = []
    if manifest and check_input_files:
        inputs = check_inputs(manifest, inputs_root)
        changed = [i["name"] for i in inputs if i["status"] == "changed"]
        missing = [i["name"] for i in inputs if i["status"] == "not_found"]
        if changed:
            problems.append(f"{len(changed)} input(s) no longer match their manifest hash: "
                            + ", ".join(map(str, changed[:10])))
        if missing:
            warnings.append(f"{len(missing)} input(s) are not on this machine and were not "
                            "re-hashed.")
    return {
        **result,
        "ok": not problems,
        "signature_valid": valid,
        "signer_fingerprint": signer,
        "signer_trusted": signer_trusted,
        "signer": trusted.get(signer, "unknown key"),
        "signed_at": statement.get("signed_at"),
        "manifest": {"inputs": len(manifest.get("inputs") or []),
                     "models": len(manifest.get("models") or []),
                     "report": manifest.get("report") or {}},
        "missing_fields": (content.get("completeness") or {}).get("missing"),
        "inputs": inputs,
        "problems": problems,
        "warnings": warnings,
    }


def verify_folder(folder: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Check a report folder: the PDF's own signature and the detached file signature."""
    folder = Path(folder)
    pdf = folder / "report.pdf"
    if not pdf.is_file():
        return {"path": str(folder), "kind": "folder", "ok": False, "signature_valid": False,
                "problems": ["No report.pdf in this folder."], "warnings": []}
    result = verify_pdf(pdf, **kwargs)
    result["kind"] = "folder"
    detached = folder / "signature.json"
    if not detached.is_file():
        result["warnings"].append("No detached signature.json; the HTML copy is unchecked.")
        return result
    signed = json.loads(detached.read_text(encoding="utf-8"))
    valid, why = _check_signature(signed)
    files = {}
    if not valid:
        result["problems"].append("signature.json: " + why)
    elif signed.get("fingerprint") != result.get("signer_fingerprint"):
        result["problems"].append("signature.json was signed by a different key than the PDF.")
    for name, digest in (signed.get("statement", {}).get("files") or {}).items():
        target = folder / name
        status = ("missing" if not target.is_file()
                  else "match" if sha256_file(target) == digest else "changed")
        files[name] = status
        if status != "match":
            result["problems"].append(f"{name} is {status} since it was signed.")
    result["files"] = files
    result["ok"] = not result["problems"]
    return result


def verify_report(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Verify a signed report given its PDF or its folder."""
    target = Path(path)
    if target.is_dir():
        return verify_folder(target, **kwargs)
    if not target.is_file():
        return {"path": str(target), "ok": False, "signature_valid": False,
                "problems": [f"{target} does not exist."], "warnings": []}
    return verify_pdf(target, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """``python -m core.report_signing verify <report.pdf|folder> [--inputs-root DIR]``."""
    import argparse

    parser = argparse.ArgumentParser(prog="odk-report-verify",
                                     description="Verify a signed OpenDroneKit report.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="check signature, manifest and inputs")
    verify.add_argument("path")
    verify.add_argument("--inputs-root", default=None,
                        help="folder to look for inputs that are not at their recorded path")
    verify.add_argument("--no-inputs", action="store_true", help="skip re-hashing inputs")
    verify.add_argument("--json", action="store_true", help="print the full result as JSON")
    sub.add_parser("key", help="show (and create on first use) this installation's key")
    args = parser.parse_args(argv)
    if args.command == "key":
        print(json.dumps(key_info(), indent=2))
        return 0
    result = verify_report(args.path, check_input_files=not args.no_inputs,
                           inputs_root=args.inputs_root)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(("VERIFIED" if result["ok"] else "NOT VERIFIED") + f": {result['path']}")
        if result.get("signer_fingerprint"):
            print(f"  signer: {result['signer_fingerprint']} ({result.get('signer')})")
        for item in result.get("inputs") or []:
            print(f"  input {item['status']:>9}: {item['name']}")
        for problem in result.get("problems") or []:
            print(f"  PROBLEM: {problem}")
        for warning in result.get("warnings") or []:
            print(f"  warning: {warning}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
