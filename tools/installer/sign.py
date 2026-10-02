"""Code-signing step for the installer: Azure Trusted Signing or a certificate, else UNSIGNED.

Configuration is by environment variable, so no secret or certificate ever lives in
the repository:

Azure Trusted Signing (``ODK_SIGN_MODE=azure``, or auto when both paths are set)
    ODK_AZURE_SIGNING_DLIB      path to Azure.CodeSigning.Dlib.dll (Trusted Signing client)
    ODK_AZURE_SIGNING_METADATA  path to metadata.json {Endpoint, CodeSigningAccountName,
                                CertificateProfileName}; credentials come from the usual
                                Azure identity chain (az login, env vars, managed identity)

A certificate with signtool (``ODK_SIGN_MODE=signtool``, or auto when one is set)
    ODK_SIGN_CERT_THUMBPRINT    SHA-1 thumbprint of a cert in the user/machine store, or
    ODK_SIGN_PFX                path to a .pfx, with ODK_SIGN_PFX_PASSWORD in the environment

    ODK_SIGNTOOL                signtool.exe, if it is not found in the Windows SDK
    ODK_SIGN_TIMESTAMP_URL      RFC 3161 timestamp server (sensible default per mode)

With none of these the build is UNSIGNED: the step says so, signs nothing, and the
installer's file name, version info and manifest all carry the word.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

AZURE_TIMESTAMP = "http://timestamp.acs.microsoft.com"
CERT_TIMESTAMP = "http://timestamp.digicert.com"


@dataclass
class SignPlan:
    mode: str                      # "azure" | "signtool" | "none"
    reason: str
    signtool: str = ""
    args: list[str] = field(default_factory=list)   # everything between `sign` and the file

    @property
    def will_sign(self) -> bool:
        return self.mode != "none"

    def command(self, target: str | Path) -> list[str]:
        return [self.signtool, "sign", *self.args, str(target)]

    def inno_signtool(self) -> str:
        """The command line Inno Setup's /S switch takes, with $f for the file."""
        quoted = " ".join(f'"{a}"' if " " in a else a for a in self.args)
        return f'"{self.signtool}" sign {quoted} $f'


def find_signtool(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    explicit = env.get("ODK_SIGNTOOL", "")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("signtool")
    if found:
        return found
    kits = env.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    candidates = sorted(glob.glob(os.path.join(kits, "Windows Kits", "10", "bin", "*", "x64", "signtool.exe")))
    return candidates[-1] if candidates else ""


def plan_signing(env: Mapping[str, str] | None = None, signtool: str | None = None) -> SignPlan:
    env = os.environ if env is None else env
    mode = env.get("ODK_SIGN_MODE", "auto").strip().lower() or "auto"
    dlib = env.get("ODK_AZURE_SIGNING_DLIB", "")
    metadata = env.get("ODK_AZURE_SIGNING_METADATA", "")
    thumb = env.get("ODK_SIGN_CERT_THUMBPRINT", "").replace(" ", "")
    pfx = env.get("ODK_SIGN_PFX", "")

    if mode == "auto":
        mode = "azure" if (dlib and metadata) else "signtool" if (thumb or pfx) else "none"
    if mode == "none":
        return SignPlan("none", "UNSIGNED: no signing certificate is configured (set "
                                "ODK_AZURE_SIGNING_DLIB + ODK_AZURE_SIGNING_METADATA for Azure "
                                "Trusted Signing, or ODK_SIGN_CERT_THUMBPRINT / ODK_SIGN_PFX for "
                                "signtool). Signing skipped.")
    tool = find_signtool(env) if signtool is None else signtool
    if not tool:
        return SignPlan("none", f"UNSIGNED: signing mode {mode!r} is configured but signtool.exe "
                                "was not found (install the Windows SDK or set ODK_SIGNTOOL). "
                                "Signing skipped.")
    if mode == "azure":
        missing = [n for n, v in (("ODK_AZURE_SIGNING_DLIB", dlib),
                                  ("ODK_AZURE_SIGNING_METADATA", metadata)) if not v]
        if missing:
            return SignPlan("none", f"UNSIGNED: Azure Trusted Signing needs {', '.join(missing)}. "
                                    "Signing skipped.")
        ts = env.get("ODK_SIGN_TIMESTAMP_URL", AZURE_TIMESTAMP)
        return SignPlan("azure", "Azure Trusted Signing", tool,
                        ["/v", "/fd", "SHA256", "/tr", ts, "/td", "SHA256",
                         "/dlib", dlib, "/dmdf", metadata])
    if mode == "signtool":
        ts = env.get("ODK_SIGN_TIMESTAMP_URL", CERT_TIMESTAMP)
        base = ["/fd", "SHA256", "/tr", ts, "/td", "SHA256"]
        if thumb:
            return SignPlan("signtool", f"certificate {thumb[:8]}...", tool, [*base, "/sha1", thumb])
        if pfx:
            password = env.get("ODK_SIGN_PFX_PASSWORD", "")
            args = [*base, "/f", pfx] + (["/p", password] if password else [])
            return SignPlan("signtool", f"certificate file {Path(pfx).name}", tool, args)
        return SignPlan("none", "UNSIGNED: ODK_SIGN_MODE=signtool but neither "
                                "ODK_SIGN_CERT_THUMBPRINT nor ODK_SIGN_PFX is set. Signing skipped.")
    return SignPlan("none", f"UNSIGNED: unknown ODK_SIGN_MODE {mode!r}. Signing skipped.")


def sign_file(plan: SignPlan, target: Path) -> dict[str, object]:
    if not plan.will_sign:
        return {"signed": False, "reason": plan.reason}
    proc = subprocess.run(plan.command(target), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"signing {target} failed: {proc.stdout}\n{proc.stderr}")
    verify = subprocess.run([plan.signtool, "verify", "/pa", str(target)], capture_output=True, text=True)
    if verify.returncode != 0:
        raise RuntimeError(f"signature on {target} does not verify: {verify.stdout}\n{verify.stderr}")
    return {"signed": True, "reason": plan.reason}
