"""Recording where a derived file came from, and proving it is still that file.

A survey produces artifacts that outlive the run: an orthomosaic goes into a report, a
DSM is measured months later, a defect layer is handed to a client. By then nobody
remembers which imagery produced it, which engine, in which CRS, at which resolution.
Without that, a number taken off the raster cannot be defended, and two artifacts from
different runs cannot be told apart.

So each derived file gets a sidecar recording its inputs, the engine and version that
produced it, its CRS, and the parameters that shaped it.

The part that makes this more than decoration is the checksum. A provenance record that
merely asserts a lineage is worth little, because the file may have been replaced,
re-exported, or edited since. Every record stores the artifact's sha256, and
``verify`` recomputes it, so a claim about a file can be checked against the file
rather than believed. A record whose digest no longer matches is reported as stale, not
quietly trusted.

Sidecars are written next to the artifact as ``<name>.provenance.json`` so they travel
with the file when it is copied, rather than living in a database the recipient will
not have.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SIDECAR_SUFFIX = ".provenance.json"

# Read in blocks: an orthomosaic can be gigabytes, and hashing it should not require
# holding it in memory.
_HASH_BLOCK = 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Provenance:
    """Where one derived artifact came from."""

    artifact: str
    sha256: str
    size_bytes: int
    engine: str
    engine_version: str = ""
    sources: list[str] = field(default_factory=list)
    source_count: int = 0
    crs_epsg: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=_now)
    software: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "sources": self.sources,
            "source_count": self.source_count,
            "crs_epsg": self.crs_epsg,
            "parameters": self.parameters,
            "created_utc": self.created_utc,
            "software": self.software,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Provenance":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def _software_versions() -> dict[str, str]:
    """Record the versions that produced the file, since results move between them."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module, key in (("rasterio", "rasterio"), ("pycolmap", "pycolmap"),
                        ("numpy", "numpy"), ("cv2", "opencv")):
        try:
            imported = __import__(module)
            versions[key] = str(getattr(imported, "__version__", "unknown"))
        except Exception:
            # An absent library is not an error here; its absence is itself a fact
            # worth not recording as a version.
            continue
    return versions


def sidecar_path(artifact: str | Path) -> Path:
    return Path(str(artifact) + SIDECAR_SUFFIX)


def record(artifact: str | Path, *, engine: str, sources: Sequence[str | Path] = (),
           crs_epsg: int | None = None, parameters: dict[str, Any] | None = None,
           engine_version: str = "", notes: str = "",
           max_sources_listed: int = 500) -> Provenance:
    """Write a provenance sidecar for a derived file.

    Source lists are capped: a survey may have thousands of frames, and a sidecar
    larger than the artifact helps nobody. The full count is always recorded, so a
    truncated list never reads as the complete input set.
    """
    path = Path(artifact)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot record provenance for {path}: the artifact does not exist. "
            "Write the file first, then record what produced it."
        )

    source_list = [str(s) for s in sources]
    listed = source_list[:max_sources_listed]
    entry_notes = notes
    if len(source_list) > max_sources_listed:
        entry_notes = (
            f"{entry_notes} " if entry_notes else ""
        ) + (
            f"Source list truncated to {max_sources_listed} of {len(source_list)} "
            "entries; source_count is the true total."
        )

    entry = Provenance(
        artifact=path.name,
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        engine=engine,
        engine_version=engine_version,
        sources=listed,
        source_count=len(source_list),
        crs_epsg=crs_epsg,
        parameters=dict(parameters or {}),
        software=_software_versions(),
        notes=entry_notes.strip(),
    )

    sidecar_path(path).write_text(
        json.dumps(entry.to_dict(), indent=2), encoding="utf-8"
    )
    return entry


def read(artifact: str | Path) -> Provenance | None:
    """Read the sidecar for an artifact, or None when it has none."""
    sidecar = sidecar_path(artifact)
    if not sidecar.exists():
        return None
    try:
        return Provenance.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def verify(artifact: str | Path) -> dict[str, Any]:
    """Check an artifact still matches the provenance recorded for it.

    This is the whole point of storing a digest. A record that cannot be checked is a
    claim; one that can is evidence.
    """
    path = Path(artifact)
    entry = read(path)

    if entry is None:
        return {
            "ok": False,
            "status": "no_provenance",
            "detail": (
                f"{path.name} has no provenance record, so its inputs, engine and CRS "
                "are unknown. Treat any measurement from it as unattributed."
            ),
        }

    if not path.exists():
        return {
            "ok": False,
            "status": "missing_artifact",
            "detail": f"{path.name} is recorded but no longer present at {path}.",
        }

    actual = sha256_of(path)
    if actual != entry.sha256:
        return {
            "ok": False,
            "status": "modified",
            "detail": (
                f"{path.name} has changed since its provenance was recorded. The "
                "recorded lineage describes a different file, so it no longer applies."
            ),
            "recorded_sha256": entry.sha256,
            "actual_sha256": actual,
        }

    return {
        "ok": True,
        "status": "verified",
        "detail": f"{path.name} matches the provenance recorded on {entry.created_utc}.",
        "engine": entry.engine,
        "crs_epsg": entry.crs_epsg,
        "source_count": entry.source_count,
    }


def record_reconstruction_outputs(output_dir: str | Path, *, engine: str,
                                  image_dir: str | Path,
                                  crs_epsg: int | None = None,
                                  parameters: dict[str, Any] | None = None,
                                  engine_version: str = "") -> dict[str, Any]:
    """Record provenance for every derived file a reconstruction produced."""
    out = Path(output_dir)
    if not out.is_dir():
        raise NotADirectoryError(f"{output_dir} is not a reconstruction output folder.")

    images = sorted(Path(image_dir).glob("*")) if Path(image_dir).is_dir() else []
    sources = [p.name for p in images if p.is_file()]

    recorded: dict[str, Any] = {}
    for artifact in sorted(out.iterdir()):
        if not artifact.is_file() or artifact.name.endswith(SIDECAR_SUFFIX):
            continue
        entry = record(
            artifact, engine=engine, sources=sources, crs_epsg=crs_epsg,
            parameters=parameters, engine_version=engine_version,
        )
        recorded[artifact.name] = entry.sha256

    return {
        "output_dir": str(out),
        "recorded": recorded,
        "artifact_count": len(recorded),
        "source_count": len(sources),
    }


def audit(directory: str | Path) -> dict[str, Any]:
    """Verify every artifact in a folder against its recorded provenance."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"{directory} is not a folder.")

    results: dict[str, Any] = {}
    for artifact in sorted(root.iterdir()):
        if not artifact.is_file() or artifact.name.endswith(SIDECAR_SUFFIX):
            continue
        results[artifact.name] = verify(artifact)

    verified = [n for n, r in results.items() if r["ok"]]
    unattributed = [n for n, r in results.items() if r["status"] == "no_provenance"]
    modified = [n for n, r in results.items() if r["status"] == "modified"]

    return {
        "directory": str(root),
        "total": len(results),
        "verified": verified,
        "unattributed": unattributed,
        "modified": modified,
        "ok": not modified and not unattributed,
        "results": results,
    }
