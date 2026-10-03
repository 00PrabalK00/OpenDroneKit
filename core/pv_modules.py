"""PV module specifications from the CEC module list shipped with NREL SAM.

The operator types or picks the module model; this module returns what the public list
states for it (rated STC power, cell technology, cell count, dimensions, electrical
ratings) and the efficiency computed from it. Every returned record carries a
``source`` string naming the list, its version and the computation, so a report can
cite it. A model that is not in the list is not guessed: ``lookup`` raises
``UnknownModule`` and ``search`` offers candidates for the operator to pick from.

Data: ``core/data/pv_modules/cec_modules.csv.gz`` (provenance and the SAM BSD-3 licence
in ``core/data/pv_modules/PROVENANCE.md``).
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data" / "pv_modules" / "cec_modules.csv.gz"
DATABASE_NAME = "CEC PV module list (NREL SAM library)"


class UnknownModule(KeyError):
    """The model is not in the list; the operator must pick a listed model or enter specs."""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


@lru_cache(maxsize=1)
def _rows() -> tuple[list[dict[str, str]], dict[str, str]]:
    raw = gzip.decompress(DATA.read_bytes()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    units = next(reader)
    return list(reader), units


def database_info() -> dict[str, Any]:
    rows, _ = _rows()
    versions = sorted({r["Version"] for r in rows if r["Version"]})
    return {"name": DATABASE_NAME, "modules": len(rows), "version": ", ".join(versions),
            "path": str(DATA), "licence": "BSD-3-Clause (NREL SAM); data from the California "
            "Energy Commission"}


def _num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record(row: dict[str, str]) -> dict[str, Any]:
    stc, area = _num(row["STC"]), _num(row["A_c"])
    efficiency = round(stc / (area * 1000.0) * 100.0, 2) if stc and area else None
    version = row.get("Version") or "?"
    return {
        "name": row["Name"], "manufacturer": row["Manufacturer"],
        "technology": row["Technology"], "bifacial": row["Bifacial"] == "1",
        "stc_w": stc, "ptc_w": _num(row["PTC"]), "area_m2": area,
        "length_m": _num(row["Length"]), "width_m": _num(row["Width"]),
        "cells_in_series": int(float(row["N_s"])) if _num(row["N_s"]) else None,
        "voc_v": _num(row["V_oc_ref"]), "isc_a": _num(row["I_sc_ref"]),
        "vmp_v": _num(row["V_mp_ref"]), "imp_a": _num(row["I_mp_ref"]),
        "gamma_pmp_pct_per_k": _num(row["gamma_pmp"]),
        "efficiency_pct": efficiency,
        "source": f"{DATABASE_NAME}, version {version}, entry {row['Name']!r}",
        "efficiency_source": (f"computed as STC {stc:g} W / (A_c {area:g} m² x 1000 W/m²) "
                              f"from {DATABASE_NAME} {version}") if efficiency else None,
    }


def search(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Listed modules whose name contains every token of ``query`` (case/punctuation-blind)."""
    tokens = [_norm(t) for t in str(query or "").split() if _norm(t)]
    if not tokens:
        return []
    rows, _ = _rows()
    hits = [r for r in rows if all(t in _norm(r["Name"]) for t in tokens)]
    return [_record(r) for r in hits[:max(1, int(limit))]]


def lookup(model: str) -> dict[str, Any]:
    """The listed module named exactly ``model`` (ignoring case and punctuation)."""
    key = _norm(model)
    if not key:
        raise UnknownModule("No module model given.")
    rows, _ = _rows()
    exact = [r for r in rows if _norm(r["Name"]) == key]
    if len(exact) == 1:
        return _record(exact[0])
    if len(exact) > 1:
        raise UnknownModule(f"{model!r} matches {len(exact)} listed entries; pick one by its "
                            "full listed name.")
    raise UnknownModule(f"{model!r} is not in the {DATABASE_NAME}. Search for the listed name "
                        "or enter the datasheet values by hand.")


def describe(record: dict[str, Any]) -> str:
    """One line for the report's component-type item."""
    parts = [record["name"], f"{record['technology']}"]
    if record.get("cells_in_series"):
        parts.append(f"{record['cells_in_series']} cells in series")
    if record.get("length_m") and record.get("width_m"):
        parts.append(f"{record['length_m']:g} x {record['width_m']:g} m")
    if record.get("bifacial"):
        parts.append("bifacial")
    return ", ".join(parts)
