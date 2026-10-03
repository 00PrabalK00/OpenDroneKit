"""What the operator knows, recorded once and reused: people, procedures, site conditions.

Four stores, each a small JSON file, each value carrying who saved it and when:

* **Operator profiles** (per installation, ``<data root>/settings/operator_profiles.json``):
  a person's name, role, qualification / certification level and certificate number.
  Saved once, picked per inspection; they fill the report's personnel items.
* **Procedure templates** (per installation, ``.../procedure_templates.json``): the
  procedure text and inspection type for a standard (optionally per mission template).
  Built-in defaults exist, but a default is a *suggestion*: it fills nothing until a named
  person adopts or edits it, and the report then cites that person and date.
* **Site conditions log** (per project, ``<project>/standards/site_conditions.json``): a
  structured on-site record per flight (cloud oktas and type, wind, air temperature,
  notes) captured in Field mode, or imported from a meteo CSV / flight-log telemetry with
  the file and row named as its source.
* **Inspection details** (per project, ``<project>/standards/inspection_details/<template>.json``):
  the per-inspection choices (which profiles fill which role, client, contract scope,
  module model, camera-to-target distance, ...) plus CoA confirmations from review.

Nothing here invents a value. Missing input stays missing and is listed on the report's
"Before this report can be issued" page.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _settings_dir() -> Path:
    from .paths import data_root

    target = data_root() / "settings"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _who(name: Any, what: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError(f"{what} must name who is saving it.")
    return text


# ── Operator profiles ─────────────────────────────────────────────────────────

PROFILE_FIELDS = ("name", "role", "qualification", "certificate_number", "organisation",
                  "contact", "notes")
ROLES = ("thermographer", "pv_expert", "bridge_inspector", "roof_inspector",
         "tower_assessor", "pilot", "attending", "other")


def _profiles_path() -> Path:
    return _settings_dir() / "operator_profiles.json"


def list_operator_profiles() -> list[dict[str, Any]]:
    return list(_read(_profiles_path(), {"profiles": []}).get("profiles") or [])


def get_operator_profile(profile_id: str) -> dict[str, Any]:
    for profile in list_operator_profiles():
        if profile["id"] == profile_id:
            return profile
    raise ValueError(f"No operator profile {profile_id!r}.")


def save_operator_profile(profile: dict[str, Any], saved_by: str) -> dict[str, Any]:
    """Add or update a person. ``name`` is required; ``id`` updates an existing one."""
    saved_by = _who(saved_by, "An operator profile")
    unknown = set(profile) - set(PROFILE_FIELDS) - {"id"}
    if unknown:
        raise ValueError(f"Unknown profile field(s): {', '.join(sorted(unknown))}.")
    clean = {k: str(profile.get(k) or "").strip() for k in PROFILE_FIELDS}
    if not clean["name"]:
        raise ValueError("An operator profile needs a name.")
    if clean["role"] and clean["role"] not in ROLES:
        raise ValueError(f"Unknown role {clean['role']!r}; use one of {', '.join(ROLES)}.")
    profiles = list_operator_profiles()
    pid = str(profile.get("id") or "").strip()
    if pid and not any(p["id"] == pid for p in profiles):
        raise ValueError(f"No operator profile {pid!r} to update.")
    record = {"id": pid or f"op_{uuid.uuid4().hex[:10]}", **clean,
              "saved_by": saved_by, "saved_at": _now()}
    profiles = [p for p in profiles if p["id"] != record["id"]] + [record]
    _write(_profiles_path(), {"profiles": profiles})
    return record


def delete_operator_profile(profile_id: str) -> bool:
    profiles = list_operator_profiles()
    kept = [p for p in profiles if p["id"] != profile_id]
    if len(kept) == len(profiles):
        return False
    _write(_profiles_path(), {"profiles": kept})
    return True


def describe_person(profile: dict[str, Any]) -> str:
    parts = [profile["name"]]
    if profile.get("qualification"):
        parts.append(profile["qualification"])
    if profile.get("certificate_number"):
        parts.append(f"cert. {profile['certificate_number']}")
    if profile.get("organisation"):
        parts.append(profile["organisation"])
    return ", ".join(parts)


def profile_source(profile: dict[str, Any]) -> str:
    return (f"operator profile '{profile['name']}' (saved by {profile['saved_by']} "
            f"{profile['saved_at'][:10]})")


# ── Procedure templates ───────────────────────────────────────────────────────

# Default procedure texts. They describe how OpenDroneKit's own workflow for each
# standard is run, and are offered to the operator to adopt or edit. Until a named person
# adopts one it is a suggestion and fills no report item.
BUILTIN_PROCEDURES: dict[str, dict[str, str]] = {
    "iec62446_3": {
        "inspection_type": "simplified (clause 7: aerial overview to locate thermal "
                           "abnormalities; detailed follow-up on the ground where needed)",
        "text": ("UAV-borne radiometric infrared survey of the PV array in operation under "
                 "steady irradiance. Frames captured at a viewing angle within the limits of "
                 "clause 6 to avoid reflections, with paired visual images. Temperatures "
                 "decoded from the radiometric files; each module's median temperature "
                 "compared with the median of the modules in the same frame (ΔT, clause 8 s). "
                 "Irradiance, air temperature and wind read from the site logger at the "
                 "capture time. Abnormalities classified by the thermographer against the "
                 "patterns of Annex C."),
    },
    "bridge_snbi": {
        "inspection_type": "routine inspection (B.IE.01), UAV-assisted",
        "text": ("UAV-assisted visual inspection: nadir and oblique imagery of the deck, "
                 "superstructure and substructure flown at a planned stand-off, with "
                 "radiometric thermal imagery where delamination is a concern. Images "
                 "reviewed by the inspector; automated crack and corrosion detections are "
                 "proposals that the inspector accepts or rejects before they become "
                 "findings. Element condition states assigned by the inspector per the "
                 "AASHTO Manual for Bridge Element Inspection."),
    },
    "roof": {
        "inspection_type": "visual roof inspection from a UAV (InterNACHI SOP 3.1 I: "
                           "\"from the ground or the eaves, or with a drone\")",
        "text": ("Roof inspected from a UAV: near-nadir passes over the roof surfaces and "
                 "oblique passes along the eaves and facades. Roof surfaces, gutters, "
                 "downspouts, penetrations and flashing reviewed on the imagery; nothing was "
                 "walked or touched. Roof slopes measured from the photogrammetric DSM where "
                 "one was produced."),
    },
    "telecom_tower": {
        "inspection_type": "condition assessment per ANSI/TIA-222-H s14 (UAV-assisted)",
        "text": ("UAV-assisted condition assessment: orbit and vertical passes of the "
                 "structure at a planned stand-off, close-ups of mounts, antennas, lines, "
                 "lighting and connections, and a nadir pass over the compound and "
                 "foundations. Each Annex J.1 heading (A-K) assessed by the assessor on the "
                 "imagery and at the base; plumb and twist measured where a survey "
                 "instrument was used."),
    },
}


def _procedures_path() -> Path:
    return _settings_dir() / "procedure_templates.json"


def _procedure_key(template_key: str, workflow_id: str = "") -> str:
    return f"{template_key}::{workflow_id}" if workflow_id else template_key


def list_procedures() -> dict[str, Any]:
    saved = _read(_procedures_path(), {"procedures": {}}).get("procedures") or {}
    return {"builtin": BUILTIN_PROCEDURES, "saved": saved}


def get_procedure(template_key: str, workflow_id: str = "") -> dict[str, Any]:
    """The adopted procedure for a standard (and mission template), else the suggestion.

    ``adopted`` says whether a named person saved it; only an adopted procedure fills
    report items.
    """
    saved = list_procedures()["saved"]
    for key in (_procedure_key(template_key, workflow_id), template_key):
        if key in saved:
            entry = saved[key]
            return {**entry, "adopted": True,
                    "source": (f"procedure template '{key}' edited by {entry['edited_by']} "
                               f"{entry['edited_at'][:10]}")}
    builtin = BUILTIN_PROCEDURES.get(template_key)
    if builtin is None:
        return {"adopted": False, "text": "", "inspection_type": "", "source": ""}
    return {**builtin, "adopted": False,
            "source": "OpenDroneKit default procedure (suggested; not adopted by anyone)"}


def save_procedure(template_key: str, text: str, inspection_type: str, edited_by: str,
                   workflow_id: str = "") -> dict[str, Any]:
    edited_by = _who(edited_by, "A procedure template")
    text, inspection_type = str(text or "").strip(), str(inspection_type or "").strip()
    if not text:
        raise ValueError("A procedure template needs its text.")
    payload = _read(_procedures_path(), {"procedures": {}})
    payload.setdefault("procedures", {})[_procedure_key(template_key, workflow_id)] = {
        "template": template_key, "workflow_id": workflow_id, "text": text,
        "inspection_type": inspection_type, "edited_by": edited_by, "edited_at": _now()}
    _write(_procedures_path(), payload)
    return get_procedure(template_key, workflow_id)


# ── Site conditions log ───────────────────────────────────────────────────────

CONDITION_FIELDS = {
    "cloud_okta": (float, 0, 8), "cloud_type": (str, None, None),
    "wind_speed_m_s": (float, 0, 60), "wind_direction_deg": (float, 0, 360),
    "air_temperature_c": (float, -60, 60), "irradiance_poa_w_m2": (float, 0, 1500),
    "notes": (str, None, None),
}
CLOUD_TYPES = ("none", "cumulus", "stratus", "cirrus", "cumulonimbus", "stratocumulus",
               "altocumulus", "altostratus", "nimbostratus", "cirrostratus", "cirrocumulus",
               "fog", "mixed")


def _conditions_path(project_root: str | Path) -> Path:
    return Path(project_root) / "standards" / "site_conditions.json"


def list_site_conditions(project_root: str | Path) -> list[dict[str, Any]]:
    return list(_read(_conditions_path(project_root), {"records": []}).get("records") or [])


def add_site_conditions(project_root: str | Path, record: dict[str, Any], recorded_by: str,
                        *, origin: str = "field entry", source: str = "") -> dict[str, Any]:
    """Add one on-site record. ``observed_at`` (local ISO time) is required."""
    recorded_by = _who(recorded_by, "A site-conditions record")
    unknown = set(record) - set(CONDITION_FIELDS) - {"observed_at", "flight_id"}
    if unknown:
        raise ValueError(f"Unknown site-condition field(s): {', '.join(sorted(unknown))}.")
    try:
        observed = datetime.fromisoformat(str(record.get("observed_at") or "").strip())
    except ValueError:
        raise ValueError("observed_at must be an ISO date-time (e.g. 2024-12-21T12:02).") from None
    clean: dict[str, Any] = {}
    for key, (kind, low, high) in CONDITION_FIELDS.items():
        value = record.get(key)
        if value in (None, ""):
            continue
        if kind is float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number.") from None
            if not low <= value <= high:
                raise ValueError(f"{key} {value:g} is outside {low}-{high}.")
        else:
            value = str(value).strip()
        if key == "cloud_type" and value.lower() not in CLOUD_TYPES:
            raise ValueError(f"cloud_type must be one of {', '.join(CLOUD_TYPES)}.")
        clean[key] = value
    if not clean:
        raise ValueError("A site-conditions record needs at least one observation.")
    entry = {"id": f"sc_{uuid.uuid4().hex[:10]}", "observed_at": observed.isoformat(sep=" "),
             "flight_id": str(record.get("flight_id") or ""), **clean,
             "recorded_by": recorded_by, "recorded_at": _now(), "origin": origin,
             "source": source or f"site conditions log, {origin} by {recorded_by}"}
    records = list_site_conditions(project_root) + [entry]
    _write(_conditions_path(project_root), {"records": records})
    return entry


def import_site_conditions_from_meteo(project_root: str | Path, csv_source: str,
                                      observed_at: datetime, columns: dict[str, str],
                                      imported_by: str, *, flight_id: str = "",
                                      timestamp_column: str = "Timestamp") -> dict[str, Any]:
    """A record from the meteo-CSV row nearest ``observed_at`` (refused beyond 5 min).

    ``columns`` maps a CONDITION_FIELDS name to the exact header prefix in the file.
    """
    from .standards_collectors import environment_at

    unknown = set(columns) - set(CONDITION_FIELDS)
    if unknown:
        raise ValueError(f"Unknown site-condition field(s): {', '.join(sorted(unknown))}.")
    env = environment_at(csv_source, observed_at, columns, timestamp_column=timestamp_column)
    return add_site_conditions(
        project_root, {"observed_at": env["row_time"], "flight_id": flight_id, **env["values"]},
        imported_by, origin="meteo CSV import",
        source=f"{env['label']} row {env['row_time']}, columns "
               f"{', '.join(repr(h) for h in env['headers'].values())}")


def import_site_conditions_from_flight_log(project_root: str | Path, flight_log_json: str | Path,
                                           imported_by: str, *, flight_id: str = "") -> dict[str, Any]:
    """Wind from a flight-log JSON whose samples carry ``wind_speed_m_s`` (median).

    Most drivers do not report wind; then this refuses rather than recording nothing.
    """
    import statistics

    path = Path(flight_log_json)
    samples = (_read(path, {}) or {}).get("samples") or []
    speeds = [float(s["wind_speed_m_s"]) for s in samples if s.get("wind_speed_m_s") is not None]
    dirs = [float(s["wind_direction_deg"]) for s in samples
            if s.get("wind_direction_deg") is not None]
    if not speeds:
        raise ValueError(f"{path.name}: no sample reports wind_speed_m_s; the vehicle's "
                         "telemetry carries no wind estimate.")
    first = samples[0].get("timestamp_utc") or _now()
    record = {"observed_at": str(first).replace("Z", "")[:19], "flight_id": flight_id,
              "wind_speed_m_s": round(statistics.median(speeds), 2)}
    if dirs:
        record["wind_direction_deg"] = round(statistics.median(dirs), 0)
    return add_site_conditions(project_root, record, imported_by, origin="flight telemetry",
                               source=f"median of {len(speeds)} telemetry samples in {path.name}")


def nearest_site_conditions(project_root: str | Path, when: datetime | None, *,
                            max_gap_minutes: float = 90.0) -> dict[str, Any] | None:
    records = list_site_conditions(project_root)
    if not records:
        return None
    if when is None:
        return records[-1] if len(records) == 1 else None
    best, gap = None, None
    for rec in records:
        try:
            delta = abs((datetime.fromisoformat(rec["observed_at"]) - when).total_seconds())
        except ValueError:
            continue
        if gap is None or delta < gap:
            best, gap = rec, delta
    return best if gap is not None and gap <= max_gap_minutes * 60 else None


# ── Inspection details (per project, per template) ────────────────────────────

# Per-inspection keys that are not report items themselves but feed them.
DETAIL_KEYS = {
    "people": "role -> operator profile id (fills personnel items)",
    "client": "client name",
    "contract_scope": "contract / scope text",
    "module_model": "PV module model as listed in the CEC module list",
    "camera_distance_m": "camera-to-target distance when no flight altitude is recorded",
    "workflow_id": "mission template the inspection was flown with",
    "site_name": "site name for the cover",
    "array_outlines": "operator outlines for DSM tilt/azimuth ([{label, polygon, crs}])",
}


def _details_path(project_root: str | Path, template_key: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", template_key):
        raise ValueError(f"Bad template key {template_key!r}.")
    return Path(project_root) / "standards" / "inspection_details" / f"{template_key}.json"


def load_inspection_details(project_root: str | Path, template_key: str) -> dict[str, Any]:
    return _read(_details_path(project_root, template_key),
                 {"details": {}, "fields": {}, "tables": {}, "coa_confirmations": {}})


def save_inspection_details(project_root: str | Path, template_key: str,
                            payload: dict[str, Any]) -> dict[str, Any]:
    _write(_details_path(project_root, template_key), payload)
    return payload
