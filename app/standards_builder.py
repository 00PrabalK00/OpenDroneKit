"""Standards-structured, signed reports for the project the desktop has open.

The facts for a report come from these places, and only these:
* the project's own data: every image in its imported datasets (hashed into the
  manifest; capture time, camera, position read from EXIF/XMP), a DSM/DTM produced by its
  reconstruction runs (roof facets), the offline gazetteer ``places.json`` for a place
  name, and the model digests recorded by its pipeline runs;
* the per-project **inspection details** (``<project>/standards/inspection_details/<t>.json``):
  which saved operator profiles fill the personnel items, client, contract scope, module
  model (looked up in the CEC module list), camera distance, operator-entered items and
  CoA confirmations from review -- each value sourced to the person who entered it;
* the installation's **procedure templates** (adopted by a named person) and the
  project's **site conditions log** (Field mode / meteo import);
* values saved through the older Api.save_standards_facts
  (``<project>/standards/<template>.json``), kept for compatibility.

Anything else is open and listed on the report's checklist. Reports are written to
``<project>/reports/standards``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import inspection_records as records
from core.standards_collectors import IMAGE_SUFFIXES, capture_facts_for, merge
from core.standards_reports import (TEMPLATES, WHO_LABELS, get_template, item_hint, item_who,
                                    resolve)

# Which procedure-template field fills which item.
PROCEDURE_ITEMS = {
    "iec62446_3": {"inspection.procedure": "text", "inspection.type": "inspection_type"},
    "bridge_snbi": {"ie.type": "inspection_type"},
}
# Site-conditions log fields per template item.
SITE_ITEMS = {
    "env.cloud_okta": "cloud_okta", "env.cloud_type": "cloud_type",
    "env.wind_speed_m_s": "wind_speed_m_s", "env.wind_direction_deg": "wind_direction_deg",
    "env.air_temperature_c": "air_temperature_c", "env.irradiance_poa_w_m2": "irradiance_poa_w_m2",
}
WEATHER_ITEMS = ("general.weather", "site.weather")


def _root(session: Any) -> Path:
    return Path(session.project_root())


def _facts_path(session: Any, template_key: str) -> Path:
    get_template(template_key)
    return _root(session) / "standards" / f"{template_key}.json"


def load_facts(session: Any, template_key: str) -> dict[str, Any]:
    path = _facts_path(session, template_key)
    if not path.is_file():
        return {"fields": {}, "tables": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_facts(session: Any, template_key: str, facts: dict[str, Any],
               entered_by: str) -> dict[str, Any]:
    """Store inspector-entered values. Unknown item ids and empty values are refused."""
    template = get_template(template_key)
    entered_by = str(entered_by or "").strip()
    if not entered_by:
        raise ValueError("Entered values must name who entered them (entered_by).")
    known_items = {item.id for item in template.items()}
    known_tables = {table.id: {c.key for c in table.columns} for table in template.tables()}
    fields_in = dict((facts or {}).get("fields") or {})
    unknown = sorted(set(fields_in) - known_items)
    if unknown:
        raise ValueError(f"Unknown item(s) for {template_key}: {', '.join(unknown)}.")
    clean_fields = {}
    for item_id, value in fields_in.items():
        raw = value.get("value") if isinstance(value, dict) else value
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue  # clearing a value leaves it open; it is never stored as blank
        clean_fields[item_id] = {"value": raw, "source": f"entered by {entered_by}"}
    clean_tables: dict[str, list[dict[str, Any]]] = {}
    for table_id, rows in dict((facts or {}).get("tables") or {}).items():
        if table_id not in known_tables:
            raise ValueError(f"Unknown table {table_id!r} for {template_key}.")
        out_rows = []
        for row in rows or []:
            extra = set(row) - known_tables[table_id] - {"source"}
            if extra:
                raise ValueError(f"Unknown column(s) {sorted(extra)} in {table_id}.")
            out_rows.append({**{k: v for k, v in row.items() if k != "source"},
                             "source": f"entered by {entered_by}"})
        clean_tables[table_id] = out_rows
    stored = {"fields": clean_fields, "tables": clean_tables}
    path = _facts_path(session, template_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2, ensure_ascii=False), encoding="utf-8")
    return stored


# ── Inspection details ────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def inspection_form(session: Any, template_key: str) -> dict[str, Any]:
    """The template-driven form: every required item, who supplies it, its current value.

    A UI renders one control per entry: a profile picker for ``who == "profile"``, the
    procedure editor link for ``procedure``, the module search for ``lookup``, the site
    log for ``site``, a text box for ``operator``/``judgement``, and a read-only value with
    its source (overridable) for ``data``.
    """
    template = get_template(template_key)
    collected = collect(session, template_key)
    content = resolve(template, collected["facts"])
    current = {i["id"]: i for s in content["sections"] for i in s["items"]}
    fields = []
    for item in template.items():
        now = current[item.id]
        fields.append({"id": item.id, "label": item.label, "ref": item.ref, "unit": item.unit,
                       "who": item_who(item), "who_label": WHO_LABELS[item_who(item)],
                       "hint": item_hint(item), "status": now["status"],
                       "value": now["value"], "source": now["source"],
                       "origin": now.get("origin"), "suggestion": now.get("suggestion")})
    tables = [{"id": t.id, "title": t.title, "ref": t.ref, "note": t.note,
               "columns": [{"key": c.key, "label": c.label, "required": c.required}
                           for c in t.columns]} for t in template.tables()]
    return {"fields": fields, "tables": tables,
            "todo": content["todo"], "completeness": content["completeness"]}


def get_inspection_details(session: Any, template_key: str) -> dict[str, Any]:
    get_template(template_key)
    details = records.load_inspection_details(_root(session), template_key)
    procedure = records.get_procedure(template_key, str((details.get("details") or {})
                                                        .get("workflow_id") or ""))
    return {"template": template_key, "stored": details,
            "form": inspection_form(session, template_key),
            "profiles": records.list_operator_profiles(), "procedure": procedure,
            "site_conditions": records.list_site_conditions(_root(session)),
            "detail_keys": records.DETAIL_KEYS}


def set_inspection_details(session: Any, template_key: str, payload: dict[str, Any],
                           entered_by: str) -> dict[str, Any]:
    """Merge per-inspection details. Every stored value records who entered it and when.

    ``payload``: ``{"details": {DETAIL_KEYS...}, "fields": {item_id: value},
    "tables": {table_id: [rows]}}``. Clearing a field (empty value) removes it.
    """
    from core import pv_modules

    template = get_template(template_key)
    entered_by = str(entered_by or "").strip()
    if not entered_by:
        raise ValueError("Inspection details must name who entered them (entered_by).")
    unknown_top = set(payload or {}) - {"details", "fields", "tables"}
    if unknown_top:
        raise ValueError(f"Unknown section(s): {', '.join(sorted(unknown_top))}.")
    stored = records.load_inspection_details(_root(session), template_key)
    stamp = {"entered_by": entered_by, "entered_at": _now()}
    details_in = dict((payload or {}).get("details") or {})
    unknown = set(details_in) - set(records.DETAIL_KEYS)
    if unknown:
        raise ValueError(f"Unknown detail(s): {', '.join(sorted(unknown))}.")
    profile_items = {i.id for i in template.items() if item_who(i) == "profile"}
    if "people" in details_in:
        people = details_in["people"] or {}
        if not isinstance(people, dict):
            raise ValueError("people maps a personnel item id to a list of profile ids.")
        bad = set(people) - profile_items
        if bad:
            raise ValueError(f"Not a personnel item of {template_key}: {', '.join(sorted(bad))}.")
        for ids in people.values():
            for pid in ([ids] if isinstance(ids, str) else ids):
                records.get_operator_profile(pid)  # raises for an unknown id
    if details_in.get("module_model"):
        pv_modules.lookup(details_in["module_model"])  # raises UnknownModule (a KeyError)
    if "camera_distance_m" in details_in and details_in["camera_distance_m"] not in (None, ""):
        distance = float(details_in["camera_distance_m"])
        if not 0.5 <= distance <= 500:
            raise ValueError("camera_distance_m must be 0.5-500 m.")
        details_in["camera_distance_m"] = distance
    for key, value in details_in.items():
        if value in (None, "", [], {}):
            stored["details"].pop(key, None)
        else:
            stored["details"][key] = {"value": value, **stamp}
    known_items = {item.id for item in template.items()}
    fields_in = dict((payload or {}).get("fields") or {})
    bad = set(fields_in) - known_items
    if bad:
        raise ValueError(f"Unknown item(s) for {template_key}: {', '.join(sorted(bad))}.")
    for item_id, value in fields_in.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            stored["fields"].pop(item_id, None)
        else:
            stored["fields"][item_id] = {"value": value, **stamp}
    known_tables = {t.id: {c.key for c in t.columns} for t in template.tables()}
    for table_id, rows in dict((payload or {}).get("tables") or {}).items():
        if table_id not in known_tables:
            raise ValueError(f"Unknown table {table_id!r} for {template_key}.")
        for row in rows or []:
            extra = set(row) - known_tables[table_id]
            if extra:
                raise ValueError(f"Unknown column(s) {sorted(extra)} in {table_id}.")
        stored.setdefault("tables", {})[table_id] = {"rows": list(rows or []), **stamp}
    records.save_inspection_details(_root(session), template_key, stored)
    return get_inspection_details(session, template_key)


def confirm_anomaly_class(session: Any, template_key: str, key: str, coa: int,
                          confirmed_by: str, abnormality_type: str = "",
                          action: str = "") -> dict[str, Any]:
    """Record a person's CoA decision for one anomaly row (the suggestion becomes theirs)."""
    if template_key != "iec62446_3":
        raise ValueError("Classes of abnormality exist only in the IEC TS 62446-3 template.")
    confirmed_by = str(confirmed_by or "").strip()
    if not confirmed_by:
        raise ValueError("A CoA confirmation must name who confirmed it.")
    if int(coa) not in (1, 2, 3):
        raise ValueError("CoA is 1, 2 or 3 (IEC TS 62446-3 7.3.2).")
    stored = records.load_inspection_details(_root(session), template_key)
    stored.setdefault("coa_confirmations", {})[str(key)] = {
        "coa": int(coa), "abnormality_type": str(abnormality_type or "").strip(),
        "action": str(action or "").strip(), "confirmed_by": confirmed_by,
        "confirmed_at": _now()}
    records.save_inspection_details(_root(session), template_key, stored)
    return stored["coa_confirmations"][str(key)]


def _apply_details(template_key: str, facts: dict[str, Any], stored: dict[str, Any],
                   project_root: Path) -> None:
    from core import pv_modules

    template = get_template(template_key)
    fields = facts.setdefault("fields", {})
    det = {k: v["value"] for k, v in (stored.get("details") or {}).items()}
    meta = stored.get("details") or {}
    cover = facts.setdefault("details", {})
    for key in ("client", "site_name"):
        if det.get(key):
            cover["client" if key == "client" else "site"] = det[key]
    # People from saved operator profiles.
    for item_id, ids in (det.get("people") or {}).items():
        ids = [ids] if isinstance(ids, str) else list(ids)
        people = []
        for pid in ids:
            try:
                people.append(records.get_operator_profile(pid))
            except ValueError:
                continue
        if people:
            fields[item_id] = {"value": "; ".join(records.describe_person(p) for p in people),
                               "source": "; ".join(records.profile_source(p) for p in people)
                               + f" — picked by {meta['people']['entered_by']}",
                               "origin": "profile"}
    # Procedure template: adopted -> reused; default -> suggestion only.
    proc = records.get_procedure(template_key, str(det.get("workflow_id") or ""))
    cover["procedure"] = proc
    for item_id, key in PROCEDURE_ITEMS.get(template_key, {}).items():
        if proc.get(key) and item_id not in fields:
            fields[item_id] = {"value": proc[key], "source": proc["source"],
                               "origin": "procedure" if proc["adopted"] else "suggested"}
    # Module lookup.
    if det.get("module_model") and template_key == "iec62446_3":
        try:
            module = pv_modules.lookup(det["module_model"])
        except KeyError:
            module = None
        if module:
            who = meta["module_model"]["entered_by"]
            fields["scope.component_types"] = {
                "value": pv_modules.describe(module),
                "source": f"{module['source']}; model picked by {who}", "origin": "lookup"}
            if module["efficiency_pct"]:
                fields["scope.module_efficiency"] = {
                    "value": f"{module['efficiency_pct']:.2f} % ({module['stc_w']:g} W STC)",
                    "source": f"{module['efficiency_source']}; model picked by {who}",
                    "origin": "lookup"}
    # Camera distance -> geometric resolution (when the files gave no distance).
    if det.get("camera_distance_m") and "camera.geometric_resolution_cm_px" not in fields:
        from core.standards_collectors import geometric_resolution_cm
        from mission.cameras import all_profiles

        profile = all_profiles().get(str(cover.get("camera_profile") or ""))
        if profile is not None:
            d = float(det["camera_distance_m"])
            fields["camera.geometric_resolution_cm_px"] = {
                "value": round(geometric_resolution_cm(d, profile.pixel_pitch_um,
                                                       profile.focal_mm), 2),
                "source": (f"computed: {d:g} m (entered by "
                           f"{meta['camera_distance_m']['entered_by']}) × "
                           f"{profile.pixel_pitch_um:.1f} µm / {profile.focal_mm:g} mm, camera "
                           f"registry profile '{profile.key}'"),
                "origin": "operator"}
    # Site conditions log nearest the inspection.
    when = None
    for item_id in ("inspection.datetime", "general.datetime", "site.datetime", "ie.begin_date"):
        if item_id in fields:
            try:
                when = datetime.fromisoformat(str(fields[item_id]["value"])[:19])
            except ValueError:
                pass
            break
    log = records.nearest_site_conditions(project_root, when)
    if log:
        known = {i.id for i in template.items()}
        for item_id, key in SITE_ITEMS.items():
            if item_id in known and item_id not in fields and log.get(key) not in (None, ""):
                fields[item_id] = {"value": log[key], "source": f"{log['source']} "
                                   f"({log['observed_at']})", "origin": "site"}
        weather = ", ".join(f"{label} {log[k]}" for k, label in (
            ("cloud_okta", "cloud okta"), ("cloud_type", "cloud"), ("wind_speed_m_s", "wind m/s"),
            ("air_temperature_c", "air °C"), ("notes", "notes")) if log.get(k) not in (None, ""))
        for item_id in WEATHER_ITEMS:
            if item_id in known and item_id not in fields and weather:
                fields[item_id] = {"value": weather, "source": f"{log['source']} "
                                   f"({log['observed_at']})", "origin": "site"}
    # Operator-entered items.
    for item_id, entry in (stored.get("fields") or {}).items():
        fields[item_id] = {"value": entry["value"],
                           "source": f"entered by {entry['entered_by']} {entry['entered_at'][:10]}",
                           "origin": "operator"}
    for table_id, entry in (stored.get("tables") or {}).items():
        facts.setdefault("tables", {})[table_id] = [
            {**row, "source": f"entered by {entry['entered_by']} {entry['entered_at'][:10]}"}
            for row in entry["rows"]]
    # CoA confirmations from review.
    confirmations = stored.get("coa_confirmations") or {}
    for row in (facts.get("tables") or {}).get("anomalies") or []:
        done = confirmations.get(str(row.get("key")))
        if not done:
            continue
        row["coa"] = f"CoA {done['coa']}"
        if done.get("abnormality_type"):
            row["abnormality_type"] = done["abnormality_type"]
        if done.get("action"):
            row["action"] = done["action"]
        row["source"] = (f"{row.get('source', '')}; CoA confirmed by {done['confirmed_by']} "
                         f"{done['confirmed_at']}")
        row.pop("suggestions", None)
    for finding in facts.get("findings") or []:
        done = confirmations.get(str(finding.get("key")))
        if done:
            finding["severity"] = {"label": f"CoA {done['coa']}", "level": done["coa"],
                                   "confirmed": True}
            finding["note"] = (f"Confirmed by {done['confirmed_by']} {done['confirmed_at'][:10]}"
                               + (f": {done['action']}" if done.get("action") else "."))
    if confirmations and facts.get("findings"):
        counts: dict[str, int] = {}
        for f in facts["findings"]:
            counts[f["severity"]["label"]] = counts.get(f["severity"]["label"], 0) + 1
        facts.setdefault("summary", {})["severity_counts"] = counts


# ── Project data ──────────────────────────────────────────────────────────────

def project_images(session: Any) -> list[Path]:
    """Every image file in the open project's dataset folders."""
    images: list[Path] = []
    for row in session.store.list_datasets(int(session.ensure_active_project()["id"])):
        folder = Path(str(row.get("path", "")))
        if folder.is_dir():
            images += sorted(p for p in folder.iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return images


def project_surface_models(session: Any) -> tuple[Path | None, Path | None]:
    """The newest DSM with a DTM beside it under the project folder, if any."""
    root = _root(session)
    pairs = [(p, p.with_name("dtm.tif")) for p in root.rglob("dsm.tif")
             if p.with_name("dtm.tif").is_file()]
    if not pairs:
        return None, None
    return max(pairs, key=lambda pair: pair[0].stat().st_mtime)


def project_model_digests(session: Any) -> list[dict[str, Any]]:
    """Model digests recorded by the project's pipeline runs (``*_model_sha256``)."""
    found: dict[str, dict[str, Any]] = {}
    analysis = _root(session) / "analysis"
    for summary_path in sorted(analysis.glob("run_*/summary.json")) if analysis.is_dir() else []:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in summary.get("images") or []:
            for key, digest in row.items():
                if not key.endswith("_model_sha256") or not digest:
                    continue
                used = str(row.get(key.replace("_sha256", "_used")) or key)
                found.setdefault(str(digest), {
                    "name": used, "sha256": str(digest),
                    "digest_source": f"recorded in {summary_path.parent.name}/summary.json"})
    return list(found.values())


def collect(session: Any, template_key: str) -> dict[str, Any]:
    root = _root(session)
    images = project_images(session)
    gazetteer = root / "places.json"
    dsm, dtm = project_surface_models(session) if template_key == "roof" else (None, None)
    derived = capture_facts_for(template_key, images, gazetteer_path=gazetteer,
                                dsm=dsm, dtm=dtm) if images else {
        "facts": {"fields": {}, "tables": {}}, "inputs": []}
    facts = merge(derived["facts"], load_facts(session, template_key))
    _apply_details(template_key, facts, records.load_inspection_details(root, template_key), root)
    return {"facts": facts, "inputs": derived["inputs"],
            "models": project_model_digests(session) + list(derived.get("models") or [])}


def readiness(session: Any, template_key: str) -> dict[str, Any]:
    """What the report would contain now: supplied items, the open checklist, checks."""
    collected = collect(session, template_key)
    content = resolve(get_template(template_key), collected["facts"])
    return {"completeness": content["completeness"], "checks": content["checks"],
            "todo": content["todo"], "status": content["status"],
            "inputs": len(collected["inputs"]), "models": len(collected["models"])}


def generate(session: Any, template_key: str, title: str = "") -> dict[str, Any]:
    from core.standards_reports import generate_standards_report

    collected = collect(session, template_key)
    project = session.ensure_active_project()
    result = generate_standards_report(
        template_key, collected["facts"], _root(session) / "reports" / "standards",
        title=title, project=str(project.get("name", "")), inputs=collected["inputs"],
        models=collected["models"])
    session.store.save_report(
        session.project_id(), result["title"], f"standards:{template_key}", result["html_path"],
        metadata={"pdf_path": result["pdf_path"], "report_id": result["id"],
                  "signing_key_fingerprint": result["signing_key_fingerprint"],
                  "missing_items": result["completeness"]["missing_items"],
                  "open_actions": result["completeness"]["open_actions"]})
    return result


def list_reports(session: Any) -> list[dict[str, Any]]:
    root = _root(session) / "reports" / "standards"
    out = []
    for manifest in sorted(root.glob("*/report.json"), reverse=True) if root.is_dir() else []:
        try:
            out.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def templates() -> list[dict[str, Any]]:
    out = []
    for t in TEMPLATES.values():
        desc = t.describe()
        whos = {i.id: item_who(i) for i in t.items()}
        for section in desc["sections"]:
            for item in section["items"]:
                item["who"] = whos[item["id"]]
        out.append(desc)
    return out
