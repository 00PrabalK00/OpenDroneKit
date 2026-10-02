"""Standards-structured, signed reports for the project the desktop has open.

The facts for a report come from two places, and only these:
* the project's own data: every image in its imported datasets (hashed into the
  manifest; capture time and camera read from EXIF), and the model digests recorded by
  its pipeline runs;
* what an inspector entered for the template, saved in
  ``<project>/standards/<template>.json`` through Api.save_standards_facts. Those values
  are printed with "entered by <who>" as their source.

Everything else is MISSING. Reports are written to ``<project>/reports/standards``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.standards_collectors import IMAGE_SUFFIXES, capture_facts_for, merge
from core.standards_reports import TEMPLATES, get_template, resolve


def _facts_path(session: Any, template_key: str) -> Path:
    get_template(template_key)
    return Path(session.project_root()) / "standards" / f"{template_key}.json"


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
            continue  # clearing a value leaves it MISSING; it is never stored as blank
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


def project_images(session: Any) -> list[Path]:
    """Every image file in the open project's dataset folders."""
    images: list[Path] = []
    for row in session.store.list_datasets(int(session.ensure_active_project()["id"])):
        folder = Path(str(row.get("path", "")))
        if folder.is_dir():
            images += sorted(p for p in folder.iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return images


def project_model_digests(session: Any) -> list[dict[str, Any]]:
    """Model digests recorded by the project's pipeline runs (``*_model_sha256``)."""
    found: dict[str, dict[str, Any]] = {}
    analysis = Path(session.project_root()) / "analysis"
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
    images = project_images(session)
    derived = capture_facts_for(template_key, images) if images else {
        "facts": {"fields": {}, "tables": {}, "observations": [], "evidence": []}, "inputs": []}
    facts = merge(derived["facts"], load_facts(session, template_key))
    return {"facts": facts, "inputs": derived["inputs"], "models": project_model_digests(session)}


def readiness(session: Any, template_key: str) -> dict[str, Any]:
    """What the report would contain now: provided items, MISSING items, checks."""
    collected = collect(session, template_key)
    content = resolve(get_template(template_key), collected["facts"])
    return {"completeness": content["completeness"], "checks": content["checks"],
            "inputs": len(collected["inputs"]), "models": len(collected["models"])}


def generate(session: Any, template_key: str, title: str = "") -> dict[str, Any]:
    from core.standards_reports import generate_standards_report

    collected = collect(session, template_key)
    project = session.ensure_active_project()
    result = generate_standards_report(
        template_key, collected["facts"], Path(session.project_root()) / "reports" / "standards",
        title=title, project=str(project.get("name", "")), inputs=collected["inputs"],
        models=collected["models"])
    session.store.save_report(
        session.project_id(), result["title"], f"standards:{template_key}", result["html_path"],
        metadata={"pdf_path": result["pdf_path"], "report_id": result["id"],
                  "signing_key_fingerprint": result["signing_key_fingerprint"],
                  "missing_items": result["completeness"]["missing_items"]})
    return result


def list_reports(session: Any) -> list[dict[str, Any]]:
    root = Path(session.project_root()) / "reports" / "standards"
    out = []
    for manifest in sorted(root.glob("*/report.json"), reverse=True) if root.is_dir() else []:
        try:
            out.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def templates() -> list[dict[str, Any]]:
    return [t.describe() for t in TEMPLATES.values()]
