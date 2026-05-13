"""Report engine — HTML / PDF inspection reports with Jinja2 (optional) fallback.

Templates are stored next to the package under `core/report_templates/` and
can be overridden by user templates at `<workspace>/report_templates/`.
"""

from __future__ import annotations

import html
import json
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotations import get_report_annotations
from .audit import list_audit_events
from .errors import AppError, ERR_REPORT_NOT_READY
from .events import REPORT_GENERATED, publish_event
from .measurements import list_measurements
from .project import get_manager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data models ───────────────────────────────────────────────────────────────

DEFAULT_SECTIONS = [
    "overview",
    "site_information",
    "mission_map",
    "dataset_summary",
    "key_photos",
    "defect_summary",
    "crack_propagation",
    "reconstruction",
    "measurements",
    "annotations",
    "recommendations",
    "audit_trail",
    "appendix",
]


@dataclass
class ReportConfig:
    project_id: str
    title: str = "Inspection Report"
    report_type: str = "standard"      # standard | engineering | executive | defect_only | dataset_quality | mission_summary
    sections: list[str] = field(default_factory=lambda: list(DEFAULT_SECTIONS))
    include_images: bool = True
    include_measurements: bool = True
    include_defects: bool = True
    include_audit_trail: bool = True
    include_annotations: bool = True
    author: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportResult:
    id: str
    project_id: str
    title: str
    html_path: str
    pdf_path: str | None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportReadiness:
    ok: bool
    missing: list[str]
    warnings: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Readiness check ───────────────────────────────────────────────────────────

def validate_report_readiness(config: ReportConfig) -> ReportReadiness:
    """Check whether required project, mission, dataset and analysis data exists."""
    mgr = get_manager()
    try:
        meta = mgr.load_project(config.project_id)
    except Exception:
        return ReportReadiness(False, ["project"], [], [])
    root = Path(str(meta.get("root_dir", "")))
    missing: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not root.exists():
        missing.append("project_folder")

    datasets_dir = root / "datasets"
    if not datasets_dir.exists() or not any(datasets_dir.iterdir()):
        if config.include_images:
            warnings.append("No dataset imported — report will not include image gallery.")

    analysis_dir = root / "analysis"
    if config.include_defects:
        defects_dir = analysis_dir / "defects"
        if not defects_dir.exists() or not any(defects_dir.iterdir()):
            warnings.append("No defect run found — defect section will be empty.")

    missions_dir = root / "missions"
    if not missions_dir.exists() or not any(missions_dir.glob("*.json")):
        notes.append("No mission saved — site map section will be sparse.")

    return ReportReadiness(ok=not missing, missing=missing, warnings=warnings, notes=notes)


# ── Context builder ───────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_report_context(config: ReportConfig) -> dict[str, Any]:
    """Collect project metadata, mission metrics, dataset summaries, defects, measurements, annotations, images."""
    mgr = get_manager()
    meta = mgr.load_project(config.project_id)
    root = Path(str(meta.get("root_dir", "")))

    # Missions
    missions: list[dict[str, Any]] = []
    missions_dir = root / "missions"
    if missions_dir.exists():
        for f in sorted(missions_dir.glob("*.json")):
            try:
                missions.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    # Datasets
    datasets: list[dict[str, Any]] = []
    datasets_dir = root / "datasets"
    if datasets_dir.exists():
        for sub in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
            meta_path = sub / "metadata.json"
            if meta_path.exists():
                d = _load_json(meta_path)
                if isinstance(d, dict):
                    datasets.append(d)

    # Defect runs
    defect_runs: list[dict[str, Any]] = []
    defects_dir = root / "analysis" / "defects"
    if defects_dir.exists():
        for sub in sorted(defects_dir.iterdir()):
            summary = sub / "defects.json"
            if summary.exists():
                data = _load_json(summary)
                if isinstance(data, dict):
                    defect_runs.append(data)

    # Crack propagation
    crack_runs: list[dict[str, Any]] = []
    crack_dir = root / "analysis" / "crack_growth"
    if crack_dir.exists():
        for sub in sorted(crack_dir.iterdir()):
            summary = sub / "crack_propagation.json"
            if summary.exists():
                data = _load_json(summary)
                if isinstance(data, dict):
                    crack_runs.append(data)

    # Reconstruction
    reconstructions: list[dict[str, Any]] = []
    recon_dir = root / "analysis" / "reconstruction"
    if recon_dir.exists():
        for sub in sorted(recon_dir.iterdir()):
            summary = sub / "reconstruction_summary.json"
            if summary.exists():
                data = _load_json(summary)
                if isinstance(data, dict):
                    reconstructions.append(data)

    measurements = [m.to_dict() for m in list_measurements(root, config.project_id)] if config.include_measurements else []
    annotations = [a.to_dict() for a in get_report_annotations(root, config.project_id)] if config.include_annotations else []
    audit = list_audit_events(config.project_id, limit=200) if config.include_audit_trail else []

    return {
        "config": config.to_dict(),
        "project": meta,
        "missions": missions,
        "datasets": datasets,
        "defect_runs": defect_runs,
        "crack_runs": crack_runs,
        "reconstructions": reconstructions,
        "measurements": measurements,
        "annotations": annotations,
        "audit": audit,
        "generated_at": _now_iso(),
        "section_count": len(config.sections or DEFAULT_SECTIONS),
    }


# ── Templates ─────────────────────────────────────────────────────────────────

_BUILTIN_TEMPLATE_DIR = Path(__file__).resolve().parent / "report_templates"


_DEFAULT_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
:root {{
    --bg: #0b1220;
    --surface: #121c2e;
    --border: #2b3d5c;
    --text: #e8eefc;
    --muted: #95a3b8;
    --primary: #3b82f6;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0; padding: 0;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 24px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
h2 {{ font-size: 20px; margin: 32px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
h3 {{ font-size: 16px; margin: 20px 0 8px; color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 13px; }}
th {{ background: rgba(59, 130, 246, 0.1); color: var(--text); }}
tr:last-child td {{ border-bottom: 0; }}
.chip {{ display: inline-block; padding: 2px 10px; border-radius: 12px; background: rgba(59,130,246,0.15); color: var(--text); font-size: 12px; margin-right: 6px; }}
.chip.warn {{ background: rgba(245,158,11,0.18); color: var(--warning); }}
.chip.danger {{ background: rgba(239,68,68,0.18); color: var(--danger); }}
.chip.ok {{ background: rgba(34,197,94,0.18); color: var(--success); }}
.muted {{ color: var(--muted); }}
.kv {{ display: grid; grid-template-columns: 200px 1fr; gap: 4px 12px; margin: 8px 0; }}
.kv dt {{ color: var(--muted); }}
.section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; margin: 18px 0; }}
.footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; background: rgba(0,0,0,0.25); padding: 10px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
    <header>
        <div class="muted">OpenDroneKit Inspection Report — {report_type}</div>
        <h1>{title}</h1>
        <div class="muted">Generated {generated_at}</div>
    </header>

    {sections_html}

    <div class="footer">
        Report ID: {report_id} · Project ID: {project_id} · Sections: {section_count}
    </div>
</div>
</body>
</html>
"""


def _builtin_template_path(name: str) -> Path:
    return _BUILTIN_TEMPLATE_DIR / name


def _ensure_builtin_template() -> Path:
    """Drop the default template to disk so reports can be customised."""
    _BUILTIN_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _builtin_template_path("standard.html")
    if not path.exists():
        path.write_text(_DEFAULT_HTML_TEMPLATE, encoding="utf-8")
    return path


# ── Renderers ─────────────────────────────────────────────────────────────────

def _render_section(section: str, context: dict[str, Any]) -> str:
    if section == "overview":
        proj = context.get("project", {})
        return f"""
<section class="section">
<h2>Overview</h2>
<dl class="kv">
<dt>Project</dt><dd>{html.escape(str(proj.get('name', '')))}</dd>
<dt>Description</dt><dd>{html.escape(str(proj.get('description', '')))}</dd>
<dt>Workflow</dt><dd>{html.escape(str(proj.get('workflow_id', '')))}</dd>
<dt>Status</dt><dd><span class="chip ok">{html.escape(str(proj.get('sync_status', 'local')))}</span></dd>
</dl>
</section>"""
    if section == "site_information":
        proj = context.get("project", {})
        return f"""
<section class="section">
<h2>Site Information</h2>
<dl class="kv">
<dt>Project ID</dt><dd>{html.escape(str(proj.get('id', '')))}</dd>
<dt>Root folder</dt><dd><code>{html.escape(str(proj.get('root_dir', '')))}</code></dd>
<dt>Created</dt><dd>{html.escape(str(proj.get('created_at', '')))}</dd>
<dt>Updated</dt><dd>{html.escape(str(proj.get('updated_at', '')))}</dd>
</dl>
</section>"""
    if section == "mission_map":
        missions = context.get("missions", []) or []
        if not missions:
            return "<section class='section'><h2>Mission</h2><p class='muted'>No missions saved.</p></section>"
        rows = []
        for m in missions:
            rows.append(f"<tr><td>{html.escape(str(m.get('name', '')))}</td>"
                        f"<td>{html.escape(str(m.get('mode', '')))}</td>"
                        f"<td>{html.escape(str(m.get('waypoint_count', len(m.get('waypoints', [])))))}</td>"
                        f"<td>{html.escape(str(m.get('altitude_m', '')))}</td></tr>")
        return f"""
<section class="section">
<h2>Missions</h2>
<table><thead><tr><th>Name</th><th>Mode</th><th>Waypoints</th><th>Altitude (m)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</section>"""
    if section == "dataset_summary":
        datasets = context.get("datasets", []) or []
        if not datasets:
            return "<section class='section'><h2>Datasets</h2><p class='muted'>No datasets imported.</p></section>"
        rows = []
        for d in datasets:
            rows.append(f"<tr><td>{html.escape(str(d.get('name', '')))}</td>"
                        f"<td>{html.escape(str(d.get('dataset_type', '')))}</td>"
                        f"<td>{int(d.get('image_count', 0) or 0)}</td>"
                        f"<td>{'yes' if d.get('has_gps_metadata') else 'no'}</td>"
                        f"<td>{html.escape(str(d.get('qa_status', 'unchecked')))}</td></tr>")
        return f"""
<section class="section">
<h2>Dataset Summary</h2>
<table><thead><tr><th>Name</th><th>Type</th><th>Images</th><th>GPS</th><th>QA</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</section>"""
    if section == "defect_summary":
        runs = context.get("defect_runs", []) or []
        if not runs:
            return "<section class='section'><h2>Defect Summary</h2><p class='muted'>No defect runs.</p></section>"
        rows = []
        for r in runs:
            for d in r.get("defects", []):
                rows.append(
                    f"<tr><td>{html.escape(Path(str(d.get('image_path', ''))).name)}</td>"
                    f"<td>{html.escape(str(d.get('defect_type', '')))}</td>"
                    f"<td>{float(d.get('confidence', 0) or 0):.2f}</td>"
                    f"<td><span class='chip {'danger' if d.get('severity') in ('critical', 'high') else 'warn' if d.get('severity') == 'medium' else 'ok'}'>{html.escape(str(d.get('severity', '')))}</span></td>"
                    f"<td>{int(d.get('area_px', 0) or 0)}</td></tr>"
                )
        body = "".join(rows[:200]) or "<tr><td colspan='5' class='muted'>No defects detected.</td></tr>"
        return f"""
<section class="section">
<h2>Defect Summary</h2>
<table><thead><tr><th>Image</th><th>Type</th><th>Confidence</th><th>Severity</th><th>Area (px)</th></tr></thead>
<tbody>{body}</tbody></table>
</section>"""
    if section == "crack_propagation":
        runs = context.get("crack_runs", []) or []
        if not runs:
            return "<section class='section'><h2>Crack Propagation</h2><p class='muted'>No crack runs.</p></section>"
        blocks = []
        for r in runs:
            blocks.append(
                f"<div class='section'><h3>Run {html.escape(str(r.get('id', '')))}</h3>"
                f"<p>{html.escape(str(r.get('summary', '')))}</p>"
                f"<p class='muted'>Risk level: <span class='chip {'danger' if r.get('risk_level') in ('Critical', 'High') else 'warn' if r.get('risk_level') == 'Medium' else 'ok'}'>{html.escape(str(r.get('risk_level', '')))}</span></p>"
                f"<h4>Assumptions</h4><ul>{''.join(f'<li>{html.escape(str(a))}</li>' for a in r.get('assumptions', []))}</ul></div>"
            )
        return f"<section class='section'><h2>Crack Propagation</h2>{''.join(blocks)}</section>"
    if section == "reconstruction":
        recons = context.get("reconstructions", []) or []
        if not recons:
            return "<section class='section'><h2>3D Reconstruction</h2><p class='muted'>No reconstruction output.</p></section>"
        blocks = []
        for r in recons:
            qm = r.get("quality_metrics", {}) or {}
            blocks.append(
                f"<div class='section'><h3>Reconstruction {html.escape(str(r.get('id', '')))}</h3>"
                f"<dl class='kv'>"
                f"<dt>Frames</dt><dd>{int(qm.get('frame_count', 0) or 0)}</dd>"
                f"<dt>Total points</dt><dd>{int(qm.get('total_points', 0) or 0)}</dd>"
                f"<dt>Profile</dt><dd>{html.escape(str(qm.get('processing_profile', '')))}</dd>"
                f"<dt>Execution</dt><dd>{html.escape(str(qm.get('execution_mode_used', '')))}</dd>"
                f"</dl></div>"
            )
        return f"<section class='section'><h2>3D Reconstruction</h2>{''.join(blocks)}</section>"
    if section == "measurements":
        items = context.get("measurements", []) or []
        if not items:
            return "<section class='section'><h2>Measurements</h2><p class='muted'>No measurements.</p></section>"
        rows = []
        for m in items:
            rows.append(
                f"<tr><td>{html.escape(str(m.get('label', '')))}</td>"
                f"<td>{html.escape(str(m.get('measurement_type', '')))}</td>"
                f"<td>{float(m.get('value', 0) or 0):.4f}</td>"
                f"<td>{html.escape(str(m.get('unit', '')))}</td></tr>"
            )
        return f"""
<section class="section">
<h2>Measurements</h2>
<table><thead><tr><th>Label</th><th>Type</th><th>Value</th><th>Unit</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</section>"""
    if section == "annotations":
        items = context.get("annotations", []) or []
        if not items:
            return "<section class='section'><h2>Annotations</h2><p class='muted'>No annotations.</p></section>"
        rows = []
        for a in items:
            rows.append(
                f"<tr><td>{html.escape(str(a.get('label', '')))}</td>"
                f"<td>{html.escape(str(a.get('annotation_type', '')))}</td>"
                f"<td>{html.escape(str(a.get('severity', '')))}</td>"
                f"<td>{html.escape(str(a.get('note', '') or ''))}</td></tr>"
            )
        return f"""
<section class="section">
<h2>Annotations</h2>
<table><thead><tr><th>Label</th><th>Type</th><th>Severity</th><th>Note</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</section>"""
    if section == "audit_trail":
        events = context.get("audit", []) or []
        if not events:
            return "<section class='section'><h2>Audit Trail</h2><p class='muted'>No audit events.</p></section>"
        rows = []
        for e in events[:50]:
            rows.append(
                f"<tr><td>{html.escape(str(e.get('timestamp', '')))}</td>"
                f"<td>{html.escape(str(e.get('event_type', '')))}</td>"
                f"<td>{html.escape(str(e.get('detail', '')))}</td></tr>"
            )
        return f"""
<section class="section">
<h2>Audit Trail</h2>
<table><thead><tr><th>Time</th><th>Event</th><th>Detail</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</section>"""
    if section == "key_photos":
        datasets = context.get("datasets", []) or []
        if not datasets:
            return ""
        return f"""
<section class="section">
<h2>Key Photos</h2>
<p class="muted">Photos are stored in dataset folders; reference: {len(datasets)} dataset(s).</p>
</section>"""
    if section == "recommendations":
        return """
<section class="section">
<h2>Recommendations</h2>
<ul>
<li>Re-fly any areas marked with low coverage.</li>
<li>Inspect cracks classified as <b>Critical</b> or <b>High</b> within 30 days.</li>
<li>Configure missing AI model paths in Developer Tools to expand coverage.</li>
</ul>
</section>"""
    if section == "appendix":
        return f"""
<section class="section">
<h2>Appendix</h2>
<pre>{html.escape(json.dumps({k: v for k, v in context.items() if k != 'audit'}, indent=2)[:6000])}</pre>
</section>"""
    return ""


def render_report_html(context: dict[str, Any], template_path: Path | None, output_path: Path | str) -> Path:
    """Render report HTML. Uses Jinja2 if available; falls back to built-in template."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template_path = template_path or _ensure_builtin_template()

    cfg = context.get("config", {})
    sections = cfg.get("sections") or DEFAULT_SECTIONS
    sections_html = "\n".join(_render_section(s, context) for s in sections)

    title = cfg.get("title", "Inspection Report")
    report_type = cfg.get("report_type", "standard")
    generated_at = context.get("generated_at", _now_iso())
    project_id = context.get("project", {}).get("id", "")
    report_id = context.get("report_id", "")
    section_count = len(sections)

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        env = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            autoescape=select_autoescape(["html", "htm"]),
        )
        tmpl = env.get_template(template_path.name)
        html_str = tmpl.render(
            title=title,
            report_type=report_type,
            generated_at=generated_at,
            sections_html=sections_html,
            report_id=report_id,
            project_id=project_id,
            section_count=section_count,
            **context,
        )
    except Exception:
        tmpl_text = template_path.read_text(encoding="utf-8")
        html_str = tmpl_text.format(
            title=html.escape(title),
            report_type=html.escape(report_type),
            generated_at=html.escape(generated_at),
            sections_html=sections_html,
            report_id=html.escape(report_id),
            project_id=html.escape(project_id),
            section_count=section_count,
        )

    out.write_text(html_str, encoding="utf-8")
    return out


def export_report_pdf(html_path: Path | str, output_path: Path | str) -> Path | None:
    """Convert HTML to PDF. Returns path or None if no renderer is available."""
    html_path = Path(html_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(out))
        return out
    except Exception:
        pass
    # Fall back to pdfkit (wkhtmltopdf) if installed
    try:
        import pdfkit
        pdfkit.from_file(str(html_path), str(out))
        return out
    except Exception:
        return None


def generate_report(config: ReportConfig) -> ReportResult:
    """End-to-end: validate readiness → build context → render HTML → export PDF."""
    readiness = validate_report_readiness(config)
    if not readiness.ok:
        raise AppError(
            ERR_REPORT_NOT_READY,
            "Report cannot be generated — required data missing.",
            technical_message=f"Missing: {readiness.missing}",
            recovery_action="Fix missing items in readiness checklist.",
        )

    mgr = get_manager()
    meta = mgr.load_project(config.project_id)
    root = Path(str(meta.get("root_dir", "")))
    report_id = str(uuid.uuid4())
    out_dir = root / "reports" / report_id
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    context = build_report_context(config)
    context["report_id"] = report_id

    html_path = render_report_html(
        context=context,
        template_path=_ensure_builtin_template(),
        output_path=out_dir / "report.html",
    )
    pdf_path = export_report_pdf(html_path, out_dir / "report.pdf")

    # Persist a manifest for list_reports
    result = ReportResult(
        id=report_id,
        project_id=config.project_id,
        title=config.title,
        html_path=str(html_path),
        pdf_path=str(pdf_path) if pdf_path else None,
    )
    (out_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # Audit log
    try:
        mgr.write_project_audit(config.project_id, "report_generated", f"Generated report: {config.title}",
                                {"report_id": report_id, "html_path": str(html_path)})
    except Exception:
        pass
    publish_event(REPORT_GENERATED, {"report_id": report_id, "project_id": config.project_id})
    return result


def list_reports(project_id: str) -> list[ReportResult]:
    """Return generated reports for a project."""
    mgr = get_manager()
    try:
        meta = mgr.load_project(project_id)
    except Exception:
        return []
    root = Path(str(meta.get("root_dir", "")))
    reports_dir = root / "reports"
    if not reports_dir.exists():
        return []
    out: list[ReportResult] = []
    for sub in sorted(reports_dir.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        manifest = sub / "report.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            allowed = {k: v for k, v in data.items() if k in ReportResult.__dataclass_fields__}
            out.append(ReportResult(**allowed))
        except Exception:
            pass
    return out
