"""PDF and DOCX report writers.

The existing report engine produces HTML and Markdown. Engineering deliverables are
usually asked for as PDF (fixed, signable) or DOCX (editable by the client), so those
are generated directly from the report payload rather than by converting HTML, which
would drag in a browser engine and lose the structure.

Findings are ordered by severity, because a report whose first page buries the
critical item behind twenty cosmetic ones is a worse deliverable regardless of how
complete it is. Numbering stays stable so a finding can be referred to in
correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ordered worst first. Anything unrecognised sorts last rather than being dropped.
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# The annotation store records a review as a STATUS (open / in_review / resolved /
# dismissed) while deliverables speak of review states. The two vocabularies met here
# with no translation, so an accepted finding (status "resolved") was still printed as
# "not confirmed by an inspector" and include_unreviewed=False dropped every finding.
REVIEW_STATE_ALIASES = {
    "accepted": "accepted", "resolved": "accepted",
    "rejected": "rejected", "dismissed": "rejected",
    "flagged": "flagged", "in_review": "flagged",
    "unreviewed": "unreviewed", "open": "unreviewed", "": "unreviewed",
}


def review_state(finding: dict[str, Any]) -> str:
    """The finding's review state in the deliverable's vocabulary."""
    raw = str(finding.get("review_state", "") or "").strip().lower()
    return REVIEW_STATE_ALIASES.get(raw, raw)


@dataclass
class ReportTemplate:
    """What an organisation wants in its deliverable."""

    title: str = "Inspection Report"
    organization: str = ""
    client: str = ""
    logo_path: str = ""
    author: str = ""
    # Section keys, in the order they should appear.
    sections: list[str] = field(default_factory=lambda: [
        "summary", "methodology", "findings", "measurements", "recommendations",
    ])
    include_unreviewed: bool = True
    severity_order: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "organization": self.organization, "client": self.client,
            "logo_path": self.logo_path, "author": self.author, "sections": list(self.sections),
            "include_unreviewed": self.include_unreviewed, "severity_order": self.severity_order,
        }


def _severity_rank(finding: dict[str, Any]) -> tuple[int, str]:
    severity = str(finding.get("severity", "")).lower()
    return SEVERITY_ORDER.get(severity, len(SEVERITY_ORDER)), str(finding.get("category", ""))


def prepare_findings(
    findings: list[dict[str, Any]], template: ReportTemplate
) -> list[dict[str, Any]]:
    """Filter and order findings, numbering them stably.

    Numbering happens after ordering so the numbers match the printed sequence, and a
    finding referred to as "3" in correspondence is the third one in the document.
    """
    selected = []
    for finding in findings:
        state = review_state(finding)
        # A finding the reviewer rejected is a model's false positive, not a finding.
        # Printing it would put a defect the inspector ruled out in front of the client.
        if state == "rejected":
            continue
        if not template.include_unreviewed and state != "accepted":
            continue
        entry = dict(finding)
        entry["review_state"] = state
        selected.append(entry)
    if template.severity_order:
        selected.sort(key=_severity_rank)
    for index, finding in enumerate(selected, start=1):
        finding["number"] = index
    return selected


def _measurement_caveat(findings: list[dict[str, Any]]) -> str:
    """State plainly when totals cover only part of what was found."""
    unmeasured = sum(
        1 for finding in findings
        if not finding.get("area_m2") and not finding.get("length_m")
    )
    if not unmeasured:
        return ""
    return (
        f"{unmeasured} of {len(findings)} findings carry no measured extent, so the "
        "area and length totals below describe only the measured ones."
    )


def _unreviewed_caveat(findings: list[dict[str, Any]]) -> str:
    """State when the document contains claims no human has confirmed."""
    model_unreviewed = sum(
        1 for finding in findings
        if finding.get("source") == "model" and review_state(finding) != "accepted"
    )
    if not model_unreviewed:
        return ""
    return (
        f"{model_unreviewed} finding(s) in this report were produced by an automated "
        "model and have not been confirmed by an inspector. They are included for "
        "review and should not be treated as verified."
    )


def _rejected_caveat(all_findings: list[dict[str, Any]]) -> str:
    rejected = sum(1 for finding in all_findings if review_state(finding) == "rejected")
    if not rejected:
        return ""
    return (
        f"{rejected} proposed finding(s) were rejected by the reviewer and are not "
        "included in this report."
    )


def _caveats(all_findings: list[dict[str, Any]], findings: list[dict[str, Any]],
             payload: dict[str, Any], sections: list[str] | None = None) -> list[str]:
    found: list[str] = []
    if sections is None or "findings" in sections:
        # About the findings table; meaningless in a document that has none.
        found += [
            _unreviewed_caveat(findings),
            _measurement_caveat(findings),
            _rejected_caveat(all_findings),
        ]
    found.extend(str(c) for c in payload.get("caveats", []) or [])
    return [c for c in found if c]


def _where(finding: dict[str, Any]) -> str:
    """Source image and, when known, the map position of a finding."""
    parts = []
    if finding.get("image"):
        parts.append(str(finding["image"]))
    location = finding.get("location_lonlat")
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        # Say what the position is: for a photo finding it is where the camera was.
        kind = finding.get("location_kind") or "position"
        parts.append(f"{kind} {float(location[1]):.6f}, {float(location[0]):.6f}")
    return "; ".join(parts) or "-"


def _extent(finding: dict[str, Any]) -> str:
    parts = []
    if finding.get("area_m2"):
        parts.append(f"{finding['area_m2']:.3f} m2")
    if finding.get("length_m"):
        parts.append(f"{finding['length_m']:.2f} m")
    return ", ".join(parts) or "not measured"


def _usable_image(path: Any) -> str:
    if not path:
        return ""
    candidate = Path(str(path))
    return str(candidate) if candidate.is_file() else ""


def _image_size(path: str, max_w: float, max_h: float) -> tuple[float, float]:
    """Fit an image into a box, keeping its aspect ratio."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001 - unreadable image: fall back to the box
        return max_w, max_h
    scale = min(max_w / max(width, 1), max_h / max(height, 1))
    return width * scale, height * scale


def _measurement_rows(measurements: Any) -> list[list[str]]:
    if isinstance(measurements, dict):
        return [[str(k).replace("_", " "), str(v)] for k, v in measurements.items()]
    rows = []
    for item in measurements or []:
        if isinstance(item, dict):
            rows.append([str(item.get("label") or item.get("measurement_type") or "-"),
                         f"{item.get('value', '-')} {item.get('unit', '')}".strip()])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append([str(item[0]), str(item[1])])
    return rows


def write_pdf(path: str | Path, payload: dict[str, Any], template: ReportTemplate | None = None) -> str:
    """Render a PDF deliverable.

    Besides the named sections (summary, methodology, findings, measurements,
    recommendations) any section key in the template that matches an entry in
    payload["blocks"] is rendered as a generic block: paragraphs, an optional table and
    optional figures. That is how the report engine hands over site, mission, dataset,
    reconstruction and audit sections without this module knowing about each one.
    """
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    template = template or ReportTemplate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    caveat_style = ParagraphStyle(
        "Caveat", parent=styles["BodyText"], textColor=colors.HexColor("#8a4b00"),
        backColor=colors.HexColor("#fff4e5"), borderPadding=6, spaceBefore=6, spaceAfter=10,
    )
    cell_style = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8, leading=10)
    caption_style = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=8,
                                   textColor=colors.HexColor("#555555"))

    def para(text: Any, style: Any = None) -> Any:
        return Paragraph(escape(str(text)), style or styles["BodyText"])

    def grid(rows: list[list[Any]], widths: list[float], header: bool = True) -> Any:
        body = [[para(cell, cell_style) for cell in row] for row in rows]
        table = Table(body, repeatRows=1 if header else 0, colWidths=widths)
        style = [
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
        else:
            style.append(("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f4f4")))
        table.setStyle(TableStyle(style))
        return table

    def figure(image_path: str, caption: str, max_w: float = 160 * mm,
               max_h: float = 95 * mm) -> Any:
        width, height = _image_size(image_path, max_w, max_h)
        return KeepTogether([Image(image_path, width=width, height=height),
                             para(caption, caption_style), Spacer(1, 3 * mm)])

    document = SimpleDocTemplate(
        str(target), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=template.title, author=template.author or template.organization or "OpenDroneKit",
    )

    all_findings = list(payload.get("findings", []) or [])
    findings = prepare_findings(all_findings, template)
    story: list[Any] = []

    logo = _usable_image(template.logo_path)
    if logo:
        width, height = _image_size(logo, 50 * mm, 20 * mm)
        story.append(Image(logo, width=width, height=height, hAlign="LEFT"))
    story.append(para(template.title, styles["Title"]))
    meta = [
        ["Organization", template.organization or "-"],
        ["Client", template.client or payload.get("client", "-") or "-"],
        ["Project", str(payload.get("project_name", "-"))],
        ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    if template.author:
        meta.append(["Prepared by", template.author])
    if payload.get("report_type"):
        meta.append(["Report type", str(payload["report_type"])])
    if payload.get("crs_epsg"):
        # A report carrying measurements must say which coordinate system they are in.
        meta.append(["Coordinate system", f"EPSG:{payload['crs_epsg']}"])
    story.extend([Spacer(1, 6 * mm), grid(meta, [45 * mm, 110 * mm], header=False),
                  Spacer(1, 8 * mm)])

    for caveat in _caveats(all_findings, findings, payload, template.sections):
        story.append(para(caveat, caveat_style))

    blocks = payload.get("blocks", {}) or {}
    for section in template.sections:
        if section == "summary" and payload.get("summary"):
            story.append(para("Executive summary", styles["Heading2"]))
            story.append(para(payload["summary"]))
            story.append(Spacer(1, 4 * mm))
        elif section == "methodology" and payload.get("methodology"):
            story.append(para("Methodology", styles["Heading2"]))
            for line in str(payload["methodology"]).split("\n"):
                if line.strip():
                    story.append(para(line))
            story.append(Spacer(1, 4 * mm))
        elif section == "findings":
            story.append(para(f"Findings ({len(findings)})", styles["Heading2"]))
            if findings:
                rows = [["#", "Category", "Severity", "Where", "Extent", "Source", "Review"]]
                for finding in findings:
                    rows.append([
                        str(finding["number"]),
                        str(finding.get("category", "-")),
                        str(finding.get("severity", "-")),
                        _where(finding),
                        _extent(finding),
                        str(finding.get("source", "-")),
                        str(finding.get("review_state", "-")),
                    ])
                story.append(grid(rows, [8 * mm, 26 * mm, 17 * mm, 45 * mm, 22 * mm,
                                         17 * mm, 20 * mm]))
                story.append(Spacer(1, 4 * mm))
                for finding in findings:
                    note = str(finding.get("note") or "").strip()
                    crop = _usable_image(finding.get("image_path"))
                    if not note and not crop:
                        continue
                    detail: list[Any] = [para(
                        f"Finding {finding['number']}: {finding.get('category', '-')} "
                        f"({finding.get('severity', '-')})", styles["Heading4"])]
                    if note:
                        detail.append(para(note))
                    if crop:
                        width, height = _image_size(crop, 90 * mm, 60 * mm)
                        detail.append(Image(crop, width=width, height=height, hAlign="LEFT"))
                        detail.append(para(f"Source image: {finding.get('image', '-')}",
                                           caption_style))
                    detail.append(Spacer(1, 3 * mm))
                    story.append(KeepTogether(detail))
            else:
                story.append(para("No findings were recorded for this inspection."))
            story.append(Spacer(1, 6 * mm))
        elif section == "measurements" and payload.get("measurements"):
            rows = _measurement_rows(payload["measurements"])
            if rows:
                story.append(para("Measurements", styles["Heading2"]))
                story.append(grid([["Quantity", "Value"]] + rows, [70 * mm, 85 * mm]))
                story.append(Spacer(1, 6 * mm))
        elif section == "recommendations" and payload.get("recommendations"):
            story.append(para("Recommendations", styles["Heading2"]))
            for item in payload["recommendations"]:
                story.append(para(f"- {item}"))
            story.append(Spacer(1, 4 * mm))
        elif section in blocks:
            block = blocks[section] or {}
            story.append(para(block.get("title") or section.replace("_", " ").title(),
                              styles["Heading2"]))
            for line in block.get("paragraphs", []) or []:
                story.append(para(line))
            table = block.get("table")
            if table and table.get("rows"):
                header = [str(h) for h in table.get("header", [])]
                columns = max(len(header), max(len(r) for r in table["rows"]))
                width = 155 * mm / max(columns, 1)
                rows = ([header] if header else []) + [[str(c) for c in r] for r in table["rows"]]
                story.append(grid(rows, [width] * columns, header=bool(header)))
            for image in block.get("images", []) or []:
                usable = _usable_image(image.get("path"))
                if usable:
                    story.append(figure(usable, str(image.get("caption", ""))))
            story.append(Spacer(1, 5 * mm))

    document.build(story)
    return str(target)


def write_docx(path: str | Path, payload: dict[str, Any], template: ReportTemplate | None = None) -> str:
    """Render an editable DOCX deliverable."""
    from docx import Document
    from docx.shared import Mm, Pt

    template = template or ReportTemplate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    all_findings = list(payload.get("findings", []) or [])
    findings = prepare_findings(all_findings, template)
    document = Document()
    document.core_properties.title = template.title
    document.core_properties.author = template.author or template.organization or "OpenDroneKit"

    def picture(image_path: str, max_w_mm: float, max_h_mm: float) -> None:
        width, height = _image_size(image_path, max_w_mm, max_h_mm)
        document.add_picture(image_path, width=Mm(width), height=Mm(height))

    logo = _usable_image(template.logo_path)
    if logo:
        picture(logo, 50, 20)
    document.add_heading(template.title, level=0)

    meta = document.add_table(rows=0, cols=2)
    meta.style = "Table Grid"
    entries = [
        ("Organization", template.organization or "-"),
        ("Client", template.client or str(payload.get("client", "-")) or "-"),
        ("Project", str(payload.get("project_name", "-"))),
        ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
    ]
    if template.author:
        entries.append(("Prepared by", template.author))
    if payload.get("report_type"):
        entries.append(("Report type", str(payload["report_type"])))
    if payload.get("crs_epsg"):
        entries.append(("Coordinate system", f"EPSG:{payload['crs_epsg']}"))
    for label, value in entries:
        row = meta.add_row().cells
        row[0].text = label
        row[1].text = value

    for caveat in _caveats(all_findings, findings, payload, template.sections):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(caveat)
        run.bold = True
        run.font.size = Pt(9)

    blocks = payload.get("blocks", {}) or {}
    for section in template.sections:
        if section == "summary" and payload.get("summary"):
            document.add_heading("Executive summary", level=1)
            document.add_paragraph(str(payload["summary"]))
        elif section == "methodology" and payload.get("methodology"):
            document.add_heading("Methodology", level=1)
            for line in str(payload["methodology"]).split("\n"):
                if line.strip():
                    document.add_paragraph(line)
        elif section == "findings":
            document.add_heading(f"Findings ({len(findings)})", level=1)
            if findings:
                table = document.add_table(rows=1, cols=7)
                table.style = "Table Grid"
                header = table.rows[0].cells
                for index, label in enumerate(
                    ["#", "Category", "Severity", "Where", "Extent", "Source", "Review"]
                ):
                    header[index].text = label
                for finding in findings:
                    cells = table.add_row().cells
                    cells[0].text = str(finding["number"])
                    cells[1].text = str(finding.get("category", "-"))
                    cells[2].text = str(finding.get("severity", "-"))
                    cells[3].text = _where(finding)
                    cells[4].text = _extent(finding)
                    cells[5].text = str(finding.get("source", "-"))
                    cells[6].text = str(finding.get("review_state", "-"))
                for finding in findings:
                    note = str(finding.get("note") or "").strip()
                    crop = _usable_image(finding.get("image_path"))
                    if not note and not crop:
                        continue
                    document.add_heading(
                        f"Finding {finding['number']}: {finding.get('category', '-')} "
                        f"({finding.get('severity', '-')})", level=3)
                    if note:
                        document.add_paragraph(note)
                    if crop:
                        picture(crop, 90, 60)
                        document.add_paragraph(f"Source image: {finding.get('image', '-')}")
            else:
                document.add_paragraph("No findings were recorded for this inspection.")
        elif section == "measurements" and payload.get("measurements"):
            rows = _measurement_rows(payload["measurements"])
            if rows:
                document.add_heading("Measurements", level=1)
                table = document.add_table(rows=1, cols=2)
                table.style = "Table Grid"
                table.rows[0].cells[0].text = "Quantity"
                table.rows[0].cells[1].text = "Value"
                for key, value in rows:
                    cells = table.add_row().cells
                    cells[0].text = key
                    cells[1].text = value
        elif section == "recommendations" and payload.get("recommendations"):
            document.add_heading("Recommendations", level=1)
            for item in payload["recommendations"]:
                document.add_paragraph(str(item), style="List Bullet")
        elif section in blocks:
            block = blocks[section] or {}
            document.add_heading(block.get("title") or section.replace("_", " ").title(), level=1)
            for line in block.get("paragraphs", []) or []:
                document.add_paragraph(str(line))
            table_data = block.get("table")
            if table_data and table_data.get("rows"):
                header = [str(h) for h in table_data.get("header", [])]
                columns = max(len(header), max(len(r) for r in table_data["rows"]))
                table = document.add_table(rows=0, cols=columns)
                table.style = "Table Grid"
                for row in ([header] if header else []) + table_data["rows"]:
                    cells = table.add_row().cells
                    for index, value in enumerate(list(row)[:columns]):
                        cells[index].text = str(value)
            for image in block.get("images", []) or []:
                usable = _usable_image(image.get("path"))
                if usable:
                    picture(usable, 160, 95)
                    document.add_paragraph(str(image.get("caption", "")))

    document.save(str(target))
    return str(target)


WRITERS = {"pdf": (write_pdf, ".pdf"), "docx": (write_docx, ".docx")}
