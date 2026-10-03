"""The PDF layout of a standards-structured inspection report (reportlab).

Order: cover (hero image, site, date, client, status) -> "Before this report can be
issued" checklist (drafts only) -> executive summary -> site map / inspected area ->
findings with image crops -> photo evidence -> method and conditions -> Appendix A
(clause-by-clause items, compact, with sources) -> Appendix B (checks, sources) ->
signature and input manifest.

Typography is Helvetica throughout (no open font ships with the repository), colour is
used sparingly: one accent for structure, an amber outline for the draft status, and
severity colours on findings only. Open items are never printed as red MISSING in the
body; they are collected on the checklist page.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

from .standards_reports import MISSING, WHO_LABELS, _cell, status_line, value_text

INK = colors.HexColor("#1d2433")
MUTED = colors.HexColor("#5b6475")
FAINT = colors.HexColor("#8a909c")
RULE = colors.HexColor("#d9dde5")
PANEL = colors.HexColor("#f3f4f6")
ACCENT = colors.HexColor("#0f5f66")
DRAFT = colors.HexColor("#9a5b00")
SEVERITY = {3: colors.HexColor("#b42318"), 2: colors.HexColor("#dc6803"),
            1: colors.HexColor("#3b7d4f"), None: colors.HexColor("#667085")}

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
TOP = 22 * mm
BOTTOM = 18 * mm
WIDTH = PAGE_W - 2 * MARGIN_X

WHO_ORDER = ("profile", "procedure", "lookup", "site", "data", "operator", "judgement")
WHO_INTRO = {
    "profile": "People: pick saved operator profiles",
    "procedure": "Procedure: adopt the procedure template",
    "lookup": "Components: pick the module from the CEC module list",
    "site": "Site conditions: record in Field mode or import",
    "data": "Not in the project's files: supply once",
    "operator": "Enter for this inspection",
    "judgement": "Assessment by a qualified person",
}


def _style(name: str, **kw: Any) -> ParagraphStyle:
    base = {"fontName": "Helvetica", "fontSize": 9, "leading": 12.5, "textColor": INK,
            "alignment": TA_LEFT}
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "kicker": _style("kicker", fontSize=8, leading=10, textColor=MUTED, fontName="Helvetica-Bold"),
    "h1": _style("h1", fontName="Helvetica-Bold", fontSize=24, leading=28),
    "h2": _style("h2", fontName="Helvetica-Bold", fontSize=15, leading=19, spaceBefore=2,
                 spaceAfter=6),
    "h3": _style("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=8,
                 spaceAfter=4),
    "body": _style("body"),
    "small": _style("small", fontSize=7.8, leading=10),
    "muted": _style("muted", fontSize=7.8, leading=10, textColor=MUTED),
    "faint": _style("faint", fontSize=7.5, leading=9.5, textColor=FAINT,
                    fontName="Helvetica-Oblique"),
    "cell": _style("cell", fontSize=7.6, leading=9.6),
    "cellmuted": _style("cellmuted", fontSize=7.2, leading=9.2, textColor=MUTED),
    "cellopen": _style("cellopen", fontSize=7.4, leading=9.4, textColor=FAINT,
                       fontName="Helvetica-Oblique"),
    "mono": _style("mono", fontName="Courier", fontSize=6.2, leading=7.6, textColor=MUTED),
    "label": _style("label", fontSize=7, leading=9, textColor=MUTED),
    "value": _style("value", fontName="Helvetica-Bold", fontSize=10, leading=13),
    "kpi": _style("kpi", fontName="Helvetica-Bold", fontSize=17, leading=20),
}


def P(text: Any, style: str = "body") -> Paragraph:
    return Paragraph(html.escape("" if text is None else str(text)).replace("\n", "<br/>"),
                     S[style])


def _rich(markup: str, style: str = "body") -> Paragraph:
    return Paragraph(markup, S[style])


# ── Page furniture ────────────────────────────────────────────────────────────

class _NumberedCanvas(rl_canvas.Canvas):
    """Draws "Page n of N" once the page count is known."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802 (reportlab API)
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont("Helvetica", 7.5)
                self.setFillColor(MUTED)
                self.drawRightString(PAGE_W - MARGIN_X, 10 * mm,
                                     f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _furniture(meta: dict[str, Any], content: dict[str, Any]):
    title = str(meta.get("title") or content["title"])
    status = status_line(content)
    draft = content["status"] == "DRAFT"

    def body(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.6)
        canvas.line(MARGIN_X, PAGE_H - 14 * mm, PAGE_W - MARGIN_X, PAGE_H - 14 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(INK)
        canvas.drawString(MARGIN_X, PAGE_H - 12 * mm, title)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(DRAFT if draft else ACCENT)
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 12 * mm,
                               "DRAFT — not for issue" if draft else "Ready to issue")
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN_X, 14 * mm, PAGE_W - MARGIN_X, 14 * mm)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN_X, 10 * mm, f"Report {meta.get('report_id', '')[:8]} · "
                          f"{content['title']} · OpenDroneKit")
        canvas.restoreState()

    def cover(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFillColor(ACCENT)
        canvas.rect(0, PAGE_H - 6 * mm, PAGE_W, 6 * mm, stroke=0, fill=1)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN_X, PAGE_H - 16 * mm, "OPENDRONEKIT · INSPECTION REPORT")
        # Status pill, top right.
        canvas.setFont("Helvetica-Bold", 8.5)
        text_w = canvas.stringWidth(status, "Helvetica-Bold", 8.5)
        x1 = PAGE_W - MARGIN_X
        canvas.setStrokeColor(DRAFT if draft else ACCENT)
        canvas.setFillColor(colors.white)
        canvas.setLineWidth(1.2)
        canvas.roundRect(x1 - text_w - 14, PAGE_H - 19 * mm, text_w + 14, 7.5 * mm, 2, stroke=1,
                         fill=1)
        canvas.setFillColor(DRAFT if draft else ACCENT)
        canvas.drawRightString(x1 - 7, PAGE_H - 19 * mm + 2.6 * mm, status)
        canvas.restoreState()

    return cover, body


# ── Building blocks ───────────────────────────────────────────────────────────

def _image(path: str | None, max_w: float, max_h: float) -> Image | None:
    if not path or not Path(path).is_file():
        return None
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), w * scale, h * scale)


def _rule(width: float = WIDTH, color: Any = RULE, thickness: float = 0.6) -> Table:
    t = Table([[""]], colWidths=[width], rowHeights=[thickness])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), thickness, color),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def _grid(rows: list[list[Any]], widths: list[float], *, header: bool = True,
          zebra: bool = False, pad: float = 3.2) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    if zebra:
        for i in range(1 if header else 0, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), PANEL))
    t.setStyle(TableStyle(style))
    return t


def _pill(text: str, color: Any) -> Table:
    t = Table([[_rich(f"<font color='white'><b>{html.escape(text)}</b></font>", "small")]])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    return t


def _section(title: str, kicker: str = "") -> list[Any]:
    out: list[Any] = []
    if kicker:
        out.append(P(kicker.upper(), "kicker"))
    out += [P(title, "h2"), _rule(color=INK, thickness=0.9), Spacer(1, 6)]
    return out


def _tiles(tiles: list[dict[str, str]]) -> Table:
    tiles = tiles[:4]
    n = max(1, len(tiles))
    gap = 4 * mm
    w = (WIDTH - gap * (n - 1)) / n
    cells = []
    for tile in tiles:
        inner = Table([[P(tile["value"], "kpi")], [P(tile["label"], "label")]], colWidths=[w - 8])
        inner.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("TOPPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        cells.append(inner)
    widths = []
    row = []
    for i, cell in enumerate(cells):
        row.append(cell)
        widths.append(w)
        if i < len(cells) - 1:
            row.append("")
            widths.append(gap)
    t = Table([row], colWidths=widths)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
             ("LEFTPADDING", (0, 0), (-1, -1), 8)]
    for i in range(0, len(row), 2):
        style += [("BACKGROUND", (i, 0), (i, 0), PANEL),
                  ("LINEABOVE", (i, 0), (i, 0), 1.6, ACCENT)]
    t.setStyle(TableStyle(style))
    return t


# ── Pages ─────────────────────────────────────────────────────────────────────

def _figures(content: dict[str, Any], role: str) -> list[dict[str, Any]]:
    return [f for f in content.get("figures") or [] if f.get("role") == role and f.get("rendered")]


def _cover(content: dict[str, Any], meta: dict[str, Any]) -> list[Any]:
    details = content.get("details") or {}
    story: list[Any] = [Spacer(1, 12 * mm), P(content["title"].upper(), "kicker"), Spacer(1, 3),
                        P(meta.get("title") or content["title"], "h1"), Spacer(1, 8)]
    hero = (_figures(content, "hero") or [None])[0]
    img = _image(hero["rendered"], WIDTH, 112 * mm) if hero else None
    if img is not None:
        frame = Table([[img]], colWidths=[WIDTH])
        frame.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                   ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                                   ("TOPPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        story += [frame, Spacer(1, 2), P(hero.get("caption", ""), "muted")]
    else:
        box = Table([[P("No imagery has been added to this inspection yet.", "muted")]],
                    colWidths=[WIDTH], rowHeights=[70 * mm])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL),
                                 ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(box)
    story.append(Spacer(1, 9 * mm))

    def field(label: str, value: Any, open_text: str = "to be entered") -> Table:
        t = Table([[P(label.upper(), "label")],
                   [P(value, "value") if value else P(open_text, "faint")]], colWidths=[WIDTH / 3 - 6])
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return t

    comp = content["completeness"]
    grid = Table([
        [field("Site", details.get("site")), field("Inspection date", details.get("date")),
         field("Client", details.get("client"))],
        [field("Report ID", str(meta.get("report_id", ""))[:18]),
         field("Generated", str(meta.get("generated_at", ""))[:16].replace("T", " ") + " UTC"),
         field("Items supplied", f"{comp['provided_items']} of {comp['required_items']}")],
    ], colWidths=[WIDTH / 3] * 3)
    grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
                              ("LINEABOVE", (0, 1), (-1, 1), 0.4, RULE),
                              ("TOPPADDING", (0, 0), (-1, -1), 6),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [grid, Spacer(1, 7 * mm),
              _rich(f"<b>{html.escape(content['structured_following'])}.</b> "
                    f"{html.escape(content['statement'])}", "muted")]
    return story


def _checklist(content: dict[str, Any]) -> list[Any]:
    todo = content["todo"]
    comp = content["completeness"]
    story = _section("Before this report can be issued", "Draft checklist")
    story.append(_rich(
        f"This report is a <b>draft</b>: <b>{len(todo)}</b> item{'s are' if len(todo) != 1 else ' is'} "
        "still open. Each one is listed once here instead of throughout the report. "
        f"{comp['provided_items']} of {comp['required_items']} required items are already "
        f"supplied ({comp['derived_items']} derived from the project's data, "
        f"{comp['reused_items']} from saved profiles and templates, {comp['entered_items']} "
        "entered for this inspection), each with its source in Appendix A.", "body"))
    story.append(Spacer(1, 6))
    for who in WHO_ORDER:
        group = [t for t in todo if t["who"] == who]
        if not group:
            continue
        story.append(CondPageBreak(30 * mm))
        story.append(_rich(f"<b>{html.escape(WHO_INTRO[who])}</b> "
                           f"<font color='#5b6475'>({len(group)})</font>", "h3"))
        rows = [[P("", "cell"), P("Ref", "cellmuted"), P("Item", "cellmuted"),
                 P("How to complete", "cellmuted")]]
        for entry in group:
            item = [P(entry["label"], "cell")]
            sug = entry.get("suggestion")
            if sug:
                item.append(_rich(f"<font color='#0f5f66'><b>Suggested:</b></font> "
                                  f"{html.escape(_cell(sug['value']))[:260]}", "cellmuted"))
            box = Table([[""]], colWidths=[3.2 * mm], rowHeights=[3.2 * mm])
            box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, INK)]))
            rows.append([box, P(entry["ref"], "cellmuted"), item, P(entry["hint"], "cellmuted")])
        story.append(_grid(rows, [7 * mm, 22 * mm, 74 * mm, WIDTH - 103 * mm]))
    return story


def _summary(content: dict[str, Any]) -> list[Any]:
    story = _section("Executive summary", "Overview")
    comp = content["completeness"]
    summary = content.get("summary") or {}
    tiles = list(summary.get("kpis") or [])[:2]
    tiles.append({"label": "Required items supplied",
                  "value": f"{comp['provided_items']} / {comp['required_items']}"})
    tiles.append({"label": "Open checklist items", "value": str(comp["open_actions"])})
    story += [_tiles(tiles), Spacer(1, 8)]
    lines = []
    sug = next((i for s in content["sections"] for i in s["items"]
                if i["id"] in ("results.summary",) and i.get("suggestion")), None)
    done = next((i for s in content["sections"] for i in s["items"]
                 if i["id"] in ("results.summary",) and i["status"] != MISSING), None)
    if done:
        lines.append(P(_cell(done["value"]), "body"))
    elif sug:
        lines.append(_rich("<font color='#0f5f66'><b>Draft summary (suggested, to be "
                           "confirmed by the thermographer):</b></font> "
                           + html.escape(_cell(sug["suggestion"]["value"])), "body"))
    for line in summary.get("lines") or []:
        lines.append(P(line, "body"))
    story += lines
    counts = summary.get("severity_counts") or {}
    findings = content.get("findings") or []
    story.append(P("Findings", "h3"))
    if counts:
        rows = []
        total = max(counts.values())
        levels = {f["severity"]["label"]: f["severity"].get("level") for f in findings}
        for label, n in counts.items():
            level = levels.get(label)
            bar_w = max(2.0, (WIDTH - 80 * mm) * n / max(1, total))
            bar = Table([[""]], colWidths=[bar_w], rowHeights=[3.6 * mm])
            bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SEVERITY.get(level, SEVERITY[None]))]))
            rows.append([P(label, "cell"), P(str(n), "cell"), bar])
        t = Table(rows, colWidths=[62 * mm, 14 * mm, WIDTH - 76 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                               ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(t)
        story.append(P("Classes marked 'suggested' and model proposals are not findings until "
                       "a qualified person confirms them in review; they are counted here so "
                       "the reviewer sees the workload.", "muted"))
    else:
        story.append(P("No findings have been recorded for this inspection.", "faint"))
    if content["checks"]:
        story.append(P("Inspection conditions and equipment against the standard", "h3"))
        rows = [[P("Ref", "cellmuted"), P("Requirement", "cellmuted"), P("Result", "cellmuted"),
                 P("Detail", "cellmuted")]]
        for check in content["checks"]:
            result = {"PASS": "<b>Met</b>", "FAIL": "<b>Not met</b>", "N/A": "N/A"}.get(
                check["status"], "<font color='#8a909c'><i>Open</i></font>")
            detail = (check["detail"] if check["status"] != MISSING else
                      check["detail"].replace("Cannot be checked: ", "Needs ")
                      .replace(" was not supplied.", " (see checklist)."))
            rows.append([P(check["ref"], "cellmuted"), P(check["label"], "cell"),
                         _rich(result, "cell"), P(detail, "cellmuted")])
        story.append(_grid(rows, [24 * mm, 52 * mm, 18 * mm, WIDTH - 94 * mm]))
    return story


def _map_pages(content: dict[str, Any]) -> list[Any]:
    maps = _figures(content, "map")
    overviews = _figures(content, "overview")
    if not maps and not overviews:
        return []
    story: list[Any] = [PageBreak()]
    story += _section("Site and inspected area" if maps else "Inspected area", "Location")
    details = content.get("details") or {}
    if details.get("site"):
        story.append(_rich(f"<b>Site:</b> {html.escape(str(details['site']))}", "body"))
        story.append(Spacer(1, 6))
    for fig in maps:
        img = _image(fig["rendered"], WIDTH, 150 * mm)
        if img is not None:
            story += [img, Spacer(1, 2), P(fig.get("caption", ""), "muted"), Spacer(1, 8)]
    if overviews:
        w = (WIDTH - 6 * mm) / 2 if len(overviews) > 1 else WIDTH
        cells = []
        for fig in overviews:
            img = _image(fig["rendered"], w, 70 * mm)
            if img is not None:
                cells.append([img, Spacer(1, 2), P(fig.get("caption", ""), "muted")])
        rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
        for row in rows:
            while len(row) < 2 and len(overviews) > 1:
                row.append("")
            t = Table([row], colWidths=[w, w] if len(overviews) > 1 else [WIDTH])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm)]))
            story.append(t)
    return story


def _finding_card(finding: dict[str, Any]) -> Any:
    sev = finding.get("severity") or {}
    color = SEVERITY.get(sev.get("level"), SEVERITY[None])
    head = Table([[P(finding.get("title", ""), "h3"), _pill(sev.get("label", ""), color)]],
                 colWidths=[WIDTH - 42 * mm, 42 * mm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    img = _image(finding.get("rendered"), 100 * mm, 72 * mm)
    facts = [[P(k, "cellmuted"), P(v, "cell")] for k, v in finding.get("facts") or []]
    fact_table = Table(facts, colWidths=[30 * mm, WIDTH - 104 * mm - 30 * mm]) if facts else P("")
    if facts:
        fact_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                                        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    right = [fact_table, Spacer(1, 5), P(finding.get("note", ""), "small"), Spacer(1, 4),
             P(f"Source: {finding.get('source', '')}", "muted")]
    body = Table([[img or P("image unavailable", "faint"), right]],
                 colWidths=[104 * mm, WIDTH - 104 * mm])
    body.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (0, 0), 4 * mm)]))
    stripe = Table([[head], [body]], colWidths=[WIDTH])
    stripe.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 2.2, color),
                                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                ("TOPPADDING", (0, 0), (-1, -1), 2),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return KeepTogether([stripe, Spacer(1, 9)])


def _findings(content: dict[str, Any]) -> list[Any]:
    findings = [f for f in content.get("findings") or []]
    story: list[Any] = [PageBreak()]
    story += _section("Findings", "Results")
    if not findings:
        story.append(P("No findings or proposals have been recorded. Findings are added by the "
                       "inspector in review; the deficiencies table in Appendix A lists them "
                       "once recorded.", "faint"))
        return story
    proposals = all((f.get("severity") or {}).get("level") is None for f in findings)
    story.append(P("Each card shows the image the value was read from. "
                   + ("These are detector proposals for the inspector's review; none is a finding "
                      "until accepted." if proposals else
                      "Classes are suggestions until the thermographer confirms them in review."),
                   "muted"))
    story.append(Spacer(1, 6))
    for finding in findings:
        story.append(_finding_card(finding))
    return story


def _evidence(content: dict[str, Any]) -> list[Any]:
    items = [e for e in content.get("evidence") or [] if e.get("rendered")]
    if not items:
        return []
    story: list[Any] = [CondPageBreak(90 * mm)]
    story += _section("Photo evidence", "Evidence")
    w = (WIDTH - 6 * mm) / 2
    cells = []
    for ev in items[:8]:
        img = _image(ev["rendered"], w, 62 * mm)
        if img is not None:
            cells.append([img, Spacer(1, 2), P(ev.get("caption", ""), "muted")])
    for i in range(0, len(cells), 2):
        row = cells[i:i + 2] + ([""] if len(cells[i:i + 2]) == 1 else [])
        t = Table([row], colWidths=[w + 3 * mm, w + 3 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.append(t)
    return story


def _method(content: dict[str, Any]) -> list[Any]:
    story: list[Any] = [PageBreak()]
    story += _section("Method and conditions", "Method")
    items = {i["id"]: i for s in content["sections"] for i in s["items"]}
    proc = next((items[k] for k in ("inspection.procedure",) if k in items), None)
    kind = next((items[k] for k in ("inspection.type", "ie.type") if k in items), None)
    method = items.get("general.method")
    story.append(P("Procedure", "h3"))
    extra = (content.get("details") or {}).get("procedure")
    if not any((kind, proc, method)) and extra and extra.get("text"):
        tag = "" if extra.get("adopted") else (" <font color='#0f5f66'><b>(suggested default, "
                                               "not yet adopted)</b></font>")
        story.append(_rich(html.escape(extra["text"]) + tag, "body"))
        story.append(P(extra.get("source", ""), "muted"))
    elif not any((kind, proc, method)):
        story.append(P("No procedure recorded.", "faint"))
    for it in (kind, proc, method):
        if it is None:
            continue
        if it["status"] == MISSING:
            sug = it.get("suggestion")
            story.append(_rich(f"<b>{html.escape(it['label'])}:</b> <font color='#8a909c'><i>"
                               "open — see the checklist"
                               + (" (a default procedure is suggested there)" if sug else "")
                               + "</i></font>", "body"))
        else:
            story.append(_rich(f"<b>{html.escape(it['label'])}:</b> "
                               f"{html.escape(_cell(it['value']))}", "body"))
            story.append(P(it["source"], "muted"))
        story.append(Spacer(1, 4))
    groups = [("Equipment", ("camera", "ie.equipment")),
              ("Conditions at the time of inspection", ("env.", "general.weather", "site.weather"))]
    for title, prefixes in groups:
        rows = [[P("Item", "cellmuted"), P("Value", "cellmuted"), P("Source", "cellmuted")]]
        for item in items.values():
            if not item["id"].startswith(prefixes):
                continue
            rows.append([P(item["label"], "cell"),
                         P(value_text(item), "cellopen" if item["status"] == MISSING else "cell"),
                         P(item["source"] or "", "cellmuted")])
        if len(rows) > 1:
            story += [CondPageBreak(40 * mm), P(title, "h3"),
                      _grid(rows, [52 * mm, 48 * mm, WIDTH - 100 * mm])]
    for m in content.get("measurements") or []:
        rows = [[P(c, "cellmuted") for c in m["columns"]]]
        rows += [[P(v, "cell") for v in r] for r in m["rows"]]
        n = len(m["columns"])
        story += [CondPageBreak(45 * mm), P(m["title"], "h3"),
                  _grid(rows, [WIDTH / n] * n)]
        if m.get("method"):
            story.append(P(m["method"], "muted"))
        if m.get("source"):
            story.append(P(f"Source: {m['source']}", "muted"))
        if m.get("status") == "suggested":
            story.append(P("Suggested measurement: the reviewer confirms each facet against the "
                           "imagery before it is relied on.", "muted"))
    obs = content.get("observations") or []
    if obs:
        story += [CondPageBreak(30 * mm), P("Supporting notes", "h3")]
        rows = [[P(o.get("text"), "cell"), P(o.get("source"), "cellmuted")] for o in obs]
        story.append(_grid(rows, [WIDTH * 0.66, WIDTH * 0.34], header=False))
    return story


def _appendix(content: dict[str, Any]) -> list[Any]:
    story: list[Any] = [PageBreak()]
    story += _section("Appendix A — Clause-by-clause items", "Appendix")
    story.append(P(f"Every required item of {content['standard']}, with the value supplied and "
                   "where it came from. Open items are on the checklist.", "muted"))
    story.append(Spacer(1, 4))
    for section in content["sections"]:
        story.append(CondPageBreak(25 * mm))
        story.append(_rich(f"<b>{html.escape(section['title'])}</b> "
                           f"<font color='#8a909c'>{html.escape(section['ref'])}</font>", "h3"))
        if section["items"]:
            rows = [[P("Ref", "cellmuted"), P("Item", "cellmuted"), P("Value", "cellmuted"),
                     P("Source", "cellmuted")]]
            for item in section["items"]:
                rows.append([P(item["ref"], "cellmuted"), P(item["label"], "cellmuted"),
                             P(value_text(item), "cellopen" if item["status"] == MISSING else "cell"),
                             P(item["source"] or f"open · {WHO_LABELS[item['who']].lower()}", "cellmuted")])
            story.append(_grid(rows, [17 * mm, 45 * mm, 52 * mm, WIDTH - 114 * mm], pad=2.2))
        for table in section["tables"]:
            story.append(_rich(f"{html.escape(table['title'])} "
                               f"<font color='#8a909c'>{html.escape(table['ref'])}</font>", "small"))
            if not table["rows"]:
                story.append(P("No rows recorded yet (see the checklist).", "faint"))
                continue
            for start in range(0, len(table["rows"]), 4):
                chunk = table["rows"][start:start + 4]
                rows = []
                for col in table["columns"]:
                    line = [P(col["label"], "cellmuted")]
                    for row in chunk:
                        value = row["cells"][col["key"]]
                        sug = (row.get("suggestions") or {}).get(col["key"])
                        if value == MISSING:
                            line.append(P(f"open ({sug['value']} suggested)" if sug else "open",
                                          "cellopen"))
                        else:
                            line.append(P(value, "cell"))
                    rows.append(line)
                rows.append([P("Source", "cellmuted")] + [P(r["source"], "cellmuted") for r in chunk])
                col_w = (WIDTH - 40 * mm) / len(chunk)
                story.append(_grid(rows, [40 * mm] + [col_w] * len(chunk), header=False, pad=2))
                story.append(Spacer(1, 4))
            if table.get("note"):
                story.append(P(table["note"], "muted"))
    story += [CondPageBreak(40 * mm), Spacer(1, 6)]
    story += _section("Appendix B — Sources for the report structure", "Appendix")
    for source in content["sources"]:
        story.append(_rich(f"{html.escape(source['label'])}<br/><font color='#0f5f66'>"
                           f"{html.escape(source['url'])}</font>", "small"))
        story.append(Spacer(1, 3))
    return story


def _signature(content: dict[str, Any], manifest: dict[str, Any], meta: dict[str, Any]) -> list[Any]:
    story: list[Any] = [PageBreak()]
    story += _section("Signature and input manifest", "Traceability")
    rows = [[P("Report ID", "cellmuted"), P(meta.get("report_id"), "cell")],
            [P("Generated", "cellmuted"), P(meta.get("generated_at"), "cell")],
            [P("Signing key (Ed25519)", "cellmuted"), P(meta.get("fingerprint"), "cell")],
            [P("Manifest", "cellmuted"),
             P(f"embedded in this PDF as {meta.get('manifest_attachment')}; "
               f"{len(manifest['inputs'])} input file(s), {len(manifest['models'])} model(s)", "cell")],
            [P("Verify", "cellmuted"),
             P("python -m core.report_signing verify report.pdf", "cell")]]
    t = Table(rows, colWidths=[42 * mm, WIDTH - 42 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL),
                           ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 4),
              P("The signature covers this PDF's bytes and the manifest. It proves the report "
                "has not changed since signing and which input files it was built from; it does "
                "not prove that the inspection meets the standard.", "muted"),
              Spacer(1, 10)]
    people = [i for s in content["sections"] for i in s["items"] if i["who"] == "profile"]
    sign = [[P("Approved for issue by", "cellmuted"), P("Role", "cellmuted"),
             P("Signature and date", "cellmuted")]]
    for item in people or [{"label": "Inspector", "status": MISSING}]:
        name = _cell(item.get("value")) if item.get("status") != MISSING else ""
        sign.append([P(name or "", "cell"), P(item["label"], "cell"), P("", "cell")])
    st = Table(sign, colWidths=[62 * mm, 52 * mm, WIDTH - 114 * mm],
               rowHeights=[None] + [13 * mm] * (len(sign) - 1))
    st.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, RULE),
                            ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                            ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    story += [st, Spacer(1, 10), P("Input files (SHA-256)", "h3")]
    if manifest["inputs"]:
        rows = [[P("Role", "cellmuted"), P("File", "cellmuted"), P("Bytes", "cellmuted"),
                 P("SHA-256", "cellmuted")]]
        rows += [[P(i["role"], "cellmuted"), P(i["name"], "cell"), P(i["bytes"], "cellmuted"),
                  P(i["sha256"], "mono")] for i in manifest["inputs"]]
        story.append(_grid(rows, [32 * mm, 52 * mm, 16 * mm, WIDTH - 100 * mm], pad=1.6))
    else:
        story.append(P("No input files: nothing in this report was read from data.", "faint"))
    story.append(P("Models", "h3"))
    if manifest["models"]:
        rows = [[P("Model", "cellmuted"), P("SHA-256", "cellmuted"), P("Digest source", "cellmuted")]]
        rows += [[P(m["name"], "cell"), P(m["sha256"], "mono"), P(m["digest_source"], "cellmuted")]
                 for m in manifest["models"]]
        story.append(_grid(rows, [44 * mm, WIDTH - 94 * mm, 50 * mm], pad=1.6))
    else:
        story.append(P("No model produced any content in this report.", "faint"))
    return story


def render_pdf(path: Path, content: dict[str, Any], manifest: dict[str, Any],
               meta: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                          topMargin=TOP, bottomMargin=BOTTOM,
                          title=str(meta.get("title") or content["title"]),
                          author="OpenDroneKit", subject=content["structured_following"])
    cover_cb, body_cb = _furniture(meta, content)
    frame = Frame(MARGIN_X, BOTTOM, WIDTH, PAGE_H - TOP - BOTTOM, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=cover_cb),
                          PageTemplate(id="body", frames=[frame], onPage=body_cb)])
    story: list[Any] = _cover(content, meta)
    story += [NextPageTemplate("body"), PageBreak()]
    if content["todo"]:
        story += _checklist(content)
        story.append(PageBreak())
    story += _summary(content)
    story += _map_pages(content)
    story += _findings(content)
    story += _evidence(content)
    story += _method(content)
    story += _appendix(content)
    story += _signature(content, manifest, meta)
    doc.build(story, canvasmaker=_NumberedCanvas)
    return path
