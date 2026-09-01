"""PDF and Word deliverables were written by code nothing called.

core/report_formats.py has had write_pdf and write_docx since that work landed, with
severity ordering, the measurement caveat and the unreviewed-findings caveat built in, and
tests covering all of it. Nothing in the application called either function -- the report
engine takes a different route, HTML then weasyprint -- so the Word output was
implemented, tested, and impossible to obtain.

That is the fourth time in this session that working code turned out to be unreachable
through the surface a user touches, after pylon/thermal/multispectral, the cockpit panels
and the template inventory. These tests are about the route rather than the rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api import Api
from app.session import AppSession
from core.annotations import create_annotation

GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}


@pytest.fixture
def project(tmp_path, monkeypatch) -> tuple[Api, Path]:
    api = Api(AppSession())
    monkeypatch.setattr(api._session, "project_root", lambda: tmp_path)
    return api, tmp_path


def add_finding(root: Path, severity: str = "medium", label: str = "crack") -> None:
    create_annotation(
        root, project_id="p", source_type="image", source_id="a.jpg",
        annotation_type="rectangle", geometry=GEOMETRY, label=label,
        severity=severity, status="open",
    )


def test_both_writers_are_reachable_from_the_application() -> None:
    """The specific failure: present in the module, absent from every route to it."""
    assert hasattr(Api, "export_report")
    assert hasattr(Api, "report_formats")


class TestExportingAReport:
    @pytest.mark.parametrize("fmt", ["pdf", "docx"])
    def test_it_writes_the_format_asked_for(self, project, fmt) -> None:
        api, root = project
        add_finding(root)
        result = api.export_report(fmt, title="Site inspection")
        assert result["ok"], result.get("error")
        written = Path(result["path"])
        assert written.suffix == f".{fmt}"
        assert written.stat().st_size > 0

    def test_the_findings_come_from_this_project(self, project) -> None:
        """Not a fixed example. A deliverable describing findings that were not recorded
        would be the worst possible output of a reporting feature."""
        api, root = project
        for severity in ("critical", "medium", "low"):
            add_finding(root, severity)
        assert api.export_report("docx")["findings"] == 3

    def test_no_findings_is_refused_rather_than_an_empty_report(self, project) -> None:
        """An empty report looks like a clean survey."""
        api, _ = project
        result = api.export_report("pdf")
        assert not result["ok"]
        assert "nothing to report" in result["error"]

    def test_an_unknown_format_names_the_ones_it_writes(self, project) -> None:
        api, root = project
        add_finding(root)
        result = api.export_report("xlsx")
        assert not result["ok"]
        assert "pdf" in result["error"] and "docx" in result["error"]

    def test_it_refuses_without_a_project(self) -> None:
        api = Api(AppSession())
        assert not api.export_report("pdf")["ok"]


class TestTheFormatListIsHonest:
    def test_it_reports_what_this_build_can_write(self) -> None:
        """Offering a format the build cannot produce is how a user discovers a missing
        dependency at the moment they are trying to send a deliverable."""
        formats = Api(AppSession()).report_formats()["formats"]
        for fmt in formats:
            assert fmt in ("pdf", "docx")

    def test_a_listed_format_actually_works(self, project) -> None:
        api, root = project
        add_finding(root)
        for fmt in api.report_formats()["formats"]:
            assert api.export_report(fmt)["ok"], f"{fmt} is offered but fails"


class TestTheDeliverableSaysWhatItCannotSay:
    def test_unmeasured_findings_are_declared(self, project) -> None:
        """Findings with no area or length cannot be totalled, and a report that quietly
        omits them presents a total that understates the site."""
        from docx import Document

        api, root = project
        add_finding(root)
        result = api.export_report("docx")
        text = " ".join(p.text for p in Document(result["path"]).paragraphs)
        assert "no measured extent" in text
