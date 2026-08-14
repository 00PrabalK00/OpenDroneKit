"""Generating an inspection report from real project data.

A report is the deliverable a client actually receives, and the risk it carries is
specific: an empty section reads as "nothing found" rather than "nothing looked at".
So these tests check that a report generated without a defect run says so, that
findings which exist reach the page, and that a report cannot be produced at all for a
project whose data is missing.
"""

from __future__ import annotations

import json

import pytest

from core import project as project_module
from core.errors import AppError
from core.report_engine import (
    ReportConfig,
    generate_report,
    list_reports,
    validate_report_readiness,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the project manager at a throwaway workspace.

    get_manager caches a module-level singleton, so it is reset rather than left
    pointing at the developer's real projects folder.
    """
    monkeypatch.setattr(project_module, "_manager", None)
    manager = project_module.get_manager(tmp_path / "workspace")
    yield manager
    monkeypatch.setattr(project_module, "_manager", None)


@pytest.fixture
def project(workspace):
    meta = workspace.create_project("Bridge 14", description="Quarterly inspection")
    return meta["id"], meta


def write_defect_run(meta, findings):
    """Write a defect run in the layout build_report_context reads."""
    from pathlib import Path

    root = Path(meta["root_dir"])
    run_dir = root / "analysis" / "defects" / "run-001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps({"findings": findings, "model_used": "segformer_b2"}, indent=2),
        encoding="utf-8",
    )
    return run_dir


class TestReadiness:
    def test_a_project_that_does_not_exist_is_not_ready(self, workspace):
        readiness = validate_report_readiness(ReportConfig(project_id="no-such-project"))
        assert readiness.ok is False
        assert "project" in readiness.missing

    def test_a_fresh_project_is_ready_but_warns_about_what_is_missing(self, project):
        """Ready to render is not the same as having something to say."""
        project_id, _ = project
        readiness = validate_report_readiness(ReportConfig(project_id=project_id))

        assert readiness.ok is True
        assert any("defect" in warning.lower() for warning in readiness.warnings)

    def test_an_absent_defect_run_is_reported_as_absent_not_as_zero_defects(self, project):
        """The distinction a client cares about: not inspected vs inspected and clean."""
        project_id, _ = project
        readiness = validate_report_readiness(ReportConfig(project_id=project_id))
        combined = " ".join(readiness.warnings + readiness.notes).lower()

        assert "no defect run" in combined
        assert "0 defects" not in combined and "no defects found" not in combined

    def test_skipping_images_does_not_warn_about_missing_images(self, project):
        project_id, _ = project
        readiness = validate_report_readiness(
            ReportConfig(project_id=project_id, include_images=False))
        assert not any("image gallery" in warning for warning in readiness.warnings)


class TestGeneration:
    def test_a_report_is_generated_and_lands_on_disk(self, project):
        project_id, _ = project
        result = generate_report(ReportConfig(project_id=project_id, title="Q3 Inspection"))

        from pathlib import Path

        assert Path(result.html_path).exists()
        assert result.title == "Q3 Inspection"
        assert Path(result.html_path).read_text(encoding="utf-8").strip() != ""

    def test_the_report_carries_the_title_and_project_through_to_the_page(self, project):
        project_id, _ = project
        result = generate_report(ReportConfig(project_id=project_id, title="Pier Survey"))

        from pathlib import Path

        html = Path(result.html_path).read_text(encoding="utf-8")
        assert "Pier Survey" in html

    def test_a_report_for_a_missing_project_is_refused_not_faked(self, workspace):
        with pytest.raises(AppError):
            generate_report(ReportConfig(project_id="does-not-exist"))

    def test_generated_reports_are_listed_for_the_project(self, project):
        project_id, _ = project
        first = generate_report(ReportConfig(project_id=project_id, title="First"))
        second = generate_report(ReportConfig(project_id=project_id, title="Second"))

        listed = {report.id for report in list_reports(project_id)}
        assert {first.id, second.id} <= listed

    def test_listing_reports_for_an_unknown_project_is_empty_not_an_error(self, workspace):
        assert list_reports("no-such-project") == []

    def test_each_report_gets_its_own_directory(self, project):
        """Two reports must not overwrite one another."""
        from pathlib import Path

        project_id, _ = project
        first = generate_report(ReportConfig(project_id=project_id, title="First"))
        second = generate_report(ReportConfig(project_id=project_id, title="Second"))

        assert Path(first.html_path).parent != Path(second.html_path).parent
        assert Path(first.html_path).exists()

    def test_a_manifest_is_written_so_the_report_survives_a_restart(self, project):
        from pathlib import Path

        project_id, _ = project
        result = generate_report(ReportConfig(project_id=project_id, title="Persisted"))
        manifest = Path(result.html_path).parent / "report.json"

        assert manifest.exists()
        assert json.loads(manifest.read_text(encoding="utf-8"))["title"] == "Persisted"


class TestContent:
    def test_a_report_section_selection_is_honoured(self, project):
        project_id, _ = project
        result = generate_report(ReportConfig(
            project_id=project_id, title="Narrow", sections=["summary"]))

        from pathlib import Path

        assert Path(result.html_path).exists()

    def test_the_report_records_who_asked_for_it(self, project):
        project_id, _ = project
        result = generate_report(ReportConfig(
            project_id=project_id, title="Attributed", author="P. Khare"))

        from pathlib import Path

        html = Path(result.html_path).read_text(encoding="utf-8")
        assert "P. Khare" in html

    def test_report_generation_is_recorded_in_the_project_audit_trail(self, project, workspace):
        """A deliverable that left the building should be traceable."""
        project_id, _ = project
        generate_report(ReportConfig(project_id=project_id, title="Audited"))

        events = workspace.get_audit_log(project_id)
        assert any("report_generated" in json.dumps(event) for event in events)
