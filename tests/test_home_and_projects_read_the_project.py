"""The first screen anyone sees, and the one that describes the survey.

Home showed a Processing Queue of three jobs -- #4471 feature matching on w-02 at 34%,
#4470 dense cloud on w-01, #4468 orthomosaic on w-03 -- and a Fleet tile reading
"Aircraft 6, Available 4, Flying 1, Service 1". There were no jobs, no workers and no
fleet. The Worker column is worth naming separately: this build runs jobs on threads
inside one process, so a column headed Worker describes a deployment that does not exist
and cannot exist here.

Projects showed the bundled example's name, registered-image count, reprojection error
and geo RMSE for every project that was ever opened. Reprojection error and geo RMSE are
the two numbers a surveyor quotes to say the work is good.

And a separate defect, not a fabrication: the Project Properties inspector had a tab
labelled **Team** whose columns were Model and Measured and whose rows were neural
networks. The title and the contents had come apart. A person looking for who worked on a
survey found a list of models.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSPACES = ROOT / "app" / "web" / "js" / "workspace" / "workspaces.js"


def strip_comments(js: str) -> str:
    out, i = [], 0
    while i < len(js):
        if js.startswith("/*", i):
            end = js.find("*/", i + 2)
            i = len(js) if end == -1 else end + 2
            continue
        if js.startswith("//", i):
            end = js.find("\n", i)
            i = len(js) if end == -1 else end
            continue
        out.append(js[i])
        i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def source() -> str:
    return strip_comments(WORKSPACES.read_text(encoding="utf-8"))


FABRICATED = [
    ("#4471", "an invented job"),
    ("#4470", "an invented job"),
    ("#4468", "an invented job"),
    ("w-02", "a worker machine that does not exist"),
    ("Pune Depot 2025", "an archived project that was never created"),
    ('chip("warehouse")', "a tag belonging to no project"),
    ('chip("quarterly")', "a tag belonging to no project"),
]


@pytest.mark.parametrize("needle,why", FABRICATED, ids=[n for n, _ in FABRICATED])
def test_home_and_projects_invent_nothing(source, needle, why) -> None:
    assert needle not in source, f"{needle} is still rendered -- {why}"


class TestTheProcessingQueueIsTheRealOne:
    def test_it_reads_the_job_manager(self, source) -> None:
        block = source.split('title: "Processing Queue"')[1][:1200]
        assert 'calls: ["list_jobs"]' in block

    def test_there_is_no_worker_column(self, source) -> None:
        """Jobs run on threads in this process. A Worker column is a claim about
        deployment, and this build has no distributed one to report."""
        block = source.split('title: "Processing Queue"')[1][:1200]
        assert "Worker" not in block

    def test_a_failed_job_shows_why(self, source) -> None:
        block = source.split('title: "Processing Queue"')[1][:1200]
        assert "job.error" in block, "a failed job's only useful column is its error"

    def test_the_job_fields_exist(self, source) -> None:
        jobs = (ROOT / "app" / "jobs.py").read_text(encoding="utf-8")
        # `class Job` alone also matches `class JobCancelled`, whose body has no fields
        # at all -- so the guard passed vacuously on an exception class.
        # Splitting on the first blank line stops at the docstring, before any field.
        declared = re.split(r"\nclass Job\b(?!\w)", jobs)[1].split("\nclass ")[0]
        assert "status:" in declared, "this is not the Job dataclass"
        block = source.split('title: "Processing Queue"')[1][:1200]
        for field in set(re.findall(r"\bjob\.(\w+)", block)):
            assert f"{field}:" in declared, f"Job has no {field}"


class TestTheFleetTileCounts:
    def test_it_uses_the_call_that_returns_counts(self, source) -> None:
        """fleet_status() has always returned exactly these counts. The tile invented
        them anyway, which is the clearest case in the cockpit of a panel ignoring a
        call written for it."""
        block = source.split('id: "home.summary"')[1][:1200]
        assert 'calls: ["fleet_status"]' in block

    def test_the_fields_are_the_ones_it_returns(self, source) -> None:
        ops = (ROOT / "app" / "desktop_ops.py").read_text(encoding="utf-8")
        shape = ops.split("def fleet_status")[1].split("\ndef ")[0]
        block = source.split('id: "home.summary"')[1][:1200]
        for field in set(re.findall(r"\bf\.(\w+)", block)):
            assert f'"{field}"' in shape, f"fleet_status() does not return {field}"


class TestTheTabIsNamedForWhatItShows:
    def test_there_is_no_team_tab_listing_models(self, source) -> None:
        inspector = source.split('id: "proj.props"')[1][:3000]
        if 'title: "Team"' in inspector:
            tab = inspector.split('title: "Team"')[1][:400]
            assert "model" not in tab.lower(), (
                'the tab labelled "Team" lists models'
            )

    def test_the_models_tab_reads_the_registry(self, source) -> None:
        inspector = source.split('id: "proj.props"')[1][:3000]
        assert 'title: "Models"' in inspector
        tab = inspector.split('title: "Models"')[1][:600]
        assert '"verify_models"' in tab


class TestAccuracyIsNotInvented:
    def test_the_general_tab_does_not_quote_an_accuracy(self, source) -> None:
        """Reprojection error and geo RMSE came from the bundled example, so every
        project reported the same ones. They belong to a reconstruction and a set of
        control points; a project that has neither has no accuracy to quote."""
        tab = source.split('title: "General"')[1][:1800]
        assert "reprojection_px" not in tab
        assert "geo_rmse_m" not in tab

    def test_it_reads_the_open_project(self, source) -> None:
        tab = source.split('title: "General"')[1][:1800]
        assert '"get_project"' in tab
