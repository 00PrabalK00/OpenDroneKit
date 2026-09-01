"""Telling the operator when something they walked away from has finished.

A reconstruction is eight minutes on this machine and longer on a real survey. Nobody
watches that, so they switch workspace and the result lands in a panel they are not
looking at. They come back, see a finished job, and cannot tell whether it finished two
seconds ago or forty minutes ago -- so they re-run it, or they act on stale output.

The tests that matter are about not losing one: a job finishing on a worker thread while
the shell reads the list, and a listener that raises.
"""

from __future__ import annotations

import json
import threading

import pytest

from core.notifications import (
    MAX_NOTIFICATIONS,
    NotificationCentre,
    describe_job,
)


@pytest.fixture
def centre(tmp_path) -> NotificationCentre:
    return NotificationCentre(tmp_path)


class TestRecordingSomethingWorthKnowing:
    def test_a_notification_persists(self, centre) -> None:
        centre.notify("Reconstruction finished", "77 of 77 frames", level="success",
                      subject_kind="job", subject_id="abc123")
        loaded = centre.load()
        assert len(loaded) == 1
        assert loaded[0].title == "Reconstruction finished"
        assert loaded[0].subject_id == "abc123"

    def test_it_carries_what_it_is_about(self, centre) -> None:
        """Without a subject a notification is a dead end: the operator is told something
        happened and left to go and find it."""
        note = centre.notify("Export written", subject_kind="file", subject_id="/tmp/r.pdf")
        assert note.subject_kind == "file"
        assert note.subject_id == "/tmp/r.pdf"

    def test_an_unknown_level_falls_back_rather_than_raising(self, centre) -> None:
        """A bad level should not lose the notification -- the message matters more than
        the colour it is drawn in."""
        assert centre.notify("Something", level="catastrophic").level == "info"

    def test_newest_first(self, centre) -> None:
        for i in range(3):
            centre.notify(f"job {i}")
        assert [n.title for n in centre.load()][0] == "job 2"

    def test_the_file_is_readable_by_a_person(self, centre) -> None:
        centre.notify("Reconstruction finished")
        raw = json.loads(centre.path.read_text(encoding="utf-8"))
        assert raw["notifications"][0]["title"] == "Reconstruction finished"


class TestUnread:
    def test_new_notifications_are_unread(self, centre) -> None:
        centre.notify("one")
        centre.notify("two")
        assert centre.unread_count() == 2

    def test_marking_one_read_leaves_the_others(self, centre) -> None:
        first = centre.notify("one")
        centre.notify("two")
        assert centre.mark_read(first.id) is True
        assert centre.unread_count() == 1

    def test_marking_an_unknown_id_reports_it(self, centre) -> None:
        assert centre.mark_read("not-a-real-id") is False

    def test_mark_all_read_reports_how_many_changed(self, centre) -> None:
        centre.notify("one")
        centre.notify("two")
        assert centre.mark_all_read() == 2
        assert centre.mark_all_read() == 0
        assert centre.unread_count() == 0

    def test_clearing_reports_how_many_went(self, centre) -> None:
        centre.notify("one")
        centre.notify("two")
        assert centre.clear() == 2
        assert centre.load() == []


class TestItDoesNotLoseOne:
    def test_concurrent_writers_all_land(self, centre) -> None:
        """Jobs finish on worker threads while the shell reads the list. Without a lock a
        reconstruction finishing mid-render drops the notification it just wrote."""
        def write(index: int) -> None:
            centre.notify(f"job {index}")

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(centre.load()) == 20

    def test_a_listener_that_raises_does_not_lose_the_notification(self, centre) -> None:
        """It was already written before any listener ran. A badge that throws must not
        take the record with it."""
        centre.subscribe(lambda note: (_ for _ in ()).throw(RuntimeError("badge broke")))
        heard: list[str] = []
        centre.subscribe(lambda note: heard.append(note.title))

        centre.notify("Reconstruction finished")
        assert len(centre.load()) == 1
        assert heard == ["Reconstruction finished"], "a raising listener blocked the others"

    def test_a_corrupt_file_reads_as_none_rather_than_crashing(self, centre) -> None:
        centre.path.parent.mkdir(parents=True, exist_ok=True)
        centre.path.write_text("{ not json", encoding="utf-8")
        assert centre.load() == []

    def test_the_log_is_trimmed_oldest_first(self, centre) -> None:
        """A notification log is a convenience; core/audit is the record that must not
        lose entries."""
        for i in range(MAX_NOTIFICATIONS + 10):
            centre.notify(f"job {i}")
        loaded = centre.load()
        assert len(loaded) == MAX_NOTIFICATIONS
        assert loaded[0].title == f"job {MAX_NOTIFICATIONS + 9}"


class TestDescribingAFinishedJob:
    def test_a_finished_job_reads_as_success(self) -> None:
        level, title, _ = describe_job({"name": "Reconstruction", "status": "done"})
        assert level == "success"
        assert "finished" in title

    def test_a_failure_carries_the_reason(self) -> None:
        """"Reconstruction failed" alone sends the operator to go and find out why, which
        is the thing the notification was supposed to save them."""
        level, title, detail = describe_job({
            "name": "Reconstruction", "status": "failed",
            "error": "CUDA error: the provided PTX was compiled with an unsupported toolchain",
        })
        assert level == "error"
        assert "PTX" in detail

    def test_a_cancelled_job_is_a_warning_not_an_error(self) -> None:
        """Someone pressed Cancel. Reporting it in red alongside real failures teaches
        people to ignore red."""
        level, _, _ = describe_job({"name": "Reconstruction", "status": "cancelled"})
        assert level == "warning"

    def test_an_unnamed_job_still_reads(self) -> None:
        level, title, _ = describe_job({"status": "done"})
        assert title.strip()
