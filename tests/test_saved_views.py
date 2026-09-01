"""How a shared model opens for the person it was sent to.

A reconstruction opens wherever the viewer's camera happened to start -- usually above
the site looking down, every clip off, every annotation showing. The recipient was sent
one thing to look at and the first thing they have to do is find it.

The tests that matter are the ones about a view that has quietly stopped meaning what it
meant when it was saved: a clip deleted underneath it, two views both claiming to be the
default, or a camera sitting on its own target.
"""

from __future__ import annotations

import json

import pytest

from core.saved_views import SavedView, ViewRefused, ViewStore, resolve_clips, validate


def a_view(name: str = "as delivered", **kwargs) -> SavedView:
    base = dict(
        name=name,
        position=(40.0, -30.0, 25.0),
        target=(0.0, 0.0, 5.0),
        fov_deg=50.0,
    )
    base.update(kwargs)
    return SavedView(**base)  # type: ignore[arg-type]


class TestAViewMustOpenSomewhere:
    def test_a_camera_on_its_target_is_refused(self) -> None:
        """A viewer given a zero-length direction either shows a blank screen or
        silently substitutes a default -- and the second is worse, because the recipient
        sees A view and assumes it is THE view."""
        with pytest.raises(ViewRefused, match="no direction"):
            validate(a_view(position=(1.0, 2.0, 3.0), target=(1.0, 2.0, 3.0)))

    def test_a_view_needs_a_name(self) -> None:
        with pytest.raises(ViewRefused):
            validate(a_view(name="   "))

    @pytest.mark.parametrize("fov", [0.0, -10.0, 180.0, 400.0])
    def test_an_unrenderable_field_of_view_is_refused(self, fov) -> None:
        with pytest.raises(ViewRefused):
            validate(a_view(fov_deg=fov))

    def test_a_normal_view_passes(self) -> None:
        validate(a_view())

    def test_the_distance_is_the_camera_to_target_separation(self) -> None:
        view = a_view(position=(3.0, 4.0, 0.0), target=(0.0, 0.0, 0.0))
        assert view.distance_m() == pytest.approx(5.0)


class TestClipsCanDisappearUnderneathAView:
    def test_a_deleted_clip_is_reported_rather_than_ignored(self) -> None:
        """Two wrong answers here: refuse to open the view, or open it silently showing
        more of the model than was intended. Naming what could not be applied lets the
        caller do neither."""
        view = a_view(visible_clips=["clean building", "north facade"])
        applied, missing = resolve_clips(view, ["clean building"])
        assert applied == ["clean building"]
        assert missing == ["north facade"]

    def test_all_present_means_nothing_missing(self) -> None:
        view = a_view(visible_clips=["a", "b"])
        applied, missing = resolve_clips(view, ["a", "b", "c"])
        assert applied == ["a", "b"]
        assert missing == []

    def test_a_view_with_no_clips_applies_none(self) -> None:
        applied, missing = resolve_clips(a_view(), ["a"])
        assert applied == [] and missing == []


class TestSavedViewsPersist:
    def test_a_view_round_trips(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view(visible_clips=["clean building"], facade_mode=True,
                         show_annotations=False))
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].name == "as delivered"
        assert loaded[0].position == pytest.approx((40.0, -30.0, 25.0))
        assert loaded[0].facade_mode is True
        assert loaded[0].show_annotations is False
        assert loaded[0].visible_clips == ["clean building"]

    def test_several_views_coexist(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view("as delivered"))
        store.add(a_view("north facade"))
        store.add(a_view("roof detail"))
        assert {v.name for v in store.load()} == {"as delivered", "north facade", "roof detail"}

    def test_the_same_name_replaces_rather_than_duplicates(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view("as delivered", fov_deg=50.0))
        store.add(a_view("as delivered", fov_deg=35.0))
        views = store.load()
        assert len(views) == 1
        assert views[0].fov_deg == pytest.approx(35.0)

    def test_an_invalid_view_never_reaches_disk(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        with pytest.raises(ViewRefused):
            store.add(a_view(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 0.0)))
        assert store.load() == []

    def test_removing_one_that_does_not_exist_says_so(self, tmp_path) -> None:
        with pytest.raises(ViewRefused):
            ViewStore(tmp_path).remove("never saved")

    def test_a_corrupt_file_reads_as_no_views(self, tmp_path) -> None:
        """Losing the views is a presentation setback; failing to open the model because
        of them would be worse."""
        store = ViewStore(tmp_path)
        store.path.write_text("{ not json", encoding="utf-8")
        assert store.load() == []

    def test_the_file_is_readable_by_a_person(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view("as delivered"))
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        assert raw["views"][0]["name"] == "as delivered"


class TestExactlyOneDefault:
    def test_marking_a_new_default_clears_the_old_one(self, tmp_path) -> None:
        """Two defaults is an unanswerable question at open time, and the viewer would
        resolve it by file order -- which is to say, arbitrarily."""
        store = ViewStore(tmp_path)
        store.add(a_view("first", is_default=True))
        store.add(a_view("second", is_default=True))
        defaults = [v.name for v in store.load() if v.is_default]
        assert defaults == ["second"]

    def test_set_default_promotes_an_existing_view(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view("first", is_default=True))
        store.add(a_view("second"))
        store.set_default("second")
        assert [v.name for v in store.load() if v.is_default] == ["second"]

    def test_set_default_on_a_missing_view_is_refused(self, tmp_path) -> None:
        with pytest.raises(ViewRefused):
            ViewStore(tmp_path).set_default("nope")

    def test_the_default_is_what_a_share_link_opens_at(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view("as delivered", is_default=True))
        store.add(a_view("working view"))
        assert store.default().name == "as delivered"

    def test_no_default_leaves_the_viewer_alone(self, tmp_path) -> None:
        """Not every project wants to dictate an opening camera."""
        store = ViewStore(tmp_path)
        store.add(a_view("working view"))
        assert store.default() is None

    def test_removing_the_default_leaves_no_default(self, tmp_path) -> None:
        store = ViewStore(tmp_path)
        store.add(a_view("as delivered", is_default=True))
        store.remove("as delivered")
        assert store.default() is None
