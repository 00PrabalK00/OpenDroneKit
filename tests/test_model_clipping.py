"""Cutting a reconstruction down to the asset that was surveyed.

A photogrammetric model arrives with everything the camera could see: the neighbour's
roof, the car park, the boundary trees. Handing a client the raw model makes them find
the building, and makes every measurement ambiguous about which roof was measured.

The tests that matter here are about what a clip must NOT do: lose data, invent geometry
at the cut, or widen the selection when the operator adds detail to it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.model_clipping import (
    Clip,
    ClipPlane,
    ClipRefused,
    ClipStore,
    clip_mesh,
    combined_mask,
    points_in_front_of_plane,
    points_inside_polygon,
)

# A building at the origin, a neighbour 50 m east, and one point above ridge level.
POINTS = np.array([
    [0.0, 0.0, 5.0],
    [2.0, 2.0, 8.0],
    [-2.0, 1.0, 6.0],
    [50.0, 0.0, 5.0],
    [52.0, 2.0, 7.0],
    [0.0, 0.0, 25.0],
])
SITE = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]


class TestThePolygonCut:
    def test_it_removes_the_neighbour(self) -> None:
        mask = points_inside_polygon(POINTS, SITE)
        assert mask.tolist() == [True, True, True, False, False, True]

    def test_a_corner_of_the_drawn_footprint_stays_in_it(self) -> None:
        """A vertex falling out of its own selection is the kind of edge case that makes
        an operator redraw the boundary larger and wonder why."""
        corners = np.array([[c[0], c[1], 0.0] for c in SITE])
        assert points_inside_polygon(corners, SITE).any()

    def test_the_ring_need_not_be_closed_by_the_caller(self) -> None:
        opened = points_inside_polygon(POINTS, SITE)
        closed = points_inside_polygon(POINTS, SITE + [SITE[0]])
        assert opened.tolist() == closed.tolist()

    def test_a_degenerate_polygon_is_refused(self) -> None:
        with pytest.raises(ClipRefused):
            points_inside_polygon(POINTS, [[0.0, 0.0], [1.0, 1.0]])

    def test_a_concave_footprint_is_handled(self) -> None:
        """An L-shaped building is the normal case, not an exotic one."""
        ell = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]
        inside = np.array([[1.0, 1.0, 0.0], [8.0, 1.0, 0.0], [1.0, 8.0, 0.0]])
        notch = np.array([[8.0, 8.0, 0.0]])
        assert points_inside_polygon(inside, ell).all()
        assert not points_inside_polygon(notch, ell).any()


class TestThePlaneCut:
    def test_pitch_ninety_cuts_horizontally(self) -> None:
        plane = ClipPlane.from_orientation([0, 0, 10], heading_deg=0, pitch_deg=90)
        assert plane.normal == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
        mask = points_in_front_of_plane(POINTS, plane)
        assert mask[-1] == np.False_, "the point above the plane should be cut"
        assert mask[:5].all()

    def test_heading_rotates_the_cut(self) -> None:
        east = ClipPlane.from_orientation([0, 0, 0], heading_deg=0)
        north = ClipPlane.from_orientation([0, 0, 0], heading_deg=90)
        assert east.normal == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
        assert north.normal == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)

    def test_the_normal_is_always_unit_length(self) -> None:
        for heading in (0, 37, 180, 359):
            for pitch in (-80, 0, 45, 90):
                plane = ClipPlane.from_orientation([1, 2, 3], heading, pitch)
                assert np.linalg.norm(plane.normal) == pytest.approx(1.0, abs=1e-9)

    def test_a_plane_needs_a_point_in_the_model(self) -> None:
        with pytest.raises(ClipRefused):
            ClipPlane.from_orientation([0, 0], heading_deg=0)


class TestClipsCombine:
    def test_clips_narrow_rather_than_widen(self) -> None:
        """Intersection, not union.

        A polygon around the building plus a plane at ridge level means "this building,
        below the ridge". Under union the second clip would ADD the neighbour's car park
        back, so adding detail would show more rather than less -- the opposite of what
        an operator drawing a second boundary is asking for.
        """
        site = Clip(name="building", kind="polygon", polygon_xy=SITE)
        ridge = Clip(name="below ridge", kind="plane",
                     plane=ClipPlane.from_orientation([0, 0, 10], 0, 90))
        assert combined_mask(POINTS, [site, ridge]).tolist() == [
            True, True, True, False, False, False]

    def test_hiding_a_clip_restores_what_it_removed(self) -> None:
        """This is what makes a clip a view rather than an edit."""
        site = Clip(name="building", kind="polygon", polygon_xy=SITE)
        ridge = Clip(name="below ridge", kind="plane",
                     plane=ClipPlane.from_orientation([0, 0, 10], 0, 90), visible=False)
        assert combined_mask(POINTS, [site, ridge])[-1] == np.True_

    def test_no_clips_keeps_everything(self) -> None:
        assert combined_mask(POINTS, []).all()

    def test_a_plane_clip_with_no_plane_is_refused(self) -> None:
        with pytest.raises(ClipRefused):
            combined_mask(POINTS, [Clip(name="broken", kind="plane")])


class TestMeshClipping:
    def test_a_triangle_is_kept_only_when_every_corner_survives(self) -> None:
        """Partly-inside triangles would hang outside the drawn boundary; splitting them
        would invent vertices that were never measured. The cut is on whole triangles so
        the edge is where the data actually stops."""
        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # inside
            [50.0, 0.0, 0.0],                                     # outside
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        kept_v, kept_f = clip_mesh(verts, faces, [Clip("site", "polygon", polygon_xy=SITE)])
        assert len(kept_f) == 1
        assert len(kept_v) == 3

    def test_surviving_faces_still_index_their_vertices(self) -> None:
        """Reindexing is where this goes wrong silently: an off-by-one leaves a mesh that
        loads and renders as garbage rather than failing."""
        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [50.0, 0.0, 0.0], [51.0, 0.0, 0.0], [50.0, 1.0, 0.0],
        ])
        faces = np.array([[3, 4, 5], [0, 1, 2]])
        kept_v, kept_f = clip_mesh(verts, faces, [Clip("site", "polygon", polygon_xy=SITE)])
        assert kept_f.max() < len(kept_v)
        # The surviving triangle must still be the one at the origin.
        assert kept_v[kept_f[0]].mean(axis=0)[0] == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_an_empty_mesh_does_not_raise(self) -> None:
        v, f = clip_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), [])
        assert len(v) == 0 and len(f) == 0


class TestSavedClips:
    def test_a_clip_round_trips_through_disk(self, tmp_path) -> None:
        store = ClipStore(tmp_path)
        plane = ClipPlane.from_orientation([1, 2, 3], heading_deg=45, pitch_deg=20)
        store.add(Clip(name="north facade", kind="plane", plane=plane))
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].name == "north facade"
        assert loaded[0].plane.normal == pytest.approx(plane.normal, abs=1e-12)

    def test_several_clips_coexist(self, tmp_path) -> None:
        """The same model carries "clean building", "north facade" and a section at once;
        that is the whole reason clips are named."""
        store = ClipStore(tmp_path)
        store.add(Clip(name="clean building", kind="polygon", polygon_xy=SITE))
        store.add(Clip(name="north facade", kind="plane",
                       plane=ClipPlane.from_orientation([0, 0, 0], 90)))
        assert {c.name for c in store.load()} == {"clean building", "north facade"}

    def test_adding_the_same_name_replaces_rather_than_duplicates(self, tmp_path) -> None:
        store = ClipStore(tmp_path)
        store.add(Clip(name="site", kind="polygon", polygon_xy=SITE))
        store.add(Clip(name="site", kind="polygon", polygon_xy=[[0, 0], [1, 0], [1, 1]]))
        clips = store.load()
        assert len(clips) == 1
        assert len(clips[0].polygon_xy) == 3

    def test_a_clip_needs_a_name(self, tmp_path) -> None:
        with pytest.raises(ClipRefused):
            ClipStore(tmp_path).add(Clip(name="  ", kind="polygon", polygon_xy=SITE))

    def test_removing_one_leaves_the_others(self, tmp_path) -> None:
        store = ClipStore(tmp_path)
        store.add(Clip(name="a", kind="polygon", polygon_xy=SITE))
        store.add(Clip(name="b", kind="polygon", polygon_xy=SITE))
        assert {c.name for c in store.remove("a")} == {"b"}

    def test_removing_one_that_does_not_exist_says_so(self, tmp_path) -> None:
        with pytest.raises(ClipRefused):
            ClipStore(tmp_path).remove("never existed")

    def test_visibility_is_persisted(self, tmp_path) -> None:
        store = ClipStore(tmp_path)
        store.add(Clip(name="section", kind="polygon", polygon_xy=SITE))
        store.set_visible("section", False)
        assert store.load()[0].visible is False

    def test_the_file_is_readable_by_a_person(self, tmp_path) -> None:
        """An operator should be able to open it and see what a deliverable was cut to."""
        store = ClipStore(tmp_path)
        store.add(Clip(name="clean building", kind="polygon", polygon_xy=SITE))
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        assert raw["clips"][0]["name"] == "clean building"

    def test_a_corrupt_file_reads_as_no_clips_rather_than_crashing(self, tmp_path) -> None:
        """Losing the clips is a presentation setback. Failing to open the model because
        of them would be data loss by another route."""
        store = ClipStore(tmp_path)
        store.path.write_text("{not json", encoding="utf-8")
        assert store.load() == []
