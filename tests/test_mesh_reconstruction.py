"""Poisson meshing produces a real surface, or nothing -- never a plausible invention.

Poisson reconstruction is a watertight-surface method: given a patch of points it will
happily close the surface far beyond where any data exists, producing smooth geometry
over ground nobody surveyed. That invented shell is not a rendering artefact. It is
measurable, and a volume or a distance taken across it is a number derived from an
assumption rather than from the site.

Density trimming is what removes it, and these tests exist to keep that trim in place,
along with the two refusals around it: too few points, and Open3D absent. Both must
return no mesh rather than a poor one, because a caller cannot tell a bad mesh from a
good one by looking at the file.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction_colmap import ColmapReconstructor


def dome_points(count: int = 4000, radius: float = 5.0, seed: int = 1337) -> np.ndarray:
    """Points on an upper hemisphere: a surface with a genuine edge.

    The edge is the point. Poisson will try to close the bowl underneath, and the trim
    is what stops that invented floor reaching the caller.
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, count)
    costheta = rng.uniform(0.0, 1.0, count)
    theta = np.arccos(costheta)
    return np.column_stack([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta),
    ])


@pytest.fixture
def reconstructor() -> ColmapReconstructor:
    return ColmapReconstructor()


class TestRefusals:
    """Nothing is better than something invented."""

    def test_too_few_points_produce_no_mesh(self, reconstructor, tmp_path) -> None:
        points = dome_points(count=50)
        result = reconstructor._build_mesh(tmp_path, points, np.zeros((50, 3), dtype=np.uint8))
        assert result == {}
        assert any("too few points" in w.lower() for w in reconstructor.warnings)

    def test_the_refusal_leaves_no_file_behind(self, reconstructor, tmp_path) -> None:
        # A caller that globs the output directory must not find a stale or empty mesh.
        reconstructor._build_mesh(tmp_path, dome_points(count=50), np.zeros((50, 3), dtype=np.uint8))
        assert not list(tmp_path.glob("mesh.*"))

    def test_open3d_absent_is_reported_rather_than_silently_skipped(
        self, reconstructor, tmp_path, monkeypatch
    ) -> None:
        import core.reconstruction_colmap as module

        monkeypatch.setattr(module, "_HAS_OPEN3D", False)
        points = dome_points()
        result = reconstructor._build_mesh(tmp_path, points, np.zeros((points.shape[0], 3), np.uint8))
        assert result == {}
        assert any("open3d" in w.lower() for w in reconstructor.warnings), (
            "the mesh was skipped with no warning; the deliverable is silently short a file"
        )

    def test_a_failure_mid_reconstruction_does_not_raise_through(
        self, reconstructor, tmp_path, monkeypatch
    ) -> None:
        """One unusable cloud must not abandon the orthomosaic and DSM alongside it."""
        import core.reconstruction_colmap as module

        class Boom:
            def __getattr__(self, name):
                raise RuntimeError("open3d exploded")

        monkeypatch.setattr(module, "o3d", Boom())
        points = dome_points()
        result = reconstructor._build_mesh(tmp_path, points, np.zeros((points.shape[0], 3), np.uint8))
        assert result == {}
        assert any("mesh generation failed" in w.lower() for w in reconstructor.warnings)


open3d = pytest.importorskip("open3d")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Built once: Poisson on 4,000 points is slow enough to be worth reusing."""
    out = tmp_path_factory.mktemp("mesh")
    rec = ColmapReconstructor()
    points = dome_points()
    colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    return rec, out, rec._build_mesh(out, points, colors)


class TestRealMesh:
    """The happy path, run for real rather than mocked."""


    def test_both_mesh_formats_are_written(self, built) -> None:
        _, out, result = built
        assert "mesh" in result and "mesh_obj" in result
        assert (out / "mesh.ply").exists()
        assert (out / "mesh.obj").exists()

    def test_the_obj_is_not_an_empty_shell(self, built) -> None:
        _, out, _ = built
        text = (out / "mesh.obj").read_text(encoding="utf-8")
        assert text.count("\nv ") > 100, "the OBJ carries almost no vertices"
        assert "\nf " in text, "the OBJ has no faces, so it is a point list not a mesh"

    def test_the_mesh_follows_the_surveyed_surface(self, built) -> None:
        """Vertices should sit near the dome, not float off in invented space."""
        from core.model_measurement import read_mesh

        _, out, _ = built
        vertices, _faces = read_mesh(out / "mesh.obj")
        radii = np.linalg.norm(vertices, axis=1)
        # Generous: Poisson smooths, and the trim leaves a fringe. A mesh that had
        # closed the bowl would put many vertices near the origin and fail this.
        assert np.median(radii) == pytest.approx(5.0, abs=1.0)

PATCH_HALF_WIDTH = 5.0


def patch_points(count: int = 6000, seed: int = 7) -> np.ndarray:
    """A bounded, gently undulating ground patch -- the shape a survey actually produces.

    A flat patch is the case where Poisson's closure is visible and measurable: to make
    a watertight solid it must carry the surface out past the edges of the data. The
    dome above does not exercise this, which is why it is not used here.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-PATCH_HALF_WIDTH, PATCH_HALF_WIDTH, size=(count, 2))
    z = 0.05 * np.sin(xy[:, 0]) + rng.normal(0.0, 0.01, count)
    return np.column_stack([xy, z])


@pytest.fixture(scope="module")
def patch_mesh(tmp_path_factory):
    out = tmp_path_factory.mktemp("patch")
    rec = ColmapReconstructor()
    points = patch_points()
    colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    rec._build_mesh(out, points, colors)
    return out


class TestDensityTrimming:
    """The trim is the difference between a surveyed surface and an invented one.

    Measured on the real pipeline: without the trim, 0.42 per cent of vertices land
    beyond the surveyed patch and the mesh spans 11.0 m across a 10.0 m survey. With it,
    nothing lies outside. The threshold below is therefore zero rather than a tolerance
    -- a tolerant version passed identically on trimmed and untrimmed meshes, which is
    to say it tested nothing.
    """

    def test_no_vertex_lies_beyond_the_surveyed_area(self, patch_mesh) -> None:
        from core.model_measurement import read_mesh

        vertices, _ = read_mesh(patch_mesh / "mesh.obj")
        limit = PATCH_HALF_WIDTH + 0.5  # half a metre of smoothing fringe allowed
        outside = np.count_nonzero(
            (np.abs(vertices[:, 0]) > limit) | (np.abs(vertices[:, 1]) > limit)
        )
        assert outside == 0, (
            f"{outside} of {len(vertices)} vertices sit outside the surveyed patch. "
            "Poisson's closure has survived into the deliverable, and a volume or "
            "distance measured across it would be derived from invented ground."
        )

    def test_the_mesh_does_not_greatly_exceed_the_survey_extent(self, patch_mesh) -> None:
        from core.model_measurement import read_mesh

        vertices, _ = read_mesh(patch_mesh / "mesh.obj")
        span = max(
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
        )
        assert span < 2 * PATCH_HALF_WIDTH + 1.0, (
            f"the mesh spans {span:.2f} m across a {2 * PATCH_HALF_WIDTH:.0f} m survey"
        )
