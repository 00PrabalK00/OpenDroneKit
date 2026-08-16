"""Point clouds can be planned from, and a truncated one cannot.

The planner used to refuse LAS outright, and the stated reason was right about the
physics and wrong about the conclusion: points carry no facing, so a stand-off plan
built straight from them would invent the orientation of every surface. But orientation
can be recovered from the data rather than assumed, which is what meshing does, so the
refusal was solving the problem by declining it.

LAS is read directly rather than through a package, because a survey should not need a
package index to open a cloud already sitting on its disk. The format is a fixed header
plus fixed-length records whose first twelve bytes are the scaled integer coordinates.

The tests build LAS files byte by byte. That is deliberate: a reader tested only against
files written by the same library it was written against will agree with itself about a
format it has misunderstood.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mission.geometry_3d import (
    NoUsableSurface,
    UnsupportedSurface,
    read_las_points,
    read_surface,
)


def write_las(
    path,
    points: np.ndarray,
    *,
    version: tuple[int, int] = (1, 2),
    record_format: int = 0,
    record_length: int = 20,
    scale: float = 0.01,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    declared_count: int | None = None,
    signature: bytes = b"LASF",
    compressed: bool = False,
    truncate_records: int = 0,
) -> str:
    """A minimal but specification-shaped LAS file."""
    header_size = 227 if version < (1, 4) else 375
    point_offset = header_size
    count = len(points)

    header = bytearray(header_size)
    header[0:4] = signature
    header[24] = version[0]
    header[25] = version[1]
    struct.pack_into("<H", header, 94, header_size)
    struct.pack_into("<I", header, 96, point_offset)
    header[104] = record_format | (0b1100_0000 if compressed else 0)
    struct.pack_into("<H", header, 105, record_length)
    reported = count if declared_count is None else declared_count
    struct.pack_into("<I", header, 107, 0 if version >= (1, 4) else reported)
    struct.pack_into("<3d", header, 131, scale, scale, scale)
    struct.pack_into("<3d", header, 155, *offset)
    if version >= (1, 4):
        struct.pack_into("<Q", header, 247, reported)

    body = bytearray()
    for point in points:
        record = bytearray(record_length)
        struct.pack_into(
            "<3i", record, 0,
            int(round((point[0] - offset[0]) / scale)),
            int(round((point[1] - offset[1]) / scale)),
            int(round((point[2] - offset[2]) / scale)),
        )
        body += record
    if truncate_records:
        body = body[: len(body) - truncate_records * record_length]

    path.write_bytes(bytes(header) + bytes(body))
    return str(path)


def grid_points(n: int = 40, span: float = 20.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-span, span, n), np.linspace(-span, span, n))
    zs = 0.4 * np.sin(xs / 4.0)
    return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])


class TestReadingTheFormat:
    def test_coordinates_round_trip_through_scale_and_offset(self, tmp_path) -> None:
        points = np.array([[1.23, 4.56, 7.89], [-10.0, 0.5, 2.25]])
        path = write_las(tmp_path / "a.las", points, offset=(100.0, 200.0, 300.0))
        loaded = read_las_points(path)
        assert np.allclose(loaded, points, atol=0.01)

    def test_a_las_1_4_count_is_read_from_the_64_bit_field(self, tmp_path) -> None:
        """1.4 moved the count; a reader stuck on the legacy field sees zero points."""
        points = grid_points(n=12)
        path = write_las(tmp_path / "b.las", points, version=(1, 4), record_format=6,
                         record_length=30)
        assert len(read_las_points(path)) == len(points)

    def test_extra_record_fields_are_skipped_not_misread(self, tmp_path) -> None:
        # Intensity, returns and classification follow the coordinates. A reader that
        # ignored the record length would walk into them and produce nonsense.
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        path = write_las(tmp_path / "c.las", points, record_format=3, record_length=34)
        assert np.allclose(read_las_points(path), points, atol=0.01)

    def test_a_missing_count_falls_back_to_the_file_length(self, tmp_path) -> None:
        points = grid_points(n=10)
        path = write_las(tmp_path / "d.las", points, declared_count=0)
        assert len(read_las_points(path)) == len(points)


class TestRefusals:
    def test_a_file_without_the_signature_is_refused(self, tmp_path) -> None:
        path = write_las(tmp_path / "e.las", grid_points(n=10), signature=b"NOPE")
        with pytest.raises(UnsupportedSurface, match="LASF"):
            read_las_points(path)

    def test_a_compressed_payload_is_refused_rather_than_misparsed(self, tmp_path) -> None:
        """The compression bits live in the format byte; ignoring them reads noise."""
        path = write_las(tmp_path / "f.las", grid_points(n=10), compressed=True)
        with pytest.raises(UnsupportedSurface, match="LAZ"):
            read_las_points(path)

    def test_a_truncated_file_is_refused(self, tmp_path) -> None:
        """A partial cloud would plan a partial structure, which is worse than failing."""
        points = grid_points(n=10)
        path = write_las(tmp_path / "g.las", points, truncate_records=20)
        with pytest.raises(NoUsableSurface, match="truncated"):
            read_las_points(path)

    def test_an_empty_cloud_is_refused(self, tmp_path) -> None:
        path = write_las(tmp_path / "h.las", np.zeros((0, 3)))
        with pytest.raises(NoUsableSurface):
            read_las_points(path)

    def test_a_cloud_too_large_to_load_says_so(self, tmp_path) -> None:
        points = grid_points(n=20)
        path = write_las(tmp_path / "i.las", points)
        with pytest.raises(NoUsableSurface, match="Decimate"):
            read_las_points(path, max_points=10)

    def test_laz_by_extension_explains_what_to_do(self, tmp_path) -> None:
        target = tmp_path / "j.laz"
        target.write_bytes(b"LASF" + bytes(300))
        with pytest.raises(UnsupportedSurface, match="uncompressed LAS"):
            read_surface(target)

    def test_ifc_is_refused_as_a_modelling_decision_not_a_conversion(self, tmp_path) -> None:
        target = tmp_path / "k.ifc"
        target.write_text("ISO-10303-21;", encoding="utf-8")
        with pytest.raises(UnsupportedSurface, match="modelling decision"):
            read_surface(target)


pytest.importorskip("open3d")


class TestPlanningFromACloud:
    def test_a_las_file_becomes_a_surface_with_faces(self, tmp_path) -> None:
        """The whole point: orientation recovered from the data, not assumed."""
        path = write_las(tmp_path / "surface.las", grid_points(n=45))
        vertices, faces = read_surface(path)
        assert len(vertices) > 100
        assert len(faces) > 100

    def test_the_recovered_surface_follows_the_points(self, tmp_path) -> None:
        points = grid_points(n=45)
        path = write_las(tmp_path / "surface2.las", points)
        vertices, _ = read_surface(path)
        # The cloud spans z in roughly [-0.4, 0.4]; a surface that had drifted off the
        # data would sit somewhere else entirely.
        assert abs(float(np.median(vertices[:, 2]))) < 1.0

    def test_too_few_points_are_refused_rather_than_meshed(self, tmp_path) -> None:
        path = write_las(tmp_path / "sparse.las", grid_points(n=5))
        with pytest.raises(NoUsableSurface, match="too few"):
            read_surface(path)
