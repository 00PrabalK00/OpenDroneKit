"""Round-trip the exported geometry through format readers, not file-size checks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from core.model_formats import ModelGeometry, export_model


VERTICES = np.array([
    [500000.125, 2000000.250, 101.5],
    [500004.125, 2000000.250, 101.5],
    [500004.125, 2000003.250, 102.0],
    [500000.125, 2000003.250, 102.0],
], dtype=np.float64)
FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [240, 220, 20]], dtype=np.uint8)


@pytest.fixture
def geometry():
    return ModelGeometry(VERTICES, FACES, COLORS, crs_epsg=32643)


def _read_gltf_payload(path: Path):
    if path.suffix == ".gltf":
        document = json.loads(path.read_text(encoding="utf-8"))
        binary = base64.b64decode(document["buffers"][0]["uri"].split(",", 1)[1])
        return document, binary
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF" and version == 2 and total == len(raw)
    json_length, json_type = struct.unpack_from("<I4s", raw, 12)
    assert json_type == b"JSON"
    document = json.loads(raw[20:20 + json_length].decode("utf-8"))
    offset = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<I4s", raw, offset)
    assert binary_type == b"BIN\x00"
    return document, raw[offset + 8:offset + 8 + binary_length]


def _gltf_geometry(path: Path):
    document, binary = _read_gltf_payload(path)
    primitive = document["meshes"][0]["primitives"][0]
    position_accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    position_view = document["bufferViews"][position_accessor["bufferView"]]
    start = position_view.get("byteOffset", 0)
    positions = np.frombuffer(
        binary[start:start + position_view["byteLength"]], dtype="<f4"
    ).reshape(-1, 3).astype(np.float64)
    origin = np.asarray(document["nodes"][0]["translation"], dtype=np.float64)
    index_accessor = document["accessors"][primitive["indices"]]
    index_view = document["bufferViews"][index_accessor["bufferView"]]
    start = index_view.get("byteOffset", 0)
    faces = np.frombuffer(
        binary[start:start + index_view["byteLength"]], dtype="<u4"
    ).reshape(-1, 3)
    crs = document["asset"]["extras"]["opendronekit"]["crs_epsg"]
    return positions + origin, faces, crs


class TestMeshFormats:
    @pytest.mark.parametrize("suffix", [".gltf", ".glb"])
    def test_gltf_variants_round_trip_geometry_and_georeferencing(
        self, tmp_path, geometry, suffix,
    ):
        path = Path(export_model(tmp_path / f"mesh{suffix}", geometry))
        vertices, faces, crs = _gltf_geometry(path)
        assert vertices == pytest.approx(VERTICES, abs=1e-5)
        assert np.array_equal(faces, FACES)
        assert crs == 32643

    def test_obj_round_trip_geometry(self, tmp_path, geometry):
        path = Path(export_model(tmp_path / "mesh.obj", geometry))
        vertices, faces, crs = [], [], None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# ODK_CRS=EPSG:"):
                crs = int(line.rsplit(":", 1)[1])
            elif line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value) - 1 for value in line.split()[1:4]])
        assert np.asarray(vertices) == pytest.approx(VERTICES)
        assert np.array_equal(np.asarray(faces), FACES)
        assert crs == 32643

    def test_ply_round_trip_geometry_colors_and_crs(self, tmp_path, geometry):
        path = Path(export_model(tmp_path / "mesh.ply", geometry))
        lines = path.read_text(encoding="utf-8").splitlines()
        end = lines.index("end_header")
        count = int(next(line.split()[-1] for line in lines if line.startswith("element vertex")))
        crs = int(next(line.rsplit(":", 1)[1] for line in lines if line.startswith("comment ODK_CRS EPSG:")))
        rows = [[float(value) for value in line.split()] for line in lines[end + 1:end + 1 + count]]
        assert np.asarray([row[:3] for row in rows]) == pytest.approx(VERTICES)
        assert np.array_equal(np.asarray([row[3:] for row in rows], dtype=np.uint8), COLORS)
        assert crs == 32643


class TestPointCloudFormats:
    @pytest.mark.parametrize("suffix,compressed", [(".las", False), (".laz", True)])
    def test_las_variants_round_trip_points_colors_crs_and_compression(
        self, tmp_path, geometry, suffix, compressed,
    ):
        laspy = pytest.importorskip("laspy")
        path = Path(export_model(tmp_path / f"cloud{suffix}", geometry))
        cloud = laspy.read(path)
        points = np.column_stack([cloud.x, cloud.y, cloud.z])
        colors = np.column_stack([cloud.red, cloud.green, cloud.blue]) // 257
        assert points == pytest.approx(VERTICES, abs=0.001)
        assert np.array_equal(colors.astype(np.uint8), COLORS)
        assert cloud.header.parse_crs().to_epsg() == 32643
        assert bool(cloud.header.are_points_compressed) is compressed


class TestModelExportRefusals:
    def test_las_without_a_crs_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="explicit CRS"):
            export_model(tmp_path / "cloud.las", ModelGeometry(VERTICES))

    def test_unknown_extension_is_refused_with_supported_list(self, tmp_path, geometry):
        with pytest.raises(ValueError, match="Supported"):
            export_model(tmp_path / "mesh.xyz", geometry)
