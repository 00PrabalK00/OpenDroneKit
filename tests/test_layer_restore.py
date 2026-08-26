"""A reconstruction's products have to survive closing the application.

`AppSession.layers` is an in-memory dict. Reconstruction registered the orthomosaic, DSM,
DTM, hillshade and camera track into it, and all five vanished the moment the application
closed or another project was opened -- the files still on disk in the project folder,
only the application's knowledge of them thrown away. That is the worst version of the
bug, because the operator can open the folder and see the products the panel says do not
exist.

So this is about reopening rather than about producing: run the products once, drop the
session, and require the layer tree to come back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_geotiff(path: Path) -> None:
    """A small but genuinely georeferenced raster, since a CRS-less one is flagged."""
    from core import geo

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.random.default_rng(0).random((3, 16, 16)) * 255).astype(np.uint8)
    # Projected, because the layer registry reads a real CRS and flags rasters without
    # one -- a flagged layer would pass this test while being unplaceable on the map.
    geo.write_geotiff(str(path), data, epsg=32617, west=437000.0, north=4573000.0,
                      pixel_size=0.05)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.session import AppSession

    session = AppSession()
    session.create_project("Layer restore", str(tmp_path / "project"))
    return session


def test_products_come_back_when_the_project_is_reopened(project, tmp_path, monkeypatch):
    reconstruction = Path(project.project_root()) / "reconstruction"
    write_geotiff(reconstruction / "orthomosaic.tif")
    write_geotiff(reconstruction / "dsm.tif")
    (reconstruction / "camera_track.geojson").write_text(
        '{"type": "FeatureCollection", "features": []}', encoding="utf-8"
    )

    restored = project.restore_layers()
    names = {layer["name"] for layer in project.layer_list()}
    assert restored, "nothing was restored from a folder holding three products"
    assert {"Orthomosaic", "DSM", "Camera positions"} <= names

    # And a second call must not stack duplicates: startup and project-switch both run it.
    project.restore_layers()
    assert len(project.layer_list()) == len(names)


def test_a_project_with_no_reconstruction_restores_nothing(project):
    assert project.restore_layers() == []
    assert project.layer_list() == []


def test_switching_projects_restores_the_new_one(project, tmp_path):
    reconstruction = Path(project.project_root()) / "reconstruction"
    write_geotiff(reconstruction / "orthomosaic.tif")
    project.restore_layers()
    assert project.layer_list()

    second = project.create_project("Empty one", str(tmp_path / "second"))
    # The previous project's layers are not this one's, and nothing invented takes
    # their place.
    project.set_active_project(int(second["id"]))
    assert project.layer_list() == []


def test_a_deleted_product_does_not_come_back(project):
    reconstruction = Path(project.project_root()) / "reconstruction"
    write_geotiff(reconstruction / "orthomosaic.tif")
    project.restore_layers()
    assert len(project.layer_list()) == 1

    # Rediscovering from disk rather than persisting a registry is what makes this true:
    # a product the operator deleted must not return as a broken row.
    (reconstruction / "orthomosaic.tif").unlink()
    project.layers.clear()
    assert project.restore_layers() == []
