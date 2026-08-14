"""Defect risk ranking: ordering, provenance, and refusal to invent a verdict."""

from __future__ import annotations

import pytest

from core.risk_scoring import score_run_risk


def test_no_defects_does_not_imply_a_sound_asset(defect_layer):
    """An empty result must not read as a clean bill of health."""
    result = score_run_risk(None, defects_geojson=None)
    assert result["defect_count"] == 0
    assert "not evidence" in result["note"]


def test_structural_defects_outrank_cosmetic_ones(defect_layer):
    result = score_run_risk(None, defects_geojson=defect_layer, structure_type="bridge")
    ordered = [d["defect_type"] for d in result["defects"]]
    assert ordered[0] == "spalling"
    assert ordered.index("spalling") < ordered.index("efflorescence")


def test_ranks_are_dense_and_sorted(defect_layer):
    result = score_run_risk(None, defects_geojson=defect_layer)
    ranks = [d["rank"] for d in result["defects"]]
    scores = [d["risk_score"] for d in result["defects"]]
    assert ranks == list(range(1, len(ranks) + 1))
    assert scores == sorted(scores, reverse=True)


def test_georeferenced_layer_is_preferred_and_declared(defect_layer):
    result = score_run_risk(None, defects_geojson=defect_layer)
    assert result["source"] == "georeferenced_defect_layer"
    assert "georeferenced areas" in result["measurement_note"]
    for defect in result["defects"]:
        assert defect["components"]["extent_basis"] in {"area_m2", "length_m"}


def test_unmeasured_extent_is_labelled_as_such(tmp_path):
    """Without a georeferenced layer the extent is estimated, and must say so."""
    import json

    summary = tmp_path / "defects.json"
    summary.write_text(
        json.dumps({"defects": [{"defect_type": "crack", "confidence": 0.8}]}),
        encoding="utf-8",
    )
    result = score_run_risk(summary)
    assert result["source"] == "detection_output"
    assert "not measured" in result["measurement_note"] or "estimated" in result["measurement_note"]
    assert result["defects"][0]["components"]["extent_basis"] == "unmeasured"


def test_structure_type_changes_the_ranking(defect_layer):
    """Corrosion matters more on a steel tower than on a concrete deck."""
    bridge = score_run_risk(None, defects_geojson=defect_layer, structure_type="bridge")
    tower = score_run_risk(None, defects_geojson=defect_layer, structure_type="tower")
    bridge_scores = {d["defect_id"]: d["risk_score"] for d in bridge["defects"]}
    tower_scores = {d["defect_id"]: d["risk_score"] for d in tower["defects"]}
    assert bridge_scores != tower_scores


def test_every_defect_carries_its_working(defect_layer):
    """A score that cannot be argued with is not useful to an inspector."""
    result = score_run_risk(None, defects_geojson=defect_layer)
    for defect in result["defects"]:
        components = defect["components"]
        assert set(components) >= {
            "class_severity", "extent_factor", "extent_basis",
            "structure_exposure", "confidence",
        }
        assert defect["action"] in {"immediate", "urgent", "planned", "monitor", "record"}


def test_index_is_driven_by_the_worst_defects(defect_layer):
    """One critical defect among many trivial ones must not average away."""
    result = score_run_risk(None, defects_geojson=defect_layer)
    worst = max(d["risk_score"] for d in result["defects"])
    mean = sum(d["risk_score"] for d in result["defects"]) / len(result["defects"])
    assert result["risk_index"] >= mean
    assert result["risk_index"] <= worst + 1e-9
    assert 0.0 <= result["integrity_score"] <= 100.0
