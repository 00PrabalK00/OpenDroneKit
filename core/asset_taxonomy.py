"""One asset vocabulary across the packs, and one geospatial form for the results.

Every pack grew its own class set. Power inspection knows "pole", rail knows "signal",
solar knows "module", agriculture counts "tree". Each was right for its own report and
wrong for anything that crosses domains: a survey covering a substation and the rail
line beside it produced two inventories with no shared vocabulary, and nothing could
answer "how many assets are on this site" without a human reading both.

This module is the shared taxonomy and the one output form. It deliberately does very
little arithmetic. What it enforces is the part that goes wrong quietly:

  * An asset instance without model provenance is refused, not counted. A count is a
    claim about the world, and a claim whose origin cannot be named cannot be checked
    later against the model that made it.
  * An asset instance without a location is refused. "Seventeen poles" with no geometry
    cannot be verified, disputed, or revisited.
  * Confidence travels with every instance and is never averaged away into a headline.

The taxonomy is intentionally small. Adding a type is cheap; removing one after it has
appeared in a client deliverable is not, so each entry has to earn its place by being
something a drone survey can actually resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


class AssetRefused(ValueError):
    """An asset inventory could not be produced honestly."""


@dataclass(frozen=True)
class AssetType:
    """One asset the toolkit can inventory.

    ``geometry`` is what a detection of this type legitimately produces. A pole is a
    point because its footprint is smaller than the GSD of most surveys; a road is a
    polygon because its extent is the measurement. Recording this stops a point
    detection being reported as an area, which is how a "12 m2 pole" reaches a report.

    ``countable`` marks types where an integer count is meaningful. Conductor and track
    are continuous -- "seven conductors" means seven spans only if someone defined a
    span, so they are measured by length instead.
    """

    name: str
    domain: str
    geometry: str          # "point" | "line" | "polygon"
    countable: bool
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


# Grounded in the class sets the packs already use, so nothing here is invented
# vocabulary: power and rail come from core/india_assets.py, canopy and plant counting
# from core/india_agriculture.py, modules from the solar inventory.
ASSET_TYPES: tuple[AssetType, ...] = (
    # -- power -------------------------------------------------------------------
    AssetType("tower", "power", "point", True,
              "Lattice or monopole transmission structure.", ("pylon", "transmission_tower")),
    AssetType("pole", "power", "point", True,
              "Distribution pole.", ("utility_pole", "electric_pole")),
    AssetType("insulator", "power", "point", True,
              "Insulator string or unit, resolvable only at close stand-off.", ("insulator_string",)),
    AssetType("transformer", "power", "point", True,
              "Pole-mounted or pad-mounted transformer.", ()),
    AssetType("conductor", "power", "line", False,
              "Conductor span. Measured by length: a count needs a span definition.", ("wire", "cable")),
    # -- rail --------------------------------------------------------------------
    AssetType("track", "rail", "line", False,
              "Running track. Measured by length, not counted.", ("rail", "railway_track")),
    AssetType("signal", "rail", "point", True, "Lineside signal.", ("rail_signal",)),
    AssetType("overhead_equipment", "rail", "line", False,
              "Overhead line equipment above the track.", ("ohe", "catenary")),
    # -- solar -------------------------------------------------------------------
    AssetType("module", "solar", "polygon", True,
              "A single PV module. Countable because its outline is resolvable.",
              ("panel", "pv_module", "solar_module")),
    AssetType("inverter", "solar", "point", True, "String or central inverter.", ()),
    # -- vegetation and agriculture ----------------------------------------------
    AssetType("tree", "vegetation", "point", True,
              "An individual tree crown.", ("trees", "crown")),
    AssetType("plant", "agriculture", "point", True,
              "An individual crop plant.", ("crop_plant", "seedling")),
    # -- built environment --------------------------------------------------------
    AssetType("building", "built", "polygon", True, "A building footprint.", ("structure",)),
    AssetType("road", "built", "polygon", False,
              "Road surface. Measured by area and length rather than counted.",
              ("road_surface", "paved_road", "carriageway")),
    AssetType("equipment", "built", "point", True,
              "Plant or equipment on site, where the survey cannot say more than that.",
              ("machinery", "plant_equipment")),
)

_BY_NAME: dict[str, AssetType] = {}
for _asset in ASSET_TYPES:
    _BY_NAME[_asset.name] = _asset
    for _alias in _asset.aliases:
        _BY_NAME[_alias] = _asset

DOMAINS: tuple[str, ...] = tuple(sorted({asset.domain for asset in ASSET_TYPES}))


def canonical_asset(name: str) -> AssetType:
    """Resolve a name or alias to its asset type, refusing anything unknown.

    Unknown names are refused rather than passed through. A taxonomy that accepts
    whatever it is handed is not a taxonomy, and the failure it produces is an
    inventory where "pole", "Pole" and "utility_pole" are three different assets.
    """
    key = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    asset = _BY_NAME.get(key)
    if asset is None:
        raise AssetRefused(
            f"{name!r} is not in the asset taxonomy. Known types: "
            f"{', '.join(sorted(a.name for a in ASSET_TYPES))}."
        )
    return asset


def assets_for_domain(domain: str) -> tuple[AssetType, ...]:
    key = str(domain or "").strip().lower()
    if key not in DOMAINS:
        raise AssetRefused(f"{domain!r} is not a known domain. Known: {', '.join(DOMAINS)}.")
    return tuple(asset for asset in ASSET_TYPES if asset.domain == key)


def _require_provenance(model: Mapping[str, Any] | None) -> dict[str, Any]:
    """Model identity or nothing.

    The digest matters more than the name. A registry key says which model was meant to
    run; the sha256 says which file did. When those disagree -- a model replaced in
    place, a checkpoint copied over -- only the digest notices, and an inventory whose
    provenance cannot be checked is a number with a story attached.
    """
    if not model:
        raise AssetRefused(
            "Asset instances need model provenance. Counting detections whose origin "
            "cannot be named produces a figure nobody can audit."
        )
    key = str(model.get("key") or "").strip()
    digest = str(model.get("sha256") or "").strip()
    if not key or not digest:
        raise AssetRefused("Model provenance needs both a registry key and a sha256 digest.")
    return {"model_key": key, "model_sha256": digest}


def _geometry_for(asset: AssetType, instance: Mapping[str, Any]) -> dict[str, Any]:
    geometry = instance.get("geometry")
    if not isinstance(geometry, Mapping) or not geometry.get("type"):
        raise AssetRefused(
            f"A {asset.name} instance has no geometry. An asset that cannot be located "
            "cannot be verified on site."
        )
    expected = {"point": "Point", "line": "LineString", "polygon": "Polygon"}[asset.geometry]
    actual = str(geometry.get("type"))
    if actual not in {expected, f"Multi{expected}"}:
        raise AssetRefused(
            f"A {asset.name} is a {asset.geometry} in this taxonomy, but arrived as "
            f"{actual}. Reporting a point detection as an area is how a survey ends up "
            "quoting a footprint it never measured."
        )
    return dict(geometry)


def build_asset_inventory(
    instances: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, Any] | None = None,
    crs: str | None = None,
) -> dict[str, Any]:
    """Turn detections into one GeoJSON inventory carrying provenance and confidence.

    ``crs`` is required and is not defaulted to WGS84. A coordinate whose reference
    system is assumed is a coordinate that lands somewhere else, and asset positions
    exist to be navigated to.
    """
    if not str(crs or "").strip():
        raise AssetRefused(
            "An asset inventory needs its CRS stated. Assuming WGS84 puts every asset "
            "wherever the assumption happens to be wrong."
        )
    provenance = _require_provenance(model)

    features: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    uncountable: dict[str, int] = {}
    for index, instance in enumerate(instances):
        asset = canonical_asset(instance.get("asset_type", ""))
        geometry = _geometry_for(asset, instance)

        confidence = instance.get("confidence")
        if confidence is None:
            raise AssetRefused(
                f"A {asset.name} instance carries no confidence. A reviewer deciding "
                "what to check on site needs to know which calls were marginal."
            )
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise AssetRefused(f"Confidence {confidence} is outside 0..1 for a {asset.name}.")

        properties = dict(instance.get("properties") or {})
        properties.update({
            "asset_type": asset.name,
            "domain": asset.domain,
            "confidence": round(confidence, 4),
            "countable": asset.countable,
            **provenance,
        })
        features.append({
            "type": "Feature",
            "id": instance.get("id") or f"{asset.name}-{index + 1}",
            "geometry": geometry,
            "properties": properties,
        })
        if asset.countable:
            counts[asset.name] = counts.get(asset.name, 0) + 1
        else:
            uncountable[asset.name] = uncountable.get(asset.name, 0) + 1

    return {
        "type": "FeatureCollection",
        "crs": str(crs),
        "features": features,
        "summary": {
            "counts": dict(sorted(counts.items())),
            # Named rather than omitted: a caller that sees only `counts` would conclude
            # the survey found no track, when what happened is that track is not counted.
            "present_but_not_counted": dict(sorted(uncountable.items())),
            "instance_total": len(features),
            **provenance,
            "reading_note": (
                "Counts are of detections, not of assets in the world. Anything the "
                "model missed is absent from both the count and this note's ability to "
                "warn about it. Continuous types are reported under "
                "present_but_not_counted and must be measured by length or area."
            ),
        },
    }


def count_assets(inventory: Mapping[str, Any], asset_type: str) -> int:
    """Count one type from an inventory, refusing types that cannot be counted."""
    asset = canonical_asset(asset_type)
    if not asset.countable:
        raise AssetRefused(
            f"{asset.name} is continuous in this taxonomy and is not counted. Measure "
            "it by length or area instead."
        )
    return int((inventory.get("summary", {}).get("counts", {}) or {}).get(asset.name, 0))


def filter_by_confidence(inventory: Mapping[str, Any], minimum: float) -> dict[str, Any]:
    """Drop low-confidence instances and recount, keeping what was removed visible."""
    if not 0.0 <= float(minimum) <= 1.0:
        raise AssetRefused("A confidence threshold must lie in 0..1.")
    kept: list[dict[str, Any]] = []
    removed = 0
    for feature in inventory.get("features", []):
        if float(feature["properties"]["confidence"]) >= float(minimum):
            kept.append(feature)
        else:
            removed += 1

    counts: dict[str, int] = {}
    uncountable: dict[str, int] = {}
    for feature in kept:
        name = feature["properties"]["asset_type"]
        target = counts if feature["properties"]["countable"] else uncountable
        target[name] = target.get(name, 0) + 1

    summary = dict(inventory.get("summary", {}))
    summary.update({
        "counts": dict(sorted(counts.items())),
        "present_but_not_counted": dict(sorted(uncountable.items())),
        "instance_total": len(kept),
        "confidence_threshold": float(minimum),
        # Reported, not silently applied. A count that dropped by a third under
        # thresholding is a different claim from one that did not move.
        "removed_below_threshold": removed,
    })
    return {**dict(inventory), "features": kept, "summary": summary}
