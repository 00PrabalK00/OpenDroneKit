"""Place search, behind a replaceable provider interface.

Finding a site by typing its address is the difference between planning a mission in
seconds and hunting across a map. But geocoding is inherently a network lookup, and
this toolkit is offline-first, so the design here is deliberate:

* No provider is contacted unless the operator asks for a search.
* The default provider is OpenStreetMap Nominatim, which is open data and can be
  self-hosted. A self-hosted instance is configured by changing one URL.
* An offline provider is always available and searches only what is already on the
  machine, so the feature degrades to something useful rather than failing.

The query text is sent to whichever provider is selected. That is unavoidable for a
remote search, so `describe_provider` states plainly where a query would go, and the
UI shows it before anything leaves the machine.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Nominatim's usage policy requires a real identifying User-Agent.
USER_AGENT = "OpenDroneKit/1.0 (open-source drone inspection toolkit)"
DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


@dataclass
class Place:
    """One search result, in the ordering the map expects."""

    name: str
    longitude: float
    latitude: float
    kind: str = ""
    # west, south, east, north -- lets the map frame the result rather than guess a zoom.
    bounds: tuple[float, float, float, float] | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lon": self.longitude,
            "lat": self.latitude,
            "kind": self.kind,
            "bounds": list(self.bounds) if self.bounds else None,
            "source": self.source,
        }


class GeocodingProvider(Protocol):
    """Implement this to add a provider; nothing else needs to change."""

    name: str
    requires_network: bool

    def describe(self) -> str:
        """Where a query goes. Shown to the operator before any search."""

    def search(self, query: str, *, limit: int = 8) -> list[Place]:
        ...


@dataclass
class NominatimProvider:
    """OpenStreetMap Nominatim. Open data, and self-hostable.

    Point `base_url` at a local instance to keep queries inside your own network.
    """

    base_url: str = DEFAULT_NOMINATIM_URL
    timeout_s: float = 8.0
    name: str = "nominatim"
    requires_network: bool = True

    def describe(self) -> str:
        host = urllib.parse.urlparse(self.base_url).netloc or self.base_url
        if host.endswith("openstreetmap.org"):
            return (
                f"Queries are sent to {host} (OpenStreetMap Nominatim, public service). "
                "Set a self-hosted URL to keep searches on your own infrastructure."
            )
        return f"Queries are sent to {host} (self-hosted Nominatim)."

    def search(self, query: str, *, limit: int = 8) -> list[Place]:
        text = str(query or "").strip()
        if not text:
            return []

        url = f"{self.base_url}?" + urllib.parse.urlencode({
            "q": text,
            "format": "jsonv2",
            "limit": int(max(1, min(20, limit))),
            "addressdetails": 0,
        })
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))

        places: list[Place] = []
        for entry in payload if isinstance(payload, list) else []:
            try:
                longitude = float(entry["lon"])
                latitude = float(entry["lat"])
            except (KeyError, TypeError, ValueError):
                continue

            bounds = None
            raw_box = entry.get("boundingbox")
            if isinstance(raw_box, list) and len(raw_box) == 4:
                try:
                    # Nominatim orders this south, north, west, east.
                    south, north, west, east = (float(v) for v in raw_box)
                    bounds = (west, south, east, north)
                except (TypeError, ValueError):
                    bounds = None

            places.append(Place(
                name=str(entry.get("display_name") or text),
                longitude=longitude,
                latitude=latitude,
                kind=str(entry.get("type") or entry.get("category") or ""),
                bounds=bounds,
                source=self.name,
            ))
        return places


@dataclass
class OfflineProvider:
    """Searches only what is already on this machine.

    Two things are matched: coordinates typed directly, and named places the operator
    has saved. This is what makes the feature usable with no connectivity, rather than
    simply unavailable.
    """

    gazetteer_path: Path | None = None
    name: str = "offline"
    requires_network: bool = False
    _entries: list[Place] = field(default_factory=list, init=False)

    def describe(self) -> str:
        return "Searches saved places and typed coordinates on this machine. Nothing is sent anywhere."

    def _load(self) -> list[Place]:
        if self._entries or self.gazetteer_path is None:
            return self._entries
        path = Path(self.gazetteer_path)
        if not path.exists():
            return self._entries
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._entries
        for entry in payload.get("places", []) if isinstance(payload, dict) else payload:
            try:
                self._entries.append(Place(
                    name=str(entry["name"]),
                    longitude=float(entry["lon"]),
                    latitude=float(entry["lat"]),
                    kind=str(entry.get("kind", "saved")),
                    source=self.name,
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return self._entries

    def search(self, query: str, *, limit: int = 8) -> list[Place]:
        text = str(query or "").strip()
        if not text:
            return []

        coordinate = parse_coordinates(text)
        results: list[Place] = []
        if coordinate is not None:
            longitude, latitude = coordinate
            results.append(Place(
                name=f"{latitude:.6f}, {longitude:.6f}",
                longitude=longitude, latitude=latitude,
                kind="coordinates", source=self.name,
            ))

        needle = text.lower()
        for place in self._load():
            if needle in place.name.lower():
                results.append(place)
            if len(results) >= limit:
                break
        return results[:limit]


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Accept a typed coordinate pair, returning (lon, lat).

    Handles "41.3042, -81.7505" and "41.3042 -81.7505". Latitude is assumed first,
    which is how coordinates are written and spoken, even though this returns them in
    lon/lat order to match GeoJSON.
    """
    cleaned = str(text or "").replace(",", " ").split()
    if len(cleaned) != 2:
        return None
    try:
        first, second = float(cleaned[0]), float(cleaned[1])
    except ValueError:
        return None

    if abs(first) <= 90.0 and abs(second) <= 180.0:
        return second, first
    # Unambiguously lon/lat when the first value cannot be a latitude.
    if abs(second) <= 90.0 and abs(first) <= 180.0:
        return first, second
    return None


PROVIDERS: dict[str, type] = {
    "nominatim": NominatimProvider,
    "offline": OfflineProvider,
}


def build_provider(name: str = "nominatim", **kwargs) -> GeocodingProvider:
    """Construct a provider by name. Unknown names fall back to offline, not to a crash."""
    factory = PROVIDERS.get(str(name or "").strip().lower())
    if factory is None:
        return OfflineProvider(**{k: v for k, v in kwargs.items() if k == "gazetteer_path"})
    return factory(**kwargs)


def search_places(
    query: str,
    *,
    provider: str = "nominatim",
    limit: int = 8,
    gazetteer_path: Path | None = None,
) -> dict[str, Any]:
    """Search, always answering with the coordinate interpretation first when one exists.

    A typed coordinate never requires the network, so it is resolved locally even when
    a remote provider is selected.
    """
    text = str(query or "").strip()
    if not text:
        return {"ok": False, "error": "Enter an address, place name, or coordinates.",
                "results": [], "provider": provider}

    results: list[Place] = []
    coordinate = parse_coordinates(text)
    if coordinate is not None:
        longitude, latitude = coordinate
        results.append(Place(
            name=f"{latitude:.6f}, {longitude:.6f}",
            longitude=longitude, latitude=latitude,
            kind="coordinates", source="typed",
        ))

    engine = build_provider(provider, **({"gazetteer_path": gazetteer_path} if provider == "offline" else {}))
    error = ""
    try:
        results.extend(p for p in engine.search(text, limit=limit) if p.longitude != 0.0 or p.latitude != 0.0)
    except Exception as exc:  # noqa: BLE001
        # A failed lookup must not lose a coordinate the operator already typed.
        error = f"{type(exc).__name__}: {exc}"

    return {
        "ok": bool(results) or not error,
        "results": [p.to_dict() for p in results[:limit]],
        "provider": getattr(engine, "name", provider),
        "provider_note": engine.describe(),
        "error": error,
    }
