"""Place search: coordinate parsing, provider selection, and offline behaviour.

No test here touches the network. The remote provider is exercised through a stub
transport so the parsing of a real Nominatim payload is covered without a lookup.
"""

from __future__ import annotations

import json
import pytest

from core.geocoding import (
    NominatimProvider,
    OfflineProvider,
    build_provider,
    parse_coordinates,
    search_places,
)


class TestCoordinateParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("41.3042, -81.7505", (-81.7505, 41.3042)),
            ("41.3042 -81.7505", (-81.7505, 41.3042)),
            ("-33.8688, 151.2093", (151.2093, -33.8688)),
            ("0, 0", (0.0, 0.0)),
        ],
    )
    def test_latitude_first_pairs_return_lon_lat(self, text, expected):
        """Coordinates are written lat/lon but GeoJSON wants lon/lat."""
        assert parse_coordinates(text) == pytest.approx(expected)

    def test_unambiguous_lon_lat_is_accepted(self):
        """151 cannot be a latitude, so the pair must be read the other way round."""
        assert parse_coordinates("151.2093 -33.8688") == pytest.approx((151.2093, -33.8688))

    @pytest.mark.parametrize("text", ["", "not coordinates", "41.3042", "1 2 3", "abc, def"])
    def test_non_coordinates_are_rejected(self, text):
        assert parse_coordinates(text) is None

    def test_out_of_range_values_are_rejected(self):
        assert parse_coordinates("999 999") is None


class TestOfflineProvider:
    def test_requires_no_network(self):
        provider = OfflineProvider()
        assert provider.requires_network is False
        assert "Nothing is sent anywhere" in provider.describe()

    def test_resolves_typed_coordinates(self):
        results = OfflineProvider().search("41.3042, -81.7505")
        assert results
        assert results[0].latitude == pytest.approx(41.3042)
        assert results[0].longitude == pytest.approx(-81.7505)
        assert results[0].kind == "coordinates"

    def test_searches_a_saved_gazetteer(self, tmp_path):
        gazetteer = tmp_path / "places.json"
        gazetteer.write_text(json.dumps({"places": [
            {"name": "Aukerman Test Site", "lon": -81.7505, "lat": 41.3042},
            {"name": "North Bridge", "lon": -0.1278, "lat": 51.5074},
        ]}), encoding="utf-8")

        results = OfflineProvider(gazetteer_path=gazetteer).search("aukerman")
        assert len(results) == 1
        assert results[0].name == "Aukerman Test Site"

    def test_empty_query_returns_nothing(self):
        assert OfflineProvider().search("   ") == []


class TestNominatimParsing:
    def test_payload_is_parsed_without_a_network_call(self, monkeypatch):
        payload = json.dumps([{
            "display_name": "Aukerman Road, Ohio, United States",
            "lat": "41.3042", "lon": "-81.7505",
            "type": "road",
            "boundingbox": ["41.3000", "41.3100", "-81.7600", "-81.7400"],
        }]).encode("utf-8")

        class StubResponse:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: StubResponse())

        results = NominatimProvider().search("aukerman road")
        assert len(results) == 1
        place = results[0]
        assert place.latitude == pytest.approx(41.3042)
        assert place.longitude == pytest.approx(-81.7505)
        # Nominatim gives south,north,west,east; the map wants west,south,east,north.
        assert place.bounds == pytest.approx((-81.76, 41.30, -81.74, 41.31))

    def test_describe_names_the_destination(self):
        assert "openstreetmap.org" in NominatimProvider().describe()

    def test_self_hosted_url_is_reported_as_such(self):
        provider = NominatimProvider(base_url="http://gis.internal:8080/search")
        assert "self-hosted" in provider.describe()
        assert "gis.internal" in provider.describe()


class TestProviderSelection:
    def test_unknown_provider_falls_back_to_offline(self):
        """An unrecognised name must not crash mission planning."""
        provider = build_provider("does-not-exist")
        assert provider.requires_network is False

    def test_named_providers_resolve(self):
        assert build_provider("offline").name == "offline"
        assert build_provider("nominatim").name == "nominatim"


class TestSearchPlaces:
    def test_coordinates_resolve_without_contacting_a_provider(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("no network call should be made for a typed coordinate")

        monkeypatch.setattr("urllib.request.urlopen", explode)

        result = search_places("41.3042, -81.7505", provider="offline")
        assert result["ok"]
        assert result["results"][0]["kind"] == "coordinates"

    def test_a_provider_failure_still_returns_the_typed_coordinate(self, monkeypatch):
        """A dead network must not discard something the operator already gave us."""
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: (_ for _ in ()).throw(OSError("network down")),
        )

        result = search_places("41.3042, -81.7505", provider="nominatim")
        assert result["results"], "the typed coordinate was lost"
        assert result["error"]

    def test_empty_query_is_refused_with_a_reason(self):
        result = search_places("  ")
        assert result["ok"] is False
        assert "coordinates" in result["error"]

    def test_result_carries_the_provider_note(self):
        result = search_places("41.3042, -81.7505", provider="offline")
        assert result["provider_note"]
