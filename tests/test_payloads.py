"""The payload database, and the commands a mission is allowed to send it.

The failure being guarded against is not a crash. It is a flight that completes, lands,
and produces nothing: a LiDAR sent a shutter trigger, a multispectral survey flown
without its calibration shot, a magnetometer flown on the airframe where the readings
are its own motors. Each of those exports cleanly and uploads cleanly.

So these tests care most about the refusals -- an undescribed payload, a command the
instrument does not accept, a multispectral head with no stated bands -- and about the
mission engine reaching for the right command at a capture point.
"""

from __future__ import annotations

import json

import pytest

from mission.payloads import (
    BUILT_IN,
    COMMANDS,
    PayloadCommandRefused,
    PayloadProfile,
    UnknownPayload,
    all_profiles,
    delete_user_profile,
    get_payload,
    list_payloads,
    load_user_profiles,
    payload_plan_notes,
    save_user_profile,
)


class TestTheDatabaseItself:
    def test_every_kind_the_engine_plans_for_is_represented(self):
        kinds = {p.kind for p in BUILT_IN.values()}
        assert {"rgb", "thermal", "multispectral", "lidar", "magnetometer"} <= kinds

    def test_every_built_in_declares_only_commands_the_engine_can_emit(self):
        for profile in BUILT_IN.values():
            assert set(profile.commands) <= set(COMMANDS)

    def test_multispectral_payloads_state_their_band_centres(self):
        """An index computed from unknown bands is a number with no meaning."""
        for profile in BUILT_IN.values():
            if profile.kind == "multispectral":
                assert len(profile.bands_nm) >= 3

    def test_mass_is_absent_rather_than_invented(self):
        """None reads as "not known"; a plausible number would read as measured."""
        rededge = BUILT_IN["micasense_rededge_mx"]
        assert rededge.mass_g is None
        assert rededge.to_dict()["mass_known"] is False


class TestRefusals:
    def test_an_undescribed_payload_is_refused_not_defaulted(self):
        with pytest.raises(UnknownPayload, match="not described"):
            get_payload("whatever_is_bolted_on_today")

    def test_the_refusal_names_what_is_available(self):
        with pytest.raises(UnknownPayload, match="zenmuse_l2"):
            get_payload("unknown")

    def test_a_command_the_payload_does_not_accept_is_refused(self):
        lidar = get_payload("zenmuse_l2")
        with pytest.raises(PayloadCommandRefused, match="does not accept 'trigger'"):
            lidar.require("trigger")

    def test_a_command_the_engine_cannot_emit_is_refused(self):
        with pytest.raises(PayloadCommandRefused, match="not a mission-engine command"):
            get_payload("rgb_generic").require("do_a_barrel_roll")

    def test_a_multispectral_payload_without_bands_is_rejected(self):
        with pytest.raises(ValueError, match="band centres"):
            PayloadProfile("bad", "Bandless", "multispectral", ("trigger",))

    def test_a_payload_accepting_no_command_cannot_be_flown(self):
        with pytest.raises(ValueError, match="cannot be flown"):
            PayloadProfile("inert", "Inert", "rgb", ())

    def test_a_continuous_payload_must_say_how_a_run_starts(self):
        with pytest.raises(ValueError, match="how a run is started"):
            PayloadProfile("streamer", "Streamer", "custom", ("trigger",), continuous=True)

    def test_an_invented_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown payload kind"):
            PayloadProfile("x", "X", "gravimeter", ("trigger",))


class TestCaptureCommand:
    def test_a_framing_payload_is_triggered_at_each_point(self):
        assert get_payload("zenmuse_p1").capture_command() == "trigger"

    def test_a_streaming_payload_starts_a_run_instead(self):
        """This is the whole point of the database: a LiDAR is not a camera."""
        assert get_payload("zenmuse_l2").capture_command() == "start_scan"
        assert get_payload("magnetometer_towed").capture_command() == "start_scan"


class TestPlanNotes:
    def test_a_lidar_says_what_it_changes_about_the_flight(self):
        notes = " ".join(payload_plan_notes(get_payload("zenmuse_l2")))
        assert "records continuously" in notes
        assert "overlapped lines" in notes

    def test_a_multispectral_head_demands_its_calibration_captures(self):
        notes = " ".join(payload_plan_notes(get_payload("micasense_rededge_mx")))
        assert "before and after the flight" in notes

    def test_a_magnetometer_warns_about_the_airframes_own_field(self):
        notes = " ".join(payload_plan_notes(get_payload("magnetometer_towed")))
        assert "its own" in notes and "motors" in notes

    def test_an_unknown_mass_is_stated_as_excluded_from_estimates(self):
        notes = " ".join(payload_plan_notes(get_payload("flir_vue_pro_r")))
        assert "not included in any estimate" in notes


class TestOperatorPayloads:
    def test_a_described_payload_is_saved_and_read_back(self, tmp_path):
        store = tmp_path / "payloads.json"
        profile = PayloadProfile("hyperspec", "Site hyperspectral rig", "custom",
                                 ("start_recording", "stop_recording"),
                                 mass_g=1250.0, continuous=True, source="user")
        save_user_profile(profile, store)

        loaded = load_user_profiles(store)
        assert loaded["hyperspec"].name == "Site hyperspectral rig"
        assert loaded["hyperspec"].source == "user"
        assert loaded["hyperspec"].mass_g == 1250.0

    def test_a_user_payload_may_not_shadow_a_published_one(self, tmp_path):
        clash = PayloadProfile("zenmuse_l2", "Not really an L2", "rgb", ("trigger",))
        with pytest.raises(ValueError, match="built-in payload"):
            save_user_profile(clash, tmp_path / "payloads.json")

    def test_one_malformed_entry_does_not_hide_the_rest_of_the_fleet(self, tmp_path):
        store = tmp_path / "payloads.json"
        store.write_text(json.dumps({
            "broken": {"name": "Broken", "kind": "not_a_kind", "commands": ["trigger"]},
            "good": {"name": "Good", "kind": "rgb", "commands": ["trigger"]},
        }), encoding="utf-8")

        loaded = load_user_profiles(store)
        assert "broken" not in loaded
        assert loaded["good"].name == "Good"

    def test_a_user_payload_can_be_removed(self, tmp_path):
        store = tmp_path / "payloads.json"
        save_user_profile(
            PayloadProfile("temp", "Temporary", "rgb", ("trigger",), source="user"), store)

        assert delete_user_profile("temp", store) is True
        assert load_user_profiles(store) == {}

    def test_removing_something_absent_is_false_not_an_error(self, tmp_path):
        assert delete_user_profile("never_saved", tmp_path / "payloads.json") is False

    def test_operator_payloads_layer_on_top_of_the_built_ins(self, tmp_path):
        store = tmp_path / "payloads.json"
        save_user_profile(
            PayloadProfile("mine", "Mine", "rgb", ("trigger",), source="user"), store)

        combined = all_profiles(store)
        assert "mine" in combined and "zenmuse_l2" in combined


class TestListing:
    def test_listing_is_serialisable_and_names_the_capture_command(self):
        rows = list_payloads()
        assert rows and all(isinstance(json.dumps(row), str) for row in rows)
        assert {row["capture_command"] for row in rows} <= set(COMMANDS)


AOI = [[-81.7510, 41.3035], [-81.7490, 41.3035],
       [-81.7490, 41.3050], [-81.7510, 41.3050]]


@pytest.fixture
def api(tmp_path):
    """A session on its own database, so tests cannot contend for the shared store."""
    from app.api import Api
    from app.session import AppSession
    from app.store import ProjectStore

    session = AppSession(store=ProjectStore(tmp_path / "projects.db"))
    session.create_project("payloads", root_dir=str(tmp_path / "project"))
    return Api(session)


class TestPlanningWithAPayload:
    """The database only earns its place if planning actually consults it."""

    def test_the_plan_carries_the_command_the_fitted_payload_understands(self, api):
        assert api.set_aoi(AOI)["ok"]
        result = api.plan_mission({"altitude_m": 60.0, "payload": "zenmuse_l2"})

        assert result["ok"], result.get("error")
        assert result["payload"]["capture_command"] == "start_scan"
        assert result["payload"]["kind"] == "lidar"

    def test_a_framing_payload_still_triggers_per_point(self, api):
        assert api.set_aoi(AOI)["ok"]
        result = api.plan_mission({"altitude_m": 60.0, "payload": "zenmuse_p1"})

        assert result["payload"]["capture_command"] == "trigger"

    def test_what_the_payload_changes_is_surfaced_as_a_warning(self, api):
        assert api.set_aoi(AOI)["ok"]
        result = api.plan_mission({"altitude_m": 60.0, "payload": "micasense_rededge_mx"})

        warnings = " ".join(result["warnings"])
        assert "before and after the flight" in warnings

    def test_planning_with_an_undescribed_payload_is_refused(self, api):
        assert api.set_aoi(AOI)["ok"]
        result = api.plan_mission({"altitude_m": 60.0, "payload": "mystery_box"})

        assert result["ok"] is False
        assert "not described" in result["error"]

    def test_planning_without_naming_a_payload_claims_nothing_about_one(self, api):
        assert api.set_aoi(AOI)["ok"]
        result = api.plan_mission({"altitude_m": 60.0})

        assert result["ok"] is True
        assert result["payload"] is None

    def test_the_api_describes_one_payload_with_its_plan_notes(self, api):
        described = api.describe_payload("magnetometer_towed")

        assert described["ok"] is True
        assert described["payload"]["continuous"] is True
        assert any("tether" in note for note in described["plan_notes"])

    def test_the_api_refuses_to_describe_an_unknown_payload(self, api):
        assert api.describe_payload("nope")["ok"] is False
