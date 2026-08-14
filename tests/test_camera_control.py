"""Camera control, and refusing to claim controls that do not exist.

MAVLink's camera protocol splits into commands any compliant camera accepts and
extended parameters most payloads never implement. The failure worth preventing is an
operator believing they set ISO 100 when nothing received the request -- they will not
find out until the imagery comes back wrong, by which time the survey is flown.

So the tests check three things: a declared capability is used, an undeclared one is
refused with a reason, and an undeclared *capability set* is reported as unknown rather
than as absent, because plenty of working payloads never send CAMERA_INFORMATION.
"""

from __future__ import annotations

import pytest

from core.camera_control import (
    CAP_CAPTURE_IMAGE,
    CAP_CAPTURE_VIDEO,
    CAP_HAS_BASIC_FOCUS,
    CAP_HAS_BASIC_ZOOM,
    CAP_HAS_MODES,
    CAP_HAS_SURVEY_MODE,
    CMD_IMAGE_START_CAPTURE,
    CMD_SET_CAMERA_MODE,
    CMD_SET_CAMERA_ZOOM,
    CMD_VIDEO_START_CAPTURE,
    CameraCapabilities,
    CameraController,
    capabilities_from_message,
)

FULL = (CAP_CAPTURE_IMAGE | CAP_CAPTURE_VIDEO | CAP_HAS_MODES
        | CAP_HAS_BASIC_ZOOM | CAP_HAS_BASIC_FOCUS | CAP_HAS_SURVEY_MODE)


class Recorder:
    """Captures the commands that would go on the wire."""

    def __init__(self, succeed: bool = True):
        self.sent: list[tuple] = []
        self.succeed = succeed

    def __call__(self, command: int, *params: float):
        self.sent.append((command, params))
        return self.succeed

    @property
    def commands(self) -> list[int]:
        return [c for c, _ in self.sent]


def controller(flags: int | None = FULL, succeed: bool = True):
    recorder = Recorder(succeed)
    return CameraController(recorder, CameraCapabilities(flags=flags)), recorder


class TestCapture:
    def test_a_photo_is_taken_when_the_camera_supports_it(self):
        camera, recorder = controller()
        result = camera.take_photo()

        assert result.ok is True
        assert CMD_IMAGE_START_CAPTURE in recorder.commands

    def test_a_burst_passes_its_count_and_interval(self):
        camera, recorder = controller()
        camera.take_photo(count=5, interval_s=2.0)

        _, params = recorder.sent[0]
        assert params[1] == pytest.approx(2.0)
        assert params[2] == pytest.approx(5.0)

    def test_video_starts_when_supported(self):
        camera, recorder = controller()
        assert camera.start_video().ok is True
        assert CMD_VIDEO_START_CAPTURE in recorder.commands

    def test_a_camera_that_cannot_take_photos_is_refused_not_attempted(self):
        """Transmitting into silence and reporting success is the failure here."""
        camera, recorder = controller(flags=CAP_CAPTURE_VIDEO)
        result = camera.take_photo()

        assert result.ok is False
        assert result.unsupported is True
        assert recorder.sent == [], "no command should reach a camera that cannot do it"

    def test_the_refusal_explains_why_rather_than_failing_generically(self):
        camera, _ = controller(flags=CAP_CAPTURE_VIDEO)
        assert "does not report image capture" in camera.take_photo().message

    def test_a_camera_that_cannot_record_video_is_refused(self):
        camera, _ = controller(flags=CAP_CAPTURE_IMAGE)
        assert camera.start_video().unsupported is True


class TestModes:
    def test_a_known_mode_is_sent(self):
        camera, recorder = controller()
        assert camera.set_mode("video").ok is True
        assert CMD_SET_CAMERA_MODE in recorder.commands

    def test_an_unknown_mode_is_refused_and_lists_the_real_ones(self):
        camera, _ = controller()
        result = camera.set_mode("cinematic")

        assert result.ok is False
        assert "photo" in result.message

    def test_survey_mode_is_refused_when_not_declared(self):
        camera, _ = controller(flags=CAP_CAPTURE_IMAGE | CAP_HAS_MODES)
        result = camera.set_mode("survey")

        assert result.unsupported is True
        assert "survey mode" in result.message

    def test_mode_switching_is_refused_when_the_camera_has_no_modes(self):
        camera, recorder = controller(flags=CAP_CAPTURE_IMAGE)
        assert camera.set_mode("video").unsupported is True
        assert recorder.sent == []


class TestZoomAndFocus:
    def test_zoom_is_sent_as_a_percentage(self):
        camera, recorder = controller()
        assert camera.set_zoom(40.0).ok is True

        command, params = recorder.sent[0]
        assert command == CMD_SET_CAMERA_ZOOM
        assert params[1] == pytest.approx(40.0)

    def test_zoom_outside_the_range_is_refused(self):
        camera, recorder = controller()
        assert camera.set_zoom(140.0).ok is False
        assert recorder.sent == []

    def test_focus_is_refused_when_the_camera_has_no_focus_control(self):
        camera, _ = controller(flags=CAP_CAPTURE_IMAGE)
        assert camera.set_focus(50.0).unsupported is True

    def test_focus_within_range_is_sent(self):
        camera, recorder = controller()
        assert camera.set_focus(75.0).ok is True
        assert recorder.sent


class TestExtendedSettings:
    @pytest.mark.parametrize("setting", ["iso", "shutter_speed", "white_balance",
                                         "exposure_compensation"])
    def test_an_exposure_setting_is_refused_rather_than_silently_dropped(self, setting):
        """The operator must not believe an exposure was applied that was not."""
        camera, recorder = controller()
        result = camera.set_exposure_setting(setting, 100)

        assert result.ok is False
        assert result.unsupported is True
        assert recorder.sent == []

    def test_the_refusal_names_the_parameter_and_says_where_to_set_it(self):
        camera, _ = controller()
        result = camera.set_exposure_setting("iso", 400)

        assert result.details["parameter"] == "CAM_ISO"
        assert "on the payload directly" in result.message
        assert "believing an exposure was applied" in result.message

    def test_an_unknown_setting_lists_the_ones_that_exist(self):
        camera, _ = controller()
        result = camera.set_exposure_setting("film_stock", "portra")
        assert "iso" in result.message


class TestUndeclaredCapabilities:
    def test_an_undeclared_camera_is_unknown_not_unsupported(self):
        """Plenty of working payloads never send CAMERA_INFORMATION."""
        camera, recorder = controller(flags=None)
        result = camera.take_photo()

        assert result.ok is True, "an unknown capability must not block the attempt"
        assert recorder.sent

    def test_the_description_says_capabilities_are_unknown(self):
        camera, _ = controller(flags=None)
        described = camera.describe()

        assert described["declared"] is False
        assert described["capture_image"] is None
        assert "unknown rather than absent" in described["note"]

    def test_a_declared_camera_reports_each_capability(self):
        camera, _ = controller(flags=CAP_CAPTURE_IMAGE | CAP_HAS_BASIC_ZOOM)
        described = camera.describe()

        assert described["capture_image"] is True
        assert described["capture_video"] is False
        assert described["zoom"] is True

    def test_the_description_warns_about_extended_settings(self):
        camera, _ = controller()
        assert "extended parameters" in camera.describe()["note"]


class TestCapabilityParsing:
    def test_camera_information_is_read_into_capabilities(self):
        class Message:
            flags = CAP_CAPTURE_IMAGE | CAP_HAS_BASIC_ZOOM
            vendor_name = b"Hasselblad\x00\x00"
            model_name = b"L1D-20c\x00"

        capabilities = capabilities_from_message(Message())
        assert capabilities.declared is True
        assert capabilities.vendor == "Hasselblad"
        assert capabilities.model == "L1D-20c"
        assert capabilities.has(CAP_CAPTURE_IMAGE) is True
        assert capabilities.has(CAP_CAPTURE_VIDEO) is False

    def test_a_name_arriving_as_a_byte_list_is_decoded(self):
        class Message:
            flags = 0
            vendor_name = [68, 74, 73, 0, 0]
            model_name = []

        assert capabilities_from_message(Message()).vendor == "DJI"


class TestTransportFailures:
    def test_a_refused_command_reports_failure_not_success(self):
        camera, _ = controller(succeed=False)
        assert camera.take_photo().ok is False

    def test_an_exception_in_the_transport_is_reported_not_raised(self):
        def explode(*_args):
            raise ConnectionError("link lost")

        camera = CameraController(explode, CameraCapabilities(flags=FULL))
        result = camera.take_photo()

        assert result.ok is False
        assert "link lost" in result.message
