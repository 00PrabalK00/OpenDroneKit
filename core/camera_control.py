"""Controlling the camera, and being honest about which controls exist.

MAVLink's camera protocol is not one thing. Taking a photograph, starting a video and
setting a zoom level are standard commands that any compliant camera accepts. ISO,
shutter speed, white balance and exposure compensation are not: they are exposed
through the extended parameter protocol, which most autopilot and payload combinations
simply do not implement. A camera can be perfectly functional and still ignore every
one of them.

So this asks the camera what it can do before offering to do it. A payload advertises
its capabilities in CAMERA_INFORMATION as a bitmask, and every command here is gated on
the relevant flag. A control the camera has not declared is refused with the reason,
rather than transmitted into silence and reported as success -- an operator who thinks
they set ISO 100 and did not will not find out until the imagery comes back wrong.

Where a camera declares nothing at all, that is reported as unknown rather than as
unsupported. Plenty of working payloads never send CAMERA_INFORMATION; the honest
statement is that we do not know, and the operator may try the command anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# MAV_CMD values from the common message set.
CMD_SET_CAMERA_MODE = 530
CMD_SET_CAMERA_ZOOM = 531
CMD_SET_CAMERA_FOCUS = 532
CMD_IMAGE_START_CAPTURE = 2000
CMD_IMAGE_STOP_CAPTURE = 2001
CMD_VIDEO_START_CAPTURE = 2500
CMD_VIDEO_STOP_CAPTURE = 2501

# CAMERA_CAP_FLAGS, from CAMERA_INFORMATION.flags.
CAP_CAPTURE_VIDEO = 1 << 0
CAP_CAPTURE_IMAGE = 1 << 1
CAP_HAS_MODES = 1 << 2
CAP_IMAGE_IN_VIDEO_MODE = 1 << 3
CAP_VIDEO_IN_IMAGE_MODE = 1 << 4
CAP_HAS_SURVEY_MODE = 1 << 5
CAP_HAS_BASIC_ZOOM = 1 << 6
CAP_HAS_BASIC_FOCUS = 1 << 7
CAP_HAS_VIDEO_STREAM = 1 << 8

CAMERA_MODE_PHOTO = 0
CAMERA_MODE_VIDEO = 1
CAMERA_MODE_SURVEY = 2

MODE_NAMES = {"photo": CAMERA_MODE_PHOTO, "video": CAMERA_MODE_VIDEO,
              "survey": CAMERA_MODE_SURVEY}

# Settings that exist only through the extended parameter protocol. Named here so the
# refusal can explain itself instead of reporting a generic failure.
EXTENDED_SETTINGS = {
    "iso": "CAM_ISO",
    "shutter_speed": "CAM_SHUTTERSPD",
    "exposure_mode": "CAM_EXPMODE",
    "exposure_compensation": "CAM_EV",
    "white_balance": "CAM_WBMODE",
    "image_format": "CAM_IMGFMT",
}


@dataclass
class CameraCapabilities:
    """What a payload has said it can do."""

    flags: int | None = None
    vendor: str = ""
    model: str = ""

    @property
    def declared(self) -> bool:
        """Whether the camera has told us anything at all."""
        return self.flags is not None

    def has(self, flag: int) -> bool | None:
        """True, False, or None when the camera has not declared its capabilities."""
        if self.flags is None:
            return None
        return bool(self.flags & flag)

    def to_dict(self) -> dict[str, Any]:
        def state(flag: int) -> bool | None:
            return self.has(flag)

        return {
            "declared": self.declared,
            "vendor": self.vendor,
            "model": self.model,
            "capture_image": state(CAP_CAPTURE_IMAGE),
            "capture_video": state(CAP_CAPTURE_VIDEO),
            "modes": state(CAP_HAS_MODES),
            "zoom": state(CAP_HAS_BASIC_ZOOM),
            "focus": state(CAP_HAS_BASIC_FOCUS),
            "survey_mode": state(CAP_HAS_SURVEY_MODE),
            "video_stream": state(CAP_HAS_VIDEO_STREAM),
            "extended_settings": sorted(EXTENDED_SETTINGS),
            "note": (
                "Capabilities come from the camera's own CAMERA_INFORMATION message. "
                if self.declared else
                "This camera has not sent CAMERA_INFORMATION, so its capabilities are "
                "unknown rather than absent. Commands may still work. "
            ) + (
                "ISO, shutter, white balance and exposure are extended parameters, not "
                "standard MAVLink commands, and many payloads do not implement them."
            ),
        }


@dataclass
class CameraResult:
    """The outcome of one camera command."""

    ok: bool
    action: str
    message: str = ""
    unsupported: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "action": self.action, "message": self.message,
            "unsupported": self.unsupported, "details": self.details,
        }


def capabilities_from_message(message: Any) -> CameraCapabilities:
    """Read a CAMERA_INFORMATION message into capabilities."""
    def text(value: Any) -> str:
        if isinstance(value, (bytes, bytearray, list)):
            return bytes(bytearray(v for v in value if v)).decode(
                "ascii", errors="replace").strip()
        return str(value or "").strip("\x00").strip()

    return CameraCapabilities(
        flags=int(getattr(message, "flags", 0) or 0),
        vendor=text(getattr(message, "vendor_name", "")),
        model=text(getattr(message, "model_name", "")),
    )


class CameraController:
    """Live camera control over a MAVLink command sender.

    The sender is any callable taking a command id and up to seven parameters, which is
    what the bridge already exposes. Keeping the dependency that narrow means this is
    testable without a vehicle and reusable by any driver that can send a command.
    """

    def __init__(self, send_command, capabilities: CameraCapabilities | None = None):
        self._send = send_command
        self.capabilities = capabilities or CameraCapabilities()

    # -- capability gate ---------------------------------------------------

    def _require(self, flag: int, action: str, what: str) -> CameraResult | None:
        """Refuse a command the camera has said it cannot perform."""
        state = self.capabilities.has(flag)
        if state is False:
            return CameraResult(
                False, action, unsupported=True,
                message=(
                    f"This camera does not report {what}. Sending the command anyway "
                    "would produce no effect and no error, so it is refused here."
                ),
            )
        return None

    def _dispatch(self, command: int, *params: float, action: str) -> CameraResult:
        try:
            result = self._send(command, *params)
        except Exception as exc:  # noqa: BLE001
            return CameraResult(False, action, f"{type(exc).__name__}: {exc}")

        ok = bool(getattr(result, "success", getattr(result, "ok", result)))
        message = str(getattr(result, "message", "") or "")
        return CameraResult(ok, action, message or ("Sent." if ok else "Refused."))

    # -- capture -----------------------------------------------------------

    def take_photo(self, count: int = 1, interval_s: float = 0.0) -> CameraResult:
        refusal = self._require(CAP_CAPTURE_IMAGE, "take_photo", "image capture")
        if refusal:
            return refusal
        # param1 reserved, param2 interval, param3 count (0 = unlimited), param4 seq.
        return self._dispatch(CMD_IMAGE_START_CAPTURE, 0, float(interval_s),
                              float(count), 0, action="take_photo")

    def stop_photo_sequence(self) -> CameraResult:
        return self._dispatch(CMD_IMAGE_STOP_CAPTURE, 0, action="stop_photo_sequence")

    def start_video(self) -> CameraResult:
        refusal = self._require(CAP_CAPTURE_VIDEO, "start_video", "video capture")
        if refusal:
            return refusal
        return self._dispatch(CMD_VIDEO_START_CAPTURE, 0, 0, action="start_video")

    def stop_video(self) -> CameraResult:
        refusal = self._require(CAP_CAPTURE_VIDEO, "stop_video", "video capture")
        if refusal:
            return refusal
        return self._dispatch(CMD_VIDEO_STOP_CAPTURE, 0, action="stop_video")

    # -- mode, zoom, focus -------------------------------------------------

    def set_mode(self, mode: str) -> CameraResult:
        key = str(mode).strip().lower()
        if key not in MODE_NAMES:
            return CameraResult(
                False, "set_mode",
                f"Unknown camera mode {mode!r}. Use: {', '.join(sorted(MODE_NAMES))}.")

        refusal = self._require(CAP_HAS_MODES, "set_mode", "mode switching")
        if refusal:
            return refusal
        if key == "survey" and self.capabilities.has(CAP_HAS_SURVEY_MODE) is False:
            return CameraResult(False, "set_mode", unsupported=True,
                                message="This camera does not report a survey mode.")

        return self._dispatch(CMD_SET_CAMERA_MODE, 0, float(MODE_NAMES[key]),
                              action="set_mode")

    def set_zoom(self, level_pct: float) -> CameraResult:
        if not 0.0 <= level_pct <= 100.0:
            return CameraResult(False, "set_zoom",
                                "Zoom is a percentage between 0 and 100.")
        refusal = self._require(CAP_HAS_BASIC_ZOOM, "set_zoom", "zoom control")
        if refusal:
            return refusal
        # ZOOM_TYPE_RANGE = 2, value as a percentage of the full range.
        return self._dispatch(CMD_SET_CAMERA_ZOOM, 2, float(level_pct), action="set_zoom")

    def set_focus(self, level_pct: float) -> CameraResult:
        if not 0.0 <= level_pct <= 100.0:
            return CameraResult(False, "set_focus",
                                "Focus is a percentage between 0 and 100.")
        refusal = self._require(CAP_HAS_BASIC_FOCUS, "set_focus", "focus control")
        if refusal:
            return refusal
        # FOCUS_TYPE_RANGE = 2.
        return self._dispatch(CMD_SET_CAMERA_FOCUS, 2, float(level_pct),
                              action="set_focus")

    # -- extended settings -------------------------------------------------

    def set_exposure_setting(self, setting: str, value: Any) -> CameraResult:
        """ISO, shutter, white balance and friends.

        These are extended parameters rather than MAVLink commands. Without a parameter
        channel to the camera there is nothing to send, and pretending otherwise would
        let an operator believe they had set an exposure they had not.
        """
        key = str(setting).strip().lower()
        if key not in EXTENDED_SETTINGS:
            return CameraResult(
                False, "set_exposure_setting",
                f"Unknown setting {setting!r}. Known: {', '.join(sorted(EXTENDED_SETTINGS))}.")

        return CameraResult(
            False, "set_exposure_setting", unsupported=True,
            details={"parameter": EXTENDED_SETTINGS[key], "requested": value},
            message=(
                f"{key} is set through the extended parameter {EXTENDED_SETTINGS[key]}, "
                "not a MAVLink command, and this driver has no parameter channel to the "
                "camera. Set it on the payload directly. Reporting success here would "
                "leave you believing an exposure was applied that was not."
            ),
        )

    def describe(self) -> dict[str, Any]:
        """What this camera can be asked to do."""
        return self.capabilities.to_dict()
