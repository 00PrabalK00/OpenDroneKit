"""Record what a page in ui_live_bridge.Browser shows, as a video and a contact sheet.

A second DevTools connection to the same page takes a JPEG screenshot as often as the
renderer can give one (a few per second under SwiftShader). Each frame keeps the time
it was taken, and ffmpeg's concat demuxer holds every frame for exactly the time until
the next one, so the video plays at the speed the flight happened rather than at
whatever rate the screenshots arrived. Nothing is drawn on the frames: they are the
page as the operator saw it.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class ScreenRecorder:
    def __init__(self, page_ws_url: str, frames_dir: Path, *, quality: int = 88,
                 max_fps: float = 10.0, browser_pid: int | None = None) -> None:
        import websocket

        # The browser's memory and the machine's, every few seconds: software WebGL at
        # 1920x1080 is heavy, and a renderer that runs out of memory stops answering.
        self.browser_pid = browser_pid
        self.memory: list[dict[str, float]] = []
        self._next_memory = 0.0

        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.min_interval_s = 1.0 / max_fps
        self.frames: list[tuple[float, Path]] = []
        self.errors: list[str] = []
        self._socket = websocket.create_connection(page_ws_url, timeout=30, suppress_origin=True)
        self._id = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ui-recorder", daemon=True)
        self.started = time.monotonic()
        self._thread.start()

    def _capture(self) -> bytes:
        self._id += 1
        wanted = self._id
        self._socket.send(json.dumps({"id": wanted, "method": "Page.captureScreenshot",
                                      "params": {"format": "jpeg", "quality": self.quality}}))
        while True:
            message = json.loads(self._socket.recv())
            if message.get("id") == wanted:
                if "error" in message:
                    raise RuntimeError(str(message["error"]))
                return base64.b64decode(message["result"]["data"])

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.monotonic()
            try:
                data = self._capture()
            except Exception as exc:  # noqa: BLE001 - a dropped frame is not a failed test
                self.errors.append(f"{type(exc).__name__}: {exc}")
                if len(self.errors) > 50:
                    return
                time.sleep(0.5)
                continue
            if self.browser_pid and began >= self._next_memory:
                self._next_memory = began + 5.0
                self._sample_memory(began)
            path = self.frames_dir / f"frame_{len(self.frames):06d}.jpg"
            path.write_bytes(data)
            self.frames.append((began - self.started, path))
            spare = self.min_interval_s - (time.monotonic() - began)
            if spare > 0:
                self._stop.wait(spare)

    def _sample_memory(self, now: float) -> None:
        try:
            import psutil

            root = psutil.Process(self.browser_pid)
            rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
            self.memory.append({"t_s": round(now - self.started, 1),
                                "browser_mb": round(rss / 2**20),
                                "system_available_mb": round(psutil.virtual_memory().available / 2**20)})
        except Exception:  # noqa: BLE001 - no psutil, or the browser is gone
            pass

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.ended = time.monotonic()
        try:
            self._socket.close()
        except Exception:  # noqa: BLE001
            pass

    # -- output ---------------------------------------------------------------

    def write_video(self, target: Path, fps: int = 30) -> Path | None:
        """H.264 at the frames' real timing; None when ffmpeg is not installed."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or not self.frames:
            return None
        listing = self.frames_dir / "frames.ffconcat"
        lines = ["ffconcat version 1.0"]
        end = getattr(self, "ended", time.monotonic()) - self.started
        for index, (t, path) in enumerate(self.frames):
            following = self.frames[index + 1][0] if index + 1 < len(self.frames) else end
            lines.append(f"file '{path.name}'")
            lines.append(f"duration {max(0.001, following - t):.4f}")
        lines.append(f"file '{self.frames[-1][1].name}'")
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-vf", f"fps={fps},format=yuv420p",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-movflags", "+faststart", str(target)],
                       check=True, cwd=str(self.frames_dir))
        return target

    def write_contact_sheet(self, target: Path, columns: int = 4, rows: int = 4,
                            marks: list[tuple[float, str]] | None = None) -> Path | None:
        """A grid of frames spread over the recording, each captioned with its time.

        `marks` are (seconds, step) pairs from the test; a tile is captioned with the
        last step reached by its time, so the sheet reads as the flight's sequence.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None
        if not self.frames:
            return None
        count = min(columns * rows, len(self.frames))
        picks = [self.frames[round(i * (len(self.frames) - 1) / max(1, count - 1))]
                 for i in range(count)]
        tile_w, tile_h, caption_h = 480, 270, 26
        sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + caption_h)), (16, 18, 22))
        draw = ImageDraw.Draw(sheet)
        try:
            from PIL import ImageFont

            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        for index, (t, path) in enumerate(picks):
            with Image.open(path) as frame:
                tile = frame.convert("RGB").resize((tile_w, tile_h))
            x, y = (index % columns) * tile_w, (index // columns) * (tile_h + caption_h)
            sheet.paste(tile, (x, y + caption_h))
            step = ""
            for mark_t, label in marks or []:
                if mark_t <= t:
                    step = label
            minutes, seconds = divmod(int(t), 60)
            draw.text((x + 6, y + 6), f"{minutes:02d}:{seconds:02d}  {step}", fill=(230, 230, 230), font=font)
        target = Path(target)
        sheet.save(target, quality=90)
        return target

    def summary(self) -> dict[str, Any]:
        duration = (getattr(self, "ended", time.monotonic()) - self.started)
        return {"frames": len(self.frames), "duration_s": round(duration, 1),
                "mean_fps": round(len(self.frames) / duration, 2) if duration else 0.0,
                "capture_errors": len(self.errors), "last_errors": self.errors[-3:],
                "browser_peak_mb": max((m["browser_mb"] for m in self.memory), default=None),
                "system_available_min_mb": min((m["system_available_mb"] for m in self.memory),
                                               default=None)}
