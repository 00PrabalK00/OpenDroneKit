"""How a model opens for the person you send it to.

A shared reconstruction opens wherever the viewer's camera happened to start: usually
above the site, looking down, with every clip off and every annotation showing. The
recipient is a client or an engineer who was sent one thing to look at, and the first
thing they have to do is find it.

A saved view is the answer to "open it like this": a camera pose, which clips are on,
whether annotated images are shown, and whether the viewer starts in facade mode. Named,
so one model can carry "as delivered", "north facade" and "roof detail" at once.

Deliberately NOT a render or a screenshot. A view is a set of numbers pointing at the
live model, so the recipient can still orbit away from it, measure, and open the source
photographs. A picture would be smaller and would answer a different question.

Stored beside the clips, in JSON, for the same reason: an operator should be able to read
what a deliverable was set up to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence


class ViewRefused(ValueError):
    """A view that cannot be saved, or would not open anywhere useful."""


@dataclass
class SavedView:
    """A named way of opening a model."""

    name: str
    #: Where the camera sits, in model coordinates.
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: What it looks at. Stored as a target rather than a direction so the view survives
    #: being read by a viewer with different Euler conventions -- a heading/pitch pair
    #: means nothing without knowing the order they are applied in.
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Vertical field of view in degrees.
    fov_deg: float = 50.0
    #: Clip names that are ON in this view. A view referring to a clip that has since
    #: been deleted simply does not apply it; see `resolve_clips`.
    visible_clips: list[str] = field(default_factory=list)
    facade_mode: bool = False
    show_annotations: bool = True
    is_default: bool = False
    created_utc: str = ""

    def distance_m(self) -> float:
        """How far the camera is from what it is looking at."""
        return math.dist(self.position, self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": list(self.position),
            "target": list(self.target),
            "fov_deg": float(self.fov_deg),
            "visible_clips": list(self.visible_clips),
            "facade_mode": bool(self.facade_mode),
            "show_annotations": bool(self.show_annotations),
            "is_default": bool(self.is_default),
            "created_utc": self.created_utc,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "SavedView":
        return SavedView(
            name=str(raw.get("name", "")),
            position=tuple(float(v) for v in raw.get("position", (0, 0, 0))),   # type: ignore[arg-type]
            target=tuple(float(v) for v in raw.get("target", (0, 0, 0))),       # type: ignore[arg-type]
            fov_deg=float(raw.get("fov_deg", 50.0)),
            visible_clips=[str(c) for c in (raw.get("visible_clips") or [])],
            facade_mode=bool(raw.get("facade_mode", False)),
            show_annotations=bool(raw.get("show_annotations", True)),
            is_default=bool(raw.get("is_default", False)),
            created_utc=str(raw.get("created_utc", "")),
        )


def validate(view: SavedView) -> None:
    """Refuse views that would open nowhere.

    A camera sitting exactly on its target has no direction to look in, and a viewer
    given one either shows a blank screen or silently substitutes a default -- which is
    worse, because the recipient sees A view and assumes it is THE view.
    """
    if not view.name.strip():
        raise ViewRefused("A view needs a name; that is how it is opened again.")
    if view.distance_m() < 1e-6:
        raise ViewRefused(
            "The camera is on top of its target, so the view has no direction. "
            "Move the camera back from the point it is looking at."
        )
    if not 1.0 <= view.fov_deg <= 179.0:
        raise ViewRefused(f"A field of view of {view.fov_deg} degrees cannot be rendered.")


def resolve_clips(view: SavedView, existing: Sequence[str]) -> tuple[list[str], list[str]]:
    """Which of the view's clips still exist, and which have gone.

    A clip can be deleted after a view referring to it was saved. Applying the view then
    has two wrong answers: fail to open it, or open it silently showing more of the model
    than intended. Reporting the missing names lets the caller open the view AND say what
    it could not apply.
    """
    have = set(existing)
    applied = [name for name in view.visible_clips if name in have]
    missing = [name for name in view.visible_clips if name not in have]
    return applied, missing


class ViewStore:
    """The views saved against one model."""

    FILENAME = "views.json"

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / self.FILENAME

    def load(self) -> list[SavedView]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [SavedView.from_dict(entry) for entry in raw.get("views", [])]

    def save(self, views: Sequence[SavedView]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"views": [v.to_dict() for v in views]}, indent=2),
            encoding="utf-8",
        )

    def add(self, view: SavedView) -> list[SavedView]:
        validate(view)
        views = [v for v in self.load() if v.name != view.name]
        view.created_utc = view.created_utc or datetime.now(timezone.utc).isoformat()
        # Exactly one default, always. Two defaults is an unanswerable question at open
        # time, and the viewer would pick by file order -- which is to say, arbitrarily.
        if view.is_default:
            for other in views:
                other.is_default = False
        views.append(view)
        self.save(views)
        return views

    def remove(self, name: str) -> list[SavedView]:
        views = self.load()
        remaining = [v for v in views if v.name != name]
        if len(remaining) == len(views):
            raise ViewRefused(f"No view named {name!r}.")
        self.save(remaining)
        return remaining

    def set_default(self, name: str) -> list[SavedView]:
        views = self.load()
        if not any(v.name == name for v in views):
            raise ViewRefused(f"No view named {name!r}.")
        for view in views:
            view.is_default = view.name == name
        self.save(views)
        return views

    def default(self) -> SavedView | None:
        """The view a share link should open at, or None to leave the viewer alone."""
        for view in self.load():
            if view.is_default:
                return view
        return None
