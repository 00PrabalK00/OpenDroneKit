"""Figures for standards reports, drawn from the report's own input files.

Each figure spec names an input file (hashed into the signed manifest) and how to draw
it. Nothing is drawn that the file does not hold: a thermal figure is the decoded
temperature with its colour scale, a box overlay draws the detector's own boxes, a map
plots the raster and the EXIF camera positions. Specs:

* ``{"type": "photo", "path"}``
* ``{"type": "thermal", "path", "regions": [{"polygon_px", "label"}], "focus": label}``
* ``{"type": "boxes", "path", "boxes": [{"bbox_xyxy", "label"}], "crop_to_boxes": bool}``
* ``{"type": "raster_map", "path", "points": [{"lat", "lon", "highlight"}], "value_label"}``
* ``{"type": "points_map", "points": [...], "background": raster path or None}``
* ``{"type": "placeholder", "text"}``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

INK = "#1d2433"
MUTED = "#5b6475"
ACCENT = "#0f5f66"


def spec_paths(spec: dict[str, Any]) -> list[str]:
    out = [spec.get("path"), spec.get("background")]
    return [str(p) for p in out if p]


def _figure(width_in: float, height_in: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.edgecolor": "#c5cad3", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    return plt, plt.figure(figsize=(width_in, height_in), dpi=200)


def _save(plt, fig, out: Path) -> Path:
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return out


def _photo(spec: dict[str, Any], out: Path, max_px: int = 1800) -> Path:
    from PIL import Image

    with Image.open(spec["path"]) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        out = out.with_suffix(".jpg")
        im.save(out, "JPEG", quality=86)
    return out


def _thermal(spec: dict[str, Any], out: Path) -> Path:
    from matplotlib.patches import Polygon

    from .thermal import load_radiometric_file

    image = load_radiometric_file(spec["path"])
    data = np.asarray(image.celsius, dtype=float)
    # Scale the colours to the outlined modules (not the rail or the sky), so a few
    # kelvin between modules stay visible; pixels outside that range saturate.
    sample = data
    polys = [r["polygon_px"] for r in spec.get("regions") or [] if r.get("polygon_px")]
    if polys:
        from matplotlib.path import Path as MplPath

        yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        pts = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
        inside = np.zeros(data.size, dtype=bool)
        for poly in polys:
            inside |= MplPath(poly).contains_points(pts)
        if inside.any():
            sample = data.ravel()[inside]
    lo, hi = np.nanpercentile(sample, [1, 99])
    plt, fig = _figure(5.6, 4.0)
    ax = fig.add_axes([0.0, 0.0, 0.86, 1.0])
    shown = ax.imshow(data, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_axis_off()
    for region in spec.get("regions") or []:
        poly = region.get("polygon_px")
        if not poly:
            continue
        focus = spec.get("focus")
        strong = focus is None or region["label"].startswith(focus)
        ax.add_patch(Polygon(poly, closed=True, fill=False,
                             edgecolor="#ffffff" if strong else "#9aa3b2",
                             linewidth=1.6 if strong else 0.8,
                             linestyle="-" if strong else "--"))
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        ax.text(min(xs) + 8, min(ys) + 20, region["label"], color="white", fontsize=7.5,
                fontweight="bold" if strong else "normal",
                bbox={"facecolor": "#1d2433", "alpha": 0.7 if strong else 0.45,
                      "edgecolor": "none", "pad": 2})
    cax = fig.add_axes([0.885, 0.06, 0.03, 0.88])
    bar = fig.colorbar(shown, cax=cax, extend="both")
    bar.set_label("Apparent temperature (°C)", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    return _save(plt, fig, out.with_suffix(".png"))


def _boxes(spec: dict[str, Any], out: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(spec["path"]) as im:
        im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    boxes = spec.get("boxes") or []
    width = max(4, im.width // 500)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(28, im.width // 70))
    except OSError:
        font = ImageFont.load_default()
    for box in boxes:
        x0, y0, x1, y1 = box["bbox_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 214, 10), width=width)
        text = box.get("label", "")
        tw = draw.textlength(text, font=font)
        draw.rectangle([x0, max(0, y0 - font.size - 12), x0 + tw + 16, y0],
                       fill=(29, 36, 51))
        draw.text((x0 + 8, max(0, y0 - font.size - 8)), text, fill=(255, 214, 10), font=font)
    if spec.get("crop_to_boxes") and boxes:
        xs = [v for b in boxes for v in (b["bbox_xyxy"][0], b["bbox_xyxy"][2])]
        ys = [v for b in boxes for v in (b["bbox_xyxy"][1], b["bbox_xyxy"][3])]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), (max(ys) - min(ys)) * 4 / 3, im.width / 3) * 0.75
        x0 = int(max(0, min(im.width - 2 * half, cx - half)))
        y0 = int(max(0, min(im.height - 1.5 * half, cy - 0.75 * half - font.size)))
        im = im.crop((x0, y0, int(min(im.width, x0 + 2 * half)),
                      int(min(im.height, y0 + 1.5 * half))))
    im.thumbnail((1600, 1600))
    out = out.with_suffix(".jpg")
    im.save(out, "JPEG", quality=86)
    return out


def _scale_bar(ax, metres_per_unit: float = 1.0) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = (x1 - x0) * metres_per_unit
    nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
    length = max(n for n in nice if n <= span / 4) if span / 4 >= 1 else 1
    bx, by = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.06
    ax.plot([bx, bx + length / metres_per_unit], [by, by], color=INK, linewidth=2.2,
            solid_capstyle="butt")
    ax.text(bx, by + (y1 - y0) * 0.02, f"{length:g} m", color=INK, fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})


def _north(ax) -> None:
    ax.annotate("N", xy=(0.95, 0.93), xytext=(0.95, 0.80), xycoords="axes fraction",
                ha="center", fontsize=9, fontweight="bold", color=INK,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4})


def _project_points(points: list[dict[str, Any]], crs: Any) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([p["lon"] for p in points], [p["lat"] for p in points])
    return np.asarray(xs), np.asarray(ys)


def _raster_map(spec: dict[str, Any], out: Path) -> Path:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(spec["path"]) as src:
        scale = max(1, int(max(src.width, src.height) / 1400))
        data = src.read(1, out_shape=(src.height // scale, src.width // scale),
                        resampling=Resampling.average, masked=True).astype(float)
        b, crs = src.bounds, src.crs
    plt, fig = _figure(6.6, 5.4)
    ax = fig.add_axes([0.0, 0.0, 0.87, 1.0])
    filled = data.filled(np.nan)
    lo, hi = np.nanpercentile(filled, [2, 98])
    shown = ax.imshow(filled, cmap="inferno", vmin=lo, vmax=hi,
                      extent=[b.left, b.right, b.bottom, b.top], interpolation="bilinear")
    points = spec.get("points") or []
    if points:
        xs, ys = _project_points(points, crs)
        hi_mask = np.array([bool(p.get("highlight")) for p in points])
        ax.scatter(xs[~hi_mask], ys[~hi_mask], s=9, c="#7fd3d9", edgecolors=INK,
                   linewidths=0.3, label="camera position", zorder=3)
        if hi_mask.any():
            ax.scatter(xs[hi_mask], ys[hi_mask], s=26, facecolors="none", edgecolors="#ffd60a",
                       linewidths=1.0, label="detector proposal in frame", zorder=4)
        ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.85)
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_xticks([])
    ax.set_yticks([])
    _scale_bar(ax)
    _north(ax)
    cax = fig.add_axes([0.89, 0.08, 0.025, 0.84])
    bar = fig.colorbar(shown, cax=cax)
    bar.set_label(spec.get("value_label") or "value", fontsize=7.5)
    return _save(plt, fig, out.with_suffix(".png"))


def _points_map(spec: dict[str, Any], out: Path) -> Path:
    from pyproj import CRS

    points = spec.get("points") or []
    plt, fig = _figure(6.6, 4.6)
    ax = fig.add_axes([0.0, 0.0, 0.9, 1.0])
    background = spec.get("background")
    if background:
        import rasterio

        with rasterio.open(background) as src:
            data = src.read(1, masked=True).astype(float).filled(np.nan)
            b, crs = src.bounds, src.crs
        shown = ax.imshow(data, cmap="cividis", extent=[b.left, b.right, b.bottom, b.top],
                          interpolation="nearest")
        bar = fig.colorbar(shown, ax=ax, fraction=0.035, pad=0.02)
        bar.set_label("DSM height (m)", fontsize=7.5)
        ax.set_facecolor("#f3f4f6")
    else:
        lat = float(np.mean([p["lat"] for p in points]))
        lon = float(np.mean([p["lon"] for p in points]))
        crs = CRS.from_epsg(32600 + int((lon + 180) // 6) + 1 if lat >= 0 else
                            32700 + int((lon + 180) // 6) + 1)
    xs, ys = _project_points(points, crs)
    ax.scatter(xs, ys, s=12, c=ACCENT, edgecolors="white", linewidths=0.4, zorder=3)
    ax.plot(xs, ys, color=ACCENT, linewidth=0.5, alpha=0.5, zorder=2)
    for mark in spec.get("marks") or []:
        ax.scatter([mark["x"]], [mark["y"]], marker="+", s=60, c="#ffffff", linewidths=1.6,
                   zorder=4)
        ax.annotate(mark["label"], (mark["x"], mark["y"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=7.5, color=INK,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
    pad = max(np.ptp(xs), np.ptp(ys), 20) * 0.12
    if not background:
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.set_facecolor("#f3f4f6")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _scale_bar(ax)
    _north(ax)
    return _save(plt, fig, out.with_suffix(".png"))


def _placeholder(spec: dict[str, Any], out: Path) -> Path:
    plt, fig = _figure(6.6, 3.6)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#eef0f3")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(0.5, 0.5, spec.get("text") or "No image yet", ha="center", va="center",
            color=MUTED, fontsize=11, wrap=True)
    return _save(plt, fig, out.with_suffix(".png"))


_RENDERERS = {"photo": _photo, "thermal": _thermal, "boxes": _boxes,
              "raster_map": _raster_map, "points_map": _points_map,
              "placeholder": _placeholder}


def render(spec: dict[str, Any], out_dir: Path) -> Path:
    """Draw one figure into ``out_dir`` (cached by spec) and return its path."""
    kind = spec.get("type")
    if kind not in _RENDERERS:
        raise ValueError(f"Unknown figure type {kind!r}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]
    stem = out_dir / f"{kind}_{digest}"
    for existing in (stem.with_suffix(".png"), stem.with_suffix(".jpg")):
        if existing.is_file():
            return existing
    return _RENDERERS[kind](spec, stem)
