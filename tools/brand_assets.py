"""Regenerate OpenDroneKit brand rasters from the SVG masters.

The SVG files in ``app/web/assets/brand/`` are the only primary artwork. Every PNG/ICO
under ``derived/`` and the repository ``logo.png`` are generated from them by this
script, so a brand change is: replace the SVG masters, run this, commit.

    python tools/brand_assets.py                       # rebuild every raster
    python tools/brand_assets.py --import-masters DIR  # copy mark-*.svg from DIR, cropped tight
    python tools/brand_assets.py --lockup-font FONT    # rebuild wordmark-*.svg from a font file
    python tools/brand_assets.py --export DIR          # large lockup/mark PNGs for other teams

Rasterising needs ``resvg-py`` and ``Pillow``. Rebuilding the lockup also needs
``fonttools``, ``brotli`` (for .woff2) and ``uharfbuzz``. None of these are app
dependencies; install them into a tools environment, not the app environment.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "app" / "web" / "assets" / "brand"
DERIVED = BRAND / "derived"

# Palette. Mark colours are measured from the supplied source artwork.
CHARCOAL = "#1D2225"   # mark + text on light backgrounds
BLUE_LIGHT = "#006BFD"  # heading arrow on light backgrounds
MIST = "#E6EAF0"       # mark + text on dark backgrounds
BLUE_DARK = "#2F86FF"  # heading arrow on dark backgrounds
UI_DARK = "#0D1015"    # --bg-app in app/web/css/tokens.css; app-icon tile colour

# Lockup geometry, in units of the mark's height.
CAP_HEIGHT = 0.44      # cap height of the wordmark text
GAP = 0.30             # space between the mark's right edge and the first letter
TRACKING = -0.012      # letter spacing, in em
WORD = "OpenDroneKit"


# ----------------------------------------------------------------------------- SVG io
def _svg_parts(svg: str) -> tuple[list[float], str]:
    """Return (viewBox, inner markup) of a flat SVG document."""
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    inner = svg[svg.index(">", svg.index("<svg")) + 1: svg.rindex("</svg>")]
    return vb, inner.strip()


def render(svg: str | Path, *, width: int | None = None, height: int | None = None) -> Image.Image:
    import resvg_py

    text = Path(svg).read_text(encoding="utf-8") if isinstance(svg, Path) else svg
    kwargs = {"width": width} if width else {"height": height}
    data = resvg_py.svg_to_bytes(svg_string=text, **kwargs)
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def fit(svg: Path, box: int) -> Image.Image:
    """Render ``svg`` as large as fits in a ``box`` x ``box`` square, centred."""
    vb, _ = _svg_parts(svg.read_text(encoding="utf-8"))
    w, h = vb[2], vb[3]
    im = render(svg, width=box) if w >= h else render(svg, height=box)
    out = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    out.alpha_composite(im, ((box - im.width) // 2, (box - im.height) // 2))
    return out


# ------------------------------------------------------------------------ masters
def import_masters(src: Path) -> None:
    """Copy mark-{light,dark,mono-white}.svg from ``src`` with the viewBox cropped to ink."""
    for name in ("mark-light", "mark-dark", "mark-mono-white"):
        svg = (src / f"{name}.svg").read_text(encoding="utf-8")
        vb, inner = _svg_parts(svg)
        scale = 4096 / vb[2]
        box = render(svg, width=4096).getbbox()
        x0 = vb[0] + box[0] / scale
        y0 = vb[1] + box[1] / scale
        x1 = vb[0] + box[2] / scale
        y1 = vb[1] + box[3] / scale
        x0, y0 = int(x0) - 1, int(y0) - 1
        w, h = int(x1 + 2) - x0, int(y1 + 2) - y0
        (BRAND / f"{name}.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {w} {h}">'
            f"{inner}</svg>\n", encoding="utf-8")
        print(f"master  {name}.svg  viewBox {x0} {y0} {w} {h}")


# ------------------------------------------------------------------------- lockup
def _text_paths(font_path: Path, text: str) -> tuple[str, float, float, float]:
    """Outline ``text``: (path data in font units, y-up), advance, cap height, units/em."""
    import uharfbuzz as hb
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(str(font_path))
    upem = font["head"].unitsPerEm
    cap = getattr(font["OS/2"], "sCapHeight", 0) or upem * 0.727
    blob = hb.Blob.from_file_path(str(font_path)) if not str(font_path).endswith(".woff2") else None
    if blob is None:  # harfbuzz needs sfnt data; decompress woff2 through fontTools
        buf = io.BytesIO()
        font.flavor = None
        font.save(buf)
        blob = hb.Blob(buf.getvalue())
        font = TTFont(io.BytesIO(buf.getvalue()))
    hbfont = hb.Font(hb.Face(blob))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hbfont, buf, {"kern": True, "liga": True})
    glyphs = font.getGlyphSet()
    order = font.getGlyphOrder()
    track = TRACKING * upem
    pen = SVGPathPen(glyphs)
    x = 0.0
    for i, (info, pos) in enumerate(zip(buf.glyph_infos, buf.glyph_positions)):
        name = order[info.codepoint]
        glyphs[name].draw(TransformPen(pen, (1, 0, 0, 1, x + pos.x_offset, pos.y_offset)))
        x += pos.x_advance + (track if i < len(buf.glyph_infos) - 1 else 0)
    return pen.getCommands(), x, float(cap), float(upem)


def build_lockup(font_path: Path) -> None:
    """Write wordmark-light.svg / wordmark-dark.svg: the mark + WORD, text as outlines."""
    d, advance, cap, _ = _text_paths(font_path, WORD)
    for variant, text_fill in (("light", CHARCOAL), ("dark", MIST)):
        vb, inner = _svg_parts((BRAND / f"mark-{variant}.svg").read_text(encoding="utf-8"))
        mh = 1000.0                                  # mark height in lockup units
        ms = mh / vb[3]
        mw = vb[2] * ms
        k = CAP_HEIGHT * mh / cap                    # font units -> lockup units
        tx = mw + GAP * mh
        baseline = mh / 2 + CAP_HEIGHT * mh / 2      # cap band centred on the mark
        total_w = tx + advance * k
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {mh:.0f}">'
            f'<title>OpenDroneKit</title>'
            f'<g transform="scale({ms:.6f}) translate({-vb[0]:.2f} {-vb[1]:.2f})">{inner}</g>'
            f'<path fill="{text_fill}" transform="translate({tx:.2f} {baseline:.2f}) '
            f'scale({k:.6f} {-k:.6f})" d="{d}"/></svg>\n'
        )
        (BRAND / f"wordmark-{variant}.svg").write_text(svg, encoding="utf-8")
        print(f"lockup  wordmark-{variant}.svg  {total_w:.0f} x {mh:.0f}  ({font_path.name})")


# ------------------------------------------------------------------------ rasters
def tile(size: int, *, mark_frac: float, radius_frac: float) -> Image.Image:
    """The dark mark on a UI-dark tile. radius_frac 0 gives a full-bleed square."""
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ss = 8  # draw the rounded rect supersampled, then downsample, for clean corners
    mask = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size * ss - 1, size * ss - 1), radius=int(size * ss * radius_frac), fill=255)
    bg = Image.new("RGBA", (size, size), UI_DARK)
    bg.putalpha(mask.resize((size, size), Image.LANCZOS))
    out.alpha_composite(bg)
    m = max(1, round(size * mark_frac))
    mark = fit(BRAND / "mark-dark.svg", m)
    out.alpha_composite(mark, ((size - m) // 2, (size - m) // 2))
    return out


def build_rasters() -> list[Path]:
    DERIVED.mkdir(parents=True, exist_ok=True)
    written: dict[Path, Image.Image] = {}
    # Browser-tab icons: a dark tile reads on both light and dark tab strips.
    written[DERIVED / "favicon-16.png"] = tile(16, mark_frac=0.88, radius_frac=0.19)
    written[DERIVED / "favicon-32.png"] = tile(32, mark_frac=0.80, radius_frac=0.19)
    # iOS masks its own corners, so the touch icon is full-bleed.
    written[DERIVED / "apple-touch-icon-180.png"] = tile(180, mark_frac=0.62, radius_frac=0)
    # Web-manifest / launcher icon: rounded tile, transparent corners.
    written[DERIVED / "app-icon-512.png"] = tile(512, mark_frac=0.64, radius_frac=0.2)
    # Toolbar mark (displayed at 28 css px) and Workspace top-bar mark (16 css px).
    written[DERIVED / "mark-on-dark-64.png"] = fit(BRAND / "mark-dark.svg", 64)
    # Hub header lockup, displayed 28 css px tall; 56 px is its 2x raster.
    written[DERIVED / "wordmark-on-dark-56.png"] = render(BRAND / "wordmark-dark.svg", height=56)
    # README logo: light lockup, transparent.
    written[ROOT / "logo.png"] = render(BRAND / "wordmark-light.svg", height=160)
    for path, im in written.items():
        im.save(path, optimize=True)
        print(f"raster  {path.relative_to(ROOT).as_posix()}  {im.width}x{im.height}")
    # Multi-size .ico: browser /favicon.ico and the desktop window icon.
    ico = DERIVED / "favicon.ico"
    big = tile(256, mark_frac=0.70, radius_frac=0.2)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = {s: tile(s, mark_frac=0.88 if s <= 16 else 0.80 if s <= 32 else 0.70,
                      radius_frac=0.19 if s <= 32 else 0.2) for s in sizes}
    big.save(ico, format="ICO", sizes=[(s, s) for s in sizes],
             append_images=[frames[s] for s in sizes if s != 256])
    print(f"icon    {ico.relative_to(ROOT).as_posix()}  {sizes}")
    # The REST service ships without app/web (see infrastructure/docker), so it
    # carries its own copies for the API docs pages.
    api_static = ROOT / "services" / "api" / "static"
    api_static.mkdir(exist_ok=True)
    shutil.copyfile(ico, api_static / "favicon.ico")
    print("copy    services/api/static/favicon.ico")
    return [*written, ico]


def export(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for variant in ("light", "dark"):
        src = BRAND / f"wordmark-{variant}.svg"
        shutil.copyfile(src, dest / src.name)
        render(src, height=512).save(dest / f"wordmark-{variant}.png", optimize=True)
        print(f"export  {dest / src.name}, wordmark-{variant}.png (512 px tall)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import-masters", type=Path, help="directory holding mark-*.svg")
    ap.add_argument("--lockup-font", type=Path, help="font file (ttf/otf/woff2) for the wordmark")
    ap.add_argument("--export", type=Path, help="also write large lockup PNG/SVGs here")
    args = ap.parse_args()
    if args.import_masters:
        import_masters(args.import_masters)
    if args.lockup_font:
        build_lockup(args.lockup_font)
    build_rasters()
    if args.export:
        export(args.export)


if __name__ == "__main__":
    main()
