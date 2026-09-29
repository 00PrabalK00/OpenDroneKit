# Brand assets

The SVG files in the parent `brand/` directory are the primary artwork. Everything in
this `derived/` directory, `services/api/static/favicon.ico` and the repository
`logo.png` is generated from them by `tools/brand_assets.py`. Do not edit the rasters
by hand; change the SVGs and regenerate.

## Masters (`brand/`)

| File | Use |
| --- | --- |
| `mark-light.svg` | Mark for light backgrounds: charcoal `#1D2225`, heading arrow blue `#006BFD`. |
| `mark-dark.svg` | Mark for dark backgrounds (the UI, `#0D1015`): `#E6EAF0`, arrow `#2F86FF`. |
| `mark-mono-white.svg` | Single-colour white mark, for photos and one-colour print. |
| `wordmark-light.svg` | Lockup: mark + "OpenDroneKit", charcoal text, for light backgrounds. |
| `wordmark-dark.svg` | Lockup for dark backgrounds, `#E6EAF0` text. |

The mark is a four-rotor quad outline with a heading arrow; it has no built-in
wordmark. The mark SVGs were vectorised from the supplied source artwork and
cropped to their ink bounds by `--import-masters`.

Lockup: the text is Inter SemiBold (600) converted to outlines, so no font is needed
to display or rasterise it. Cap height is 0.44 of the mark height and centred on it,
the gap between mark and text is 0.30 of the mark height, tracking is -0.012 em, and
the font's own kerning is applied. The constants are at the top of
`tools/brand_assets.py`.

## Derived files

| File | Pixels | Used by |
| --- | --- | --- |
| `favicon-16.png`, `favicon-32.png` | 16, 32 | `<link rel="icon">` in index/hub/workspace.html. A dark tile, so it reads on light and dark tab strips. |
| `favicon.ico` | 16-256 | Desktop window/taskbar icon (`app/shell.py`); copied to `services/api/static/` for the API docs pages. |
| `apple-touch-icon-180.png` | 180 | Full-bleed dark square; iOS rounds the corners itself. |
| `app-icon-512.png` | 512 | `site.webmanifest`; rounded dark tile, transparent corners. |
| `mark-on-dark-64.png` | 64 | Cockpit toolbar mark (28 css px) and Workspace top-bar mark (16 css px). |
| `wordmark-on-dark-56.png` | 304x56 | Hub header lockup, shown 28 css px tall (2x raster). |
| `logo.png` (repo root) | 866x160 | README; light lockup on transparent. The README swaps in `wordmark-dark.svg` on dark themes. |

## Regenerating

```
pip install resvg-py pillow                          # rasterising
pip install fonttools brotli uharfbuzz               # only for --lockup-font

python tools/brand_assets.py                         # rebuild every raster above
python tools/brand_assets.py --import-masters DIR    # take new mark-*.svg from DIR first
python tools/brand_assets.py --lockup-font FONT      # rebuild wordmark-*.svg from a font
python tools/brand_assets.py --export DIR            # 512 px tall lockup PNGs + SVGs for other teams
```

Install these into a tools environment; they are not application dependencies.
The current files were produced with resvg-py 0.5.0 (resvg 0.48.1), Pillow 12.3 and
`--lockup-font inter-latin-600-normal.woff2` (Inter 4.001, SIL Open Font License).
