# v1 interface: architecture and contract

This is the contract every rebuild unit (U1 to U15) works to. It is frozen in U00: a
unit that needs to change it changes this file in the same commit and says why.

Design source: `brand/ui-v1/DESIGN_SPEC.md` and the approved artboards (desk direction A
"Survey Instrument", Field mode direction C "Field Daylight", approved 2026-09-29).
Plan: `brand/ui-v1/REBUILD_PLAN.md`. Inventory: [`INVENTORY.md`](INVENTORY.md), generated
from [`inventory.json`](inventory.json).

## How it opens

The new interface is `app/web/odk.html`. It opens only when asked for:

```
python main.py --ui odk          # or
set ODK_UI=odk                   # Windows; ODK_UI=odk elsewhere
```

`app/shell.py` maps `odk` to `odk.html` in `UI_PAGES`. The default stays `classic` until
the switchover unit (U15). `odk.html?gallery` renders every design-system component in
both modes; `?mode=field` or `?mode=desk` forces a mode for one load.

## Decisions

- **Plain ES modules, no framework, no build step.** Everything is served raw from
  `app/web`; the guard tests read JavaScript source with regular expressions, which a
  compiled framework would break.
- **Reuse the cockpit's honest data layer; replace its layout.** `api.js`, `live.js`,
  `modal.js`, `actions.js`, `draw.js`, `viewer3d.js`, `mapview.js`, `demo.js` and the
  `el`/`table`/selection-bus parts of `primitives.js` stay where they are during the
  rebuild and move to `js/core/` in U15. `dock.js`, `workspaces.js`, `shell.js` and the
  old CSS are replaced.
- **Offline.** No font, script, stylesheet or image is fetched from the network. Fonts are
  vendored in `app/web/vendor/fonts` (OFL, licences alongside, sources in `SOURCES.txt`).
  The artboards' Google Fonts links and `/_blob/` images must never be copied in.

## File layout

```
app/web/odk.html                      entry page (renamed index.html in U15)
app/web/css/odk/tokens.css            design tokens: :root = desk (A); [data-mode="field"] = field (C)
app/web/css/odk/fonts.css             @font-face -> ../../vendor/fonts/*.woff2 only
app/web/css/odk/base.css              reset, body, text utilities
app/web/css/odk/components.css        buttons, inputs, chips, tags, toast, tabs, table, rows,
                                      stats, popover, map furniture (desk geometry)
app/web/css/odk/desk.css              frame grid 48 / 264 / 1fr / 312 / 28, collapsible panels
app/web/css/odk/field.css             field layout and component overrides (72 / 88 / 64 px, 3-4 px rules)
app/web/vendor/fonts/                 IBM Plex Sans 400/500/600, Plex Mono 400/500,
                                      Atkinson Hyperlegible 400/700 + OFL texts + SOURCES.txt
app/web/js/odk/main.js                boot: mode, gallery or frame
app/web/js/odk/components.js          component factories (U1)
app/web/js/odk/gallery.js             odk.html?gallery
app/web/js/odk/shell.js               class Shell (U2)
app/web/js/odk/frame.js               header, tabs, search, field toggle, footer, collapse (U2)
app/web/js/odk/palette.js             command palette incl. search_places (U2)
app/web/js/odk/menu.js                window.odk.onMenu: every menu id -> action or handler (U2)
app/web/js/odk/registry.js            core ACTIONS + screens/*/actions.js, refuses duplicates (U2)
app/web/js/odk/map/                   furniture.js, theme.js (U3)
app/web/js/odk/imageview.js           image-space viewer, keeps img.canvas-product (U3)
app/web/js/odk/screens/index.js       screen table, old-id aliases, loadAll()
app/web/js/odk/screens/<ws>/          surface.js, model.js, desk.js, actions.js
                                      ws in home | plan | fly | process | analyse | review | fleet | report | settings
app/web/js/odk/screens/field/         plan.js, fly.js, process.js, analyse.js, report.js
```

Screen-specific CSS, when a screen needs any, goes in `screens/<ws>/<ws>.css`; nobody but
U1 edits `components.css`.

### Screen modules

| File | Holds | Rule |
|---|---|---|
| `model.js` | `CALLS` (the Api methods the screen reads) and `derive(results)` | Shared by the desk renderer and the Field renderer, so both modes show the same values. |
| `desk.js` | `render({left, main, inspector}, context)` | Builds DOM from `components.js` only. |
| `surface.js` | `SCREEN`, `PLACEMENTS` | Machine-readable; tests read it instead of `toolbar: [`. |
| `actions.js` | `ACTIONS` new to this screen | The core `js/workspace/actions.js` is frozen; screens add verbs only here. |
| `field/<ws>.js` | `render({left, main}, context)`, re-exports `CALLS` from `../<ws>/model.js` | Field screens never fetch on their own. |

U00 ships these as stubs that render nothing. A stub must never render a sample value.

## Shell contract (frozen)

`js/odk/shell.js` (U2) exports `class Shell`. The browser harnesses
(`cockpit_harness.py`, `cockpit_actions_in_a_browser.py` and the registry-cited tests
that use them) drive the application only through this surface, so a ported test needs
its import changed and nothing else.

| Member | Contract |
|---|---|
| `new Shell(root)` | Mounts the frame into `root`; reads the persisted mode. |
| `open(id)` | Opens a workspace. Accepts the new ids and every old cockpit id (alias table below). Returns a promise that resolves when the screen has rendered. |
| `runAction(key)` | Runs an `ACTIONS` key from the merged registry with `actionContext()`. Resolves to the action's result (`{message}`, `{skipped}`, `{error}`...). Unknown key: resolves `{error: "unknown action <key>"}`, never throws. |
| `actionContext()` | The `ctx` object actions receive (`prompt`, `confirm`, `choose`, `toast`, `refresh`, `open`, `resetLayout`, map and selection access...). Same members the cockpit provided; U2 lists them in the shell source. |
| `toast(text, kind)` | Shows a toast; `kind` in `ok | warn | danger | running`. Returns the node. |
| `confirmAll` | Boolean. When true, `ctx.confirm()` answers yes without a dialog (tests only). |
| `refreshState()` | Re-reads `get_state` and re-renders live panels (`REFRESH_EVENT`). |
| `setMode("desk" \| "field")` | Sets `<html data-mode>`, persists it in localStorage `odk.mode`, re-renders the frame for the mode, re-applies the map theme. |

Also kept from the cockpit: `img.canvas-product` on the image view, and `draw.js`'s
`addVertex` / `finishDrawing` / `canDraw` reachable from the shell's map.

### Old workspace ids

`Shell.open()` accepts these, mapped in `screens/index.js` (`WORKSPACE_ALIASES`):

| Old cockpit id | New screen |
|---|---|
| `planning` | `plan` |
| `flight` | `fly` |
| `processing`, `verification` | `process` |
| `measurements`, `twin`, `thermal` | `analyse` |
| `inspection` | `review` |
| `reports` | `report` |
| `projects` | `home` |
| `developers` | `settings` |

`home`, `fleet` and `settings` keep their ids. Header tabs: Home, Plan, Fly, Process,
Analyse, Review, Fleet, Report; Settings is a header icon button. Field tabs: Plan, Fly,
Process, Analyse, Report; Home, Review, Fleet and Settings are reached by switching back
to Desk.

## surface.js schema

```js
export const SCREEN = "plan";

export const PLACEMENTS = [
  { action: "Import Boundary", where: "plan.aoi", label: "Import KML" },
  { action: "Delete Measurement", where: "analyse.measurements.row", label: "Delete", modes: ["desk"] },
];
```

| Field | Required | Meaning |
|---|---|---|
| `action` | yes | An `ACTIONS` key in the merged registry (core plus `screens/*/actions.js`). |
| `where` | yes | `<screen>.<region>[.<sub>]`: the region of the screen that renders the button. Regions are named after the artboard sections (`plan.aoi`, `plan.inspector.export`, `fly.commandbar`...). |
| `label` | yes | The visible label from the artboard. Tests never look buttons up by it. |
| `modes` | no | `["desk"]`, `["field"]` or both (default both where a Field screen exists, else desk). |
| `inventory` | no | The `inventory.json` row id this placement satisfies, when it is not obvious from `action`. |

A workspace unit is done when every `inventory.json` row of its workspace has a
placement, and a live test sees `[data-action="<action>"]` rendered in `where` for each.

## The data-action rule

Every button that runs a verb renders `data-action="<ACTIONS key>"`. Tests, the command
palette and the menu bridge find buttons by key, never by label (labels follow the
artboards and change; keys are stable because every browser test drives by key).

A button that only changes the interface (dismiss a toast, open a popover, switch a tab,
zoom the map) renders `data-control="<name>"` instead. `components.button()`,
`iconButton()` and `popover()` items throw if given neither, so the rule cannot be
skipped by accident.

## Modes

`<html data-mode="desk|field">`, persisted in localStorage (`odk.mode`). Tokens swap
under `[data-mode="field"]`; the attribute works on any element, which is how the gallery
shows both modes on one page. Field is a different layout, not only a theme: 72 px black
header, 88 px bottom tab bar, every touch target at least 64 px (asserted by computed
style in `tests/test_ui_v1_gallery.py`), 3-4 px ink rules, yellow for the one action per
screen, black for commit, outline for the alternate.

## Honesty rules

These carry the project's no-fabrication rule into the interface.

1. **No numerals in `screens/**` outside a `live()` render.** A number on screen comes
   from an Api result in that screen's `model.js`. A literal digit in a rendered value
   slot under `js/odk/screens/` is a defect. The artboards are full of real Aukerman
   figures; copying markup from them pastes those figures in, which is exactly what this
   rule exists to catch. (The U00 stubs contain no digits at all.)
2. **Tags come from Api fields.** `SIMULATED` is drawn when the record has
   `is_simulated: true`; `DEMO RECORD` when it has `demo: true` (or `is_demo: true`). Use
   `components.simulatedTag(record)` / `demoTag(record)`, which return nothing for any
   other record. A screen never decides on its own that a value is simulated, and never
   omits a tag the Api set.
3. **Missing is said, not filled.** A value the Api does not report renders "not
   reported" (`components.stat()` does this for `null`). A bracketed artboard placeholder
   such as `[battery %]` renders "not reported".
4. **Unbacked design elements** (inventory rows marked `unbacked`) ship only with their
   backend (owner decision 2026-09-29), never as a fake.
5. **Offline.** No `http(s)://` in `odk.html`, `css/odk` or `js/odk` except the opt-in
   basemap table U3 will add; no `fonts.googleapis`; no `/_blob/`; every `@font-face`
   resolves to a file under `vendor/fonts`.

## Guards added in U00 / U1

| Test | Guards |
|---|---|
| `tests/test_ui_v1_inventory.py` | Every native menu item, every cockpit `ACTIONS` key and every public `Api` method has an inventory row; every row has a new home; `INVENTORY.md` is generated from the JSON; every screen stub exists; `UI_PAGES` maps `odk` without changing the default. |
| `tests/test_ui_v1_gallery.py` | `odk.html?gallery` renders every component in desk and field mode; field targets are at least 64 px; text tokens meet 4.5:1; all fonts load from `vendor/fonts`; no request leaves 127.0.0.1; every screen module imports; the blank frame opens. |

## Baseline

[`BASELINE_FEATURE_STATUS.md`](BASELINE_FEATURE_STATUS.md) records `tools/feature_status.py`
on `ship/fix-all` before any rebuild code. Every unit's definition of done includes "no
feature downgraded against this baseline".
