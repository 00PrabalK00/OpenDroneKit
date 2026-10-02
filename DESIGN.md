---
name: OpenDroneKit
description: Offline-first drone survey and inspection workbench. Desk "Survey Instrument" with a switchable "Field Daylight" mode.
colors:
  canvas: "#0E1116"
  panel: "#11151B"
  control: "#151A21"
  selected-row: "#18202B"
  map-well: "#0A0D11"
  hairline: "#232A33"
  hairline-strong: "#2E3743"
  ink-light: "#E6EAF0"
  ink-secondary: "#B4BCC8"
  label-grey: "#8A94A3"
  survey-blue: "#2F86FF"
  survey-blue-hover: "#6BA8FF"
  measured-green: "#4ADE80"
  caution-amber: "#F5A524"
  fault-red: "#F06A6A"
  synthetic-violet: "#C084FC"
  field-paper: "#FFFFFF"
  field-ink: "#0A0A0A"
  field-panel: "#F4F4F4"
  safety-yellow: "#FFD400"
  field-selection: "#0047B3"
  field-ok: "#15803D"
  field-danger: "#B91C1C"
typography:
  hero:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    letterSpacing: "-0.02em"
  value:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 500
  title:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    letterSpacing: "-0.01em"
  body:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "11px"
    fontWeight: 500
    letterSpacing: "0.12em"
  data:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "11px"
    fontWeight: 400
    fontFeature: "tnum"
  field-body:
    fontFamily: "Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 400
  field-label:
    fontFamily: "Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    letterSpacing: "0.04em"
  field-number:
    fontFamily: "Atkinson Hyperlegible, system-ui, sans-serif"
    fontSize: "64px"
    fontWeight: 700
rounded:
  none: "0px"
  chip: "2px"
spacing:
  sp-1: "4px"
  sp-2: "8px"
  sp-3: "12px"
  sp-4: "16px"
  sp-5: "24px"
components:
  button-primary:
    backgroundColor: "{colors.survey-blue}"
    textColor: "{colors.canvas}"
    rounded: "{rounded.none}"
    height: "36px"
    padding: "0 16px"
  button-primary-hover:
    backgroundColor: "{colors.survey-blue-hover}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink-light}"
    rounded: "{rounded.none}"
    height: "36px"
    padding: "0 16px"
  input:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-light}"
    rounded: "{rounded.none}"
    height: "30px"
  chip:
    textColor: "{colors.ink-light}"
    rounded: "{rounded.chip}"
    height: "22px"
    typography: "{typography.data}"
  tag-simulated:
    backgroundColor: "{colors.synthetic-violet}"
    textColor: "{colors.field-ink}"
    rounded: "{rounded.chip}"
  header-desk:
    backgroundColor: "{colors.panel}"
    height: "48px"
  footer-desk:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.label-grey}"
    height: "28px"
    typography: "{typography.data}"
  header-field:
    backgroundColor: "{colors.field-ink}"
    textColor: "{colors.field-paper}"
    height: "72px"
  tab-field-active:
    backgroundColor: "{colors.safety-yellow}"
    textColor: "{colors.field-ink}"
    height: "88px"
  button-field-action:
    backgroundColor: "{colors.safety-yellow}"
    textColor: "{colors.field-ink}"
    rounded: "{rounded.none}"
    height: "64px"
  button-field-commit:
    backgroundColor: "{colors.field-ink}"
    textColor: "{colors.field-paper}"
    rounded: "{rounded.none}"
    height: "64px"
---

# Design System: OpenDroneKit

The approved v1 system (owner sign-off 2026-09-29). Source of truth for values:
`app/web/css/odk/tokens.css`; the brief: `brand/ui-v1/DESIGN_SPEC.md` and the `Sys*.dc.html`
artboards. This file was written from the shipped shell (U2) and tokens; where it and
`tokens.css` disagree, `tokens.css` wins and this file is stale.

## Overview

**Creative North Star: "The Survey Instrument"** (desk), with **"Field Daylight"** as
its outdoor mode.

The desk interface reads like a calibrated instrument rather than a dashboard: dark,
quiet surfaces, one blue for what the operator does and for the geometry they measure,
hairline rules instead of boxes, and data set in mono so a coordinate looks like a
coordinate. The map or the photograph is the subject; the frame (48 px header, 264 px
panel, 312 px inspector, 28 px footer) stays still around it and never decorates.

Field Daylight is a different layout, not a theme: white paper, black ink, 3–4 px rules,
safety yellow for the one action on the screen, 64 px targets and Atkinson Hyperlegible,
so it reads on a tablet in direct sun with gloves on.

Both modes carry the project's honesty rule into the pixels: every figure comes from an
Api answer, a missing value says "not reported", and anything simulated or from the demo
carries the violet tag because the record says so.

**Key Characteristics:**
- Flat, square, hairline: radius 0 (2 px only on chips), 1 px rules, no shadows, no
  gradients except the turbo elevation ramp.
- One accent per mode: survey blue (desk), safety yellow (field action) plus field blue
  for selection.
- Mono for data, labels and the status bar; sans for UI and prose.
- The frame states the truth about itself: project, CRS, scale, vehicle, jobs and
  network are read live, or the footer says they are absent.

## Colors

A near-black instrument palette with one saturated blue, four status hues, and a violet
reserved for synthetic data; the Field palette is paper, ink and yellow.

### Primary
- **Survey Blue** (#2F86FF): primary buttons, the active tab underline, selected rows'
  2 px inset, focus rings, measured geometry on the map. Hover is **Survey Blue Light**
  (#6BA8FF), also used for links.
- **Safety Yellow** (#FFD400, Field): the single action on a Field screen and the active
  Field tab. Ink text, never white.

### Status
- **Measured Green** (#4ADE80): ok, "local · no network". Field: #15803D.
- **Caution Amber** (#F5A524): warnings, a network use the footer is reporting.
- **Fault Red** (#F06A6A): danger, a lost link, a refusal. Field: #B91C1C.
- **Synthetic Violet** (#C084FC): SIMULATED and DEMO RECORD tags and the example-data
  banner. Nothing else may be this colour.

### Neutral
- **Canvas** (#0E1116) app background; **Panel** (#11151B) header, side panels, footer,
  popovers, palette; **Control** (#151A21) inputs' alternate fill, hover rows, toasts;
  **Selected Row** (#18202B); **Map Well** (#0A0D11) behind maps and photographs.
- **Hairline** (#232A33) every 1 px rule; **Hairline Strong** (#2E3743) hover edges,
  chip and dialog outlines.
- **Ink** (#E6EAF0) text; **Ink Secondary** (#B4BCC8) live data in the footer;
  **Label Grey** (#8A94A3) labels and idle text, never below 11 px.
- Field: **Paper** (#FFFFFF), **Ink** (#0A0A0A), **Field Panel** (#F4F4F4).

### Named Rules
**The Violet Means Synthetic Rule.** #C084FC appears only on a value the Api marked
`is_simulated` / `demo`, or on the demo banner. A screen never decides on its own that
something is simulated.

**The One Blue Rule.** Survey blue is for the operator's action, the selection and the
measured geometry. It is not a decoration colour.

## Typography

**UI Font:** IBM Plex Sans 400/500/600 (system-ui fallback)
**Data / Label Font:** IBM Plex Mono 400/500
**Field Font:** Atkinson Hyperlegible 400/700 (numbers too)

All vendored under `app/web/vendor/fonts`; nothing is fetched.

**Character:** an engineering pairing: Plex Sans for anything read as language, Plex
Mono for anything read as a measurement, an id or a key.

### Hierarchy
- **Hero** (600, 30 px, -0.02em): the one big figure on a card (area, volume).
- **Value** (500, 20 px): stat values.
- **Title** (600, 15 px): the wordmark, dialog titles, the palette input.
- **Body** (400, 13 px, 1.4): UI text, tabs, rows, menu items.
- **Small** (400, 12 px): field labels, prose hints, toast details.
- **Label** (500 mono, 11 px, +0.12em, uppercase): section labels, palette groups.
- **Data** (400 mono, 11 px, tabular): footer, coordinates, keys, kbd hints.
- Field: body 18 px, labels 16 px bold uppercase +0.04em, titles 22 px bold, numbers
  48–64 px bold.

### Named Rules
**The Mono Is Data Rule.** Mono sets coordinates, ids, keys and the status bar. A
sentence (an action's description, an error) is set in sans even inside a mono context.

## Layout

Desk is a fixed frame on a CSS grid: rows `48 / auto (banner) / 1fr / 28`, columns
`264 / 1fr / 312`. Both side panels collapse (edge handles, Ctrl/⌘ B, persisted), which
the 1100 px minimum window needs; at ≤1320 px the header search folds to its 30 px icon.
Spacing is the 4/8/12/16/24 scale; header items sit 16 px apart, footer segments 24 px.

Field is rows `72 / auto / 1fr / 88`, columns `1fr / 404` (map, then the side panel on
the right), with a five-tab bottom bar (Plan, Fly, Process, Analyse, Report). Home,
Review, Fleet and Settings exist only on Desk; opening one from Field switches mode.

## Elevation & Depth

Flat. There are no shadows. Depth is tonal (canvas → panel → control → selected row) and
drawn with 1 px hairlines; overlays (palette, dialogs) sit on a 72 % map-well scrim and
are separated by a Hairline Strong outline, not a shadow. Field uses 3–4 px ink rules for
the same job.

### Named Rules
**The Rule Not Shadow Rule.** Separate with a rule or a tone step. A drop shadow, a glow
or a glass blur is outside this system.

## Shapes

Square. Radius 0 everywhere except chips and the kbd hint (2 px). Selection is a 2 px
inset bar (desk) or a 6 px inset bar / a full blue fill (field), never a coloured
left-border card. Status is a dot plus words, never colour alone.

## Components

### Buttons
- **Shape:** square (0).
- **Primary:** survey blue fill, canvas text, 600, 36 px high, 16 px side padding.
- **Secondary:** transparent, 1 px hairline, ink text; hover fills Control and lifts the
  edge to Hairline Strong. The header "Field mode" toggle is a 30 px secondary.
- **Ghost / Danger / Commit:** text only (underline on hover); red outline; ink fill.
- **Field:** 64 px; yellow = the action, black = commit, 3 px ink outline = alternate.

### Inputs
- 30 px, canvas fill, 1 px hairline; focus = blue edge + 1 px inset blue. Field: 64 px,
  3 px ink, focus 2 px inset field blue.

### Navigation
- **Desk header tabs:** 13 px, label grey; active = ink + 2 px inset blue underline.
  Settings is the gear at the far right with the same active underline.
- **Field tab bar:** 88 px, 20 px bold with a 24 px stroke icon; active = yellow fill.

### Status bar (desk footer)
- 28 px, mono 11 px, label grey; live values in Ink Secondary; segments: cursor (+UTM),
  CRS, scale, vehicle (+ SIMULATED tag), jobs, screen segment, last message (opens the
  session log), network ("● local · no network" in green only while true).

### Command palette
- Panel fill, Hairline Strong outline, 48 px input at 15 px, groups in mono labels
  (Workspaces, Projects, Actions, Places, Shortcuts), 36 px rows, selected row = Selected
  Row tone + 2 px blue inset. Field: white sheet, 4 px ink, 64 px rows, selected = blue
  fill, white text.

### Toasts
- Control fill, full 1 px outline and a status dot (never a left-border card); title and
  detail split at the first ": ". At most four stacked; every one is kept in the session
  log.

## Do's and Don'ts

### Do:
- **Do** read every figure from an Api answer and render "not reported" for a missing one.
- **Do** give every verb button `data-action="<ACTIONS key>"` and every UI-only button
  `data-control`.
- **Do** keep Field targets ≥ 64 px and Desk labels ≥ 11 px.
- **Do** use tokens from `tokens.css`; Field values live under `[data-mode="field"]`.

### Don't:
- **Don't** copy a number, a font link or a `/_blob/` image from the artboards.
- **Don't** add shadows, gradients (other than the elevation ramp), rounded cards or
  glass.
- **Don't** use violet for anything but synthetic data, or blue for decoration.
- **Don't** fetch anything from the network in the interface's own assets.
