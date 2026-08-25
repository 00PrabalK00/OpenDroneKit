# The workspace cockpit

`app/web/workspace.html` is the operations interface: fourteen workspaces built from one
dockable framework, arranged around a canvas that keeps most of the screen.

Open it with the desktop shell (`python main.py`), or serve `app/web/` and visit
`/workspace.html`.

This document described the cockpit as the interface for some time while `app/shell.py`
opened `index.html` instead, so the UI documented here was one no user could reach and
the one they did reach was documented nowhere. `python main.py` now opens the cockpit.

The older single-screen shell is still there behind `ODK_UI=classic`, because it is the
more completely wired of the two while the cockpit's workspaces are connected to the Api
one at a time, and removing a working screen before its replacement is finished loses
capability quietly.

## What is real on screen, and what is not

The cockpit shows the structural sample until it can talk to the application. That
sample is deliberately impossible to mistake for a survey: sites are named DEMO,
coordinates are Null Island, clocks sit at the epoch, and the wording matches
`core/demo_mode.py` so the desktop demo and the API demo agree about what synthetic
looks like.

A banner across the top says so, and it is shown in every state except `connected` --
the only state in which what you are looking at came from the application. It used to
be a chip at the end of the status bar, which was clipped off screen at 1600px, so the
one thing declaring the data synthetic was the first thing to disappear.

`?demo=1` keeps the demo even when a bridge is available, for demonstrating the product
on a machine that has real projects on it.

## The idea

A workspace is not a page. It is an arrangement of panels pointed at the project you are
already working on. Moving from planning to flight to verification does not change the
survey — it changes which instruments you are looking through, which is why the project,
the selection and the coordinate system all survive the switch.

```
┌─ global navigation ────────────────────────────────────────────────┐
├─ contextual toolbar (changes per workspace) ───────────────────────┤
│ ┌────────┬──────────────────────────────────┬──────────────────┐   │
│ │ left   │            CANVAS                │ right            │   │
│ │ trees  │   map · 3D · image · thermal     │ properties       │   │
│ │ layers │                                  │ inspector        │   │
│ │        ├──────────────────────────────────┤ telemetry        │   │
│ │        │ bottom: timeline · logs · queue  │                  │   │
│ └────────┴──────────────────────────────────┴──────────────────┘   │
├─ status bar ───────────────────────────────────────────────────────┤
```

## Workspaces

| # | Workspace | Canvas | What it answers |
|---|---|---|---|
| 1 | Home | Operations map | What is happening across every project |
| 2 | Projects | Project extent | What this project is and what it has produced |
| 3 | Mission Planning | Mission map / 3D | What will be flown, and what it will cost |
| 4 | Flight | Live map | Where the aircraft is and what it is doing |
| 5 | Verification | Planned vs actual | Did we capture what we planned |
| 6 | Processing | Reconstruction | How the photogrammetry is progressing |
| 7 | Digital Twin | 3D model | What the asset looks like, over time |
| 8 | AI Inspection | Image + detection + 3D | What the models found, and where it is |
| 9 | Thermal | RGB / thermal / fused | What is hot, and by how much |
| 10 | Measurements | Ortho / terrain | How big, how far, how much |
| 11 | Fleet | Fleet map | What can fly, and what needs service |
| 12 | Reports | Live preview | What the client receives |
| 13 | Developers | API console | How to integrate |
| 14 | Settings | — | Units, CRS, models, keyboard |

## Using it: a run from an empty window to a report

Three things frame every workspace. The **top row** switches which instruments point at
your project -- it does not change which project you are on. The **second row** is the
toolbar, and it changes per workspace: that is where the verbs live. The **bottom right**
is where every action reports. If you click something and no message appears there, that
is a bug worth reporting rather than a feature you have not found.

Two habits make the panels make sense:

1. **Select first, then act.** Clicking a row tells the toolbar what to work on. Cancel
   acts on the selected job, Accept/Reject/Flag on the selected finding, Log Maintenance
   on the selected aircraft. Nothing has to be typed twice.
2. **In a tree, click the label rather than the ▸.** The arrow expands; the text selects.

| # | Workspace | Do this | What happens |
|---|---|---|---|
| 1 | Projects | `New Project`, choose a folder, name it | Created and made active |
| 2 | Projects | `Import`, choose a folder of images | Imported, and made the active dataset. Reselected automatically next launch |
| 3 | Mission Planning | Edit altitude and overlap on the right, then `Plan` | Each edit is confirmed, and the planner runs **with those values** |
| 4 | Mission Planning | `Save`, then `Export` | Stored in the project, then written in the flight-controller formats |
| 5 | Processing | `Process` for reconstruction, or `Start` for the full pipeline | Both confirm first, then run as background jobs reporting progress |
| 6 | Processing | Select a row in the queue, then `Cancel` | Cancels the job you selected |
| 7 | Verification | `Match Captures`, select a finding, then `Accept` / `Reject` / `Flag` | Moves its status and records who moved it |
| 8 | Thermal, AI Inspection | `RGB`, `Thermal`, `Semantic` | The canvas draws that product, or says which one has not been produced |
| 9 | Fleet | `Add Aircraft`, `Add Battery`, `Add Pilot` | Real rows, in the same database the web service reads |
| 10 | Fleet | Select an aircraft, then `Log Maintenance` | Records it and resets the service clock |
| 11 | Reports | `Generate Report` | Builds it, or refuses with a checklist of what is missing first |
| 12 | Projects | `Share` | Issues a token in a blocking dialog. Copy it there: only its hash is stored |

### Two refusals that are working correctly

`Generate Report` refusing with a list is the report engine declining to emit a document
with empty sections. The checklist is the useful part.

A view saying "no thermal product yet" means exactly that. It is not a broken canvas; it
is a canvas with nothing true to draw, and it names what to run to produce one.

### Starting it

```bash
python main.py                 # the cockpit
ODK_UI=classic python main.py  # the older single-screen shell
```

The UI is served over 127.0.0.1 on an ephemeral port rather than opened as a file. ES
modules fetched from `file://` have origin "null" and the webview refuses them, which
produces a blank window and no error anywhere. Nothing leaves the machine: the server
binds loopback only.

## Panels

Every panel supports the same operations, because they are the same component:

- **Resize** — drag the splitter between regions
- **Move** — drag a panel's header into another region
- **Collapse** — `▾` in the header
- **Hide** — `✕`, restored from *Layout ▾*
- **Tab-stack** — panels with several views show tabs
- **Pop out** — `⧉` opens the panel as its own window for a second monitor

Layout is saved per workspace and restored on return. *Layout ▾ → Reset* undoes it.

Saving is keyed by workspace and panel id rather than by position, so a panel added in a
later release appears at its default size instead of scrambling a layout you arranged.

## Selection is shared

Selecting anything publishes it to one selection bus, and every panel that cares
subscribes. Select a finding in the findings table and the source image, the 3D context
and the finding inspector all update — none of them knows the table exists.

That is what makes it an application rather than a set of dashboards, and it is why
adding a panel never means editing another one.

## Keyboard

| Key | Action |
|---|---|
| `Ctrl/⌘ K` | Command palette — workspaces, projects, findings, commands |
| `1` – `9` | Switch workspace |
| `F11` | Full-screen canvas |
| `Ctrl/⌘ B` | Toggle side panels |
| `F` | Fit view |

## Colour means something

Colour is never decorative here. A red border always means something is wrong, or it
stops meaning anything at all.

| Colour | Meaning |
|---|---|
| Blue | Interaction, selection, the active workspace |
| Green | Healthy operational state |
| Amber | Warning, degraded, needs attention |
| Red | Error, safety-critical action, serious defect |
| Purple | Thermal and semantic visualisation only |

Safety-critical actions — Abort, Land, RTL, Manual Override — are styled apart from
routine ones in the toolbar, and a test asserts that rule holds. An action that stops an
aircraft should never be one careless click from Save.

## What it is not, yet

The status bar says **sample data — not connected to a project**, permanently, until a
project is attached. The shell is the framework and the arrangement; the numbers in it
are illustrative structure, not measurements. Nothing here should be read as a survey
result, and the frame of the application says so rather than leaving you to work it out.

Toolbar actions are not wired to the API. Pressing one reports that it is not wired
rather than appearing to work — a button that silently does nothing is worse than one
that admits it.

The canvas regions are panel-mounted placeholders. MapLibre is already vendored in
`app/web/vendor/` and the existing hub uses it; mounting a real map, 3D viewport or image
viewer into a canvas element is the next step and does not change the framework.

## Extending it

Add a workspace by adding an object to `WORKSPACES` in `js/workspace/workspaces.js`:

```js
const myWorkspace = {
  id: "thing", title: "Thing",
  toolbar: ["New", "|", "Process"],
  left:   [{ id: "thing.tree", title: "Items", render: () => tree([...]) }],
  canvas: () => canvas({ title: "Thing", tools: MAP_TOOLS }),
  right:  [{ id: "thing.props", title: "Properties", render: () => properties([...]) }],
  bottom: [{ id: "thing.log", title: "Log", render: () => consoleView([...]) }],
};
```

Nothing else changes: navigation, the toolbar, docking, persistence, the palette and the
selection bus all pick it up. `tests/test_workspace_ui.py` will then execute it — every
workspace is mounted through the real dock in Node and every panel rendered, because a
panel that throws disappears silently and leaves a gap where telemetry should be.
