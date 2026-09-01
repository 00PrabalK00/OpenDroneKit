/* The parts every workspace is built from.
 *
 * Thirteen workspaces composed from a dozen primitives, rather than thirteen hand-built
 * pages. That is not only less code -- it is why selecting a defect in one panel updates
 * the image, the 3D location and the properties in three others without any of them
 * knowing about each other. They all speak through the selection bus below.
 */

/* ------------------------------------------------------------ selection bus */

/** One selection model for the whole application.
 *
 * The rule the spec sets is that selecting anything anywhere updates everything related.
 * Implemented as a topic bus rather than direct wiring: a panel publishes what the user
 * picked and subscribes to what it cares about, so adding a panel never means editing
 * another one.
 */
export class SelectionBus {
  constructor() {
    this.current = {};
    this.listeners = new Map();
  }

  select(kind, value) {
    this.current[kind] = value;
    for (const fn of this.listeners.get(kind) || []) {
      // A throwing subscriber must not stop the others: half-updated panels are worse
      // than one stale one, and a broken inspector should not freeze the map.
      try { fn(value); } catch (error) { console.error(`selection ${kind}`, error); }
    }
    for (const fn of this.listeners.get("*") || []) {
      try { fn(kind, value); } catch (error) { console.error("selection *", error); }
    }
  }

  on(kind, fn) {
    if (!this.listeners.has(kind)) this.listeners.set(kind, []);
    this.listeners.get(kind).push(fn);
    return () => {
      const list = this.listeners.get(kind) || [];
      const index = list.indexOf(fn);
      if (index >= 0) list.splice(index, 1);
    };
  }

  get(kind) { return this.current[kind]; }
}

export const selection = new SelectionBus();

/* -------------------------------------------------------------------- utils */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value != null) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

/* --------------------------------------------------------------------- tree */

export function tree(nodes, { selectKind, onSelect } = {}) {
  const root = el("div", { class: "tree", "data-rows": "tree" });
  const expanded = new Set(nodes.filter((n) => n.expanded !== false).map((n) => n.id));

  const draw = () => {
    root.innerHTML = "";
    const walk = (list, depth) => {
      for (const node of list) {
        const hasChildren = node.children && node.children.length;
        const row = el("div", {
          class: "tree-node" + (node.selected ? " selected" : ""),
          style: `padding-left:${8 + depth * 12}px`,
        }, [
          el("span", { class: "twisty", text: hasChildren ? (expanded.has(node.id) ? "▾" : "▸") : "" }),
          el("span", { class: "ico", text: node.icon || "" }),
          el("span", { class: "label", text: node.label }),
          node.meta ? el("span", { class: "meta", text: node.meta }) : null,
        ]);
        row.onclick = (event) => {
          if (hasChildren && event.offsetX < 16 + depth * 12) {
            expanded.has(node.id) ? expanded.delete(node.id) : expanded.add(node.id);
            draw();
            return;
          }
          root.querySelectorAll(".tree-node").forEach((n) => n.classList.remove("selected"));
          row.classList.add("selected");
          if (selectKind) selection.select(selectKind, node);
          if (onSelect) onSelect(node);
        };
        root.appendChild(row);
        if (hasChildren && expanded.has(node.id)) walk(node.children, depth + 1);
      }
    };
    walk(nodes, 0);
  };
  draw();
  return root;
}

/* -------------------------------------------------------------------- table */

/* Rows on screen are marked, so the shell can tell measured from illustrative.
 *
 * The EXAMPLE DATA banner used to decide by searching the rendered text for a handful of
 * sentinel strings -- the demo organisation's name and three demo site names. That works
 * only for panels whose sample content happens to mention a demo site. Fifty panels do
 * not: a Processing Queue of "#4471 / Feature matching / w-02", a Fleet readout of
 * "Aircraft 6, Available 4", flight telemetry, alerts, battery estimates and maintenance
 * records are all written into the source and contain no sentinel at all. They therefore
 * rendered in connected mode with the banner hidden, which is the precise failure this
 * banner exists to prevent, on the screens where it matters most.
 *
 * A longer sentinel list would have the same shape and fail the same way on the next
 * panel. So the test is structural instead: anything that draws rows marks itself, live()
 * marks its subtree as answered, and content is synthetic exactly when rows exist that no
 * API call produced. Converting a panel to live() removes it from the count automatically,
 * with no list to maintain.
 */
export function table(columns, rows, { selectKind, onSelect } = {}) {
  const head = el("tr", {}, columns.map((c) => el("th", { text: c.title })));
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr", {}, columns.map((c) => {
      const value = typeof c.value === "function" ? c.value(row) : row[c.key];
      if (value instanceof Node) return el("td", {}, [value]);
      return el("td", { class: c.num ? "num" : "", text: value == null ? "—" : String(value) });
    }));
    tr.onclick = () => {
      body.querySelectorAll("tr").forEach((r) => r.classList.remove("selected"));
      tr.classList.add("selected");
      if (selectKind) selection.select(selectKind, row);
      if (onSelect) onSelect(row);
    };
    body.appendChild(tr);
  }
  return el("table", { class: "grid", "data-rows": "table" }, [el("thead", {}, [head]), body]);
}

/* --------------------------------------------------------------- properties */

/** A property inspector. `spec` is a list of {group} or {label, value, unit}. */
export function properties(spec) {
  const body = el("tbody");
  for (const item of spec) {
    if (item.group) {
      body.appendChild(el("tr", { class: "group" }, [el("th", { colspan: "2", text: item.group })]));
      continue;
    }
    const value = item.value instanceof Node
      ? el("td", {}, [item.value])
      : el("td", { text: item.value == null ? "—" : `${item.value}${item.unit ? " " + item.unit : ""}` });
    body.appendChild(el("tr", {}, [el("th", { text: item.label }), value]));
  }
  return el("table", { class: "props", "data-rows": "properties" }, [body]);
}

export function fields(spec, onChange) {
  const wrap = el("div");
  for (const item of spec) {
    if (item.group) {
      wrap.appendChild(el("div", {
        class: "readout", style: "padding-top:8px",
      }, [el("span", { class: "k", text: item.group })]));
      continue;
    }
    const input = item.options
      ? el("select", {}, item.options.map((o) =>
          el("option", { value: o, text: o, ...(o === item.value ? { selected: "" } : {}) })))
      : el("input", { type: item.type || "text", value: item.value ?? "" });
    input.addEventListener("change", () => onChange && onChange(item.key, input.value));
    wrap.appendChild(el("div", { class: "field" }, [
      el("label", { text: item.label }),
      input,
      el("span", { class: "unit", text: item.unit || "" }),
    ]));
  }
  return wrap;
}

/* ----------------------------------------------------------------- readouts */

export function readouts(items) {
  return el("div", { class: "readout-grid", "data-rows": "readouts" }, items.map((item) =>
    el("div", { class: "readout" }, [
      el("span", { class: "k", text: item.k }),
      el("span", { class: "v", style: item.tone ? `color:var(--${item.tone})` : "", text: item.v }),
    ])));
}

export function chip(text, tone = "") {
  return el("span", { class: `chip ${tone}`, text });
}

export function meter(fraction, tone = "") {
  return el("div", { class: `meter ${tone}` }, [
    el("span", { style: `width:${Math.max(0, Math.min(1, fraction)) * 100}%` }),
  ]);
}

/* ------------------------------------------------------------------ console */

export function consoleView(lines) {
  return el("div", { class: "console" }, lines.map((line) =>
    el("div", { class: `line ${line.level || ""}` }, [
      el("span", { class: "t", text: line.t }),
      el("span", { text: line.text }),
    ])));
}

/* ------------------------------------------------------------------- canvas */

/** The central view. `kind` only changes the placeholder wording and overlays --
 *  a real map or 3D viewport mounts into the same element. */
/* Which of the project's layers answers a canvas view.
 *
 * The view names come from the toolbar buttons ("ortho", "dsm", "mesh"); the layer names
 * come from what the reconstruction wrote. Matching is on substrings of the layer name so
 * a run that labels its output "DSM hillshade" still answers a request for the DSM.
 */
const VIEW_LAYERS = {
  // "rgb" is what the toolbar calls the orthomosaic -- the button is labelled RGB and
  // describes itself as "Show the RGB orthomosaic". Omitting it meant the one view most
  // likely to have a product behind it reported having none.
  rgb: ["orthomosaic", "ortho"],
  ortho: ["orthomosaic", "ortho"],
  orthomosaic: ["orthomosaic", "ortho"],
  dsm: ["dsm hillshade", "dsm"],
  dtm: ["dtm"],
  hillshade: ["hillshade"],
  semantic: ["semantic", "classification"],
  thermal: ["thermal"],
  // Deliberately absent: mesh, pointcloud, profile, fused, radiometric, thermal3d.
  // Those are produced as files rather than registered layers, or need a viewer this
  // canvas does not have, so they report "no <view> product yet" -- which is true.
};

/**
 * Render one of the active project's products for this view, or null.
 *
 * Null means the project genuinely has no such layer, which the caller reports as
 * "no <view> product yet" rather than falling back to the bundled example. Showing
 * another project's orthomosaic under this project's title would be the worst
 * available outcome.
 */
async function projectProduct(view) {
  const { tryCall } = await import("./api.js");
  const listed = await tryCall("list_layers");
  const layers = ((listed && listed.layers) || []).filter((l) => l.path);
  if (!layers.length) return null;

  const wanted = VIEW_LAYERS[view] || [view];
  const match = layers.find((l) =>
    wanted.some((w) => String(l.name || l.id).toLowerCase().includes(w)));
  if (!match) return null;

  const preview = await tryCall("raster_preview", match.id);
  if (!preview || !preview.data_uri) return null;
  return {
    src: preview.data_uri,
    note: match.crs_epsg ? `${match.name} — EPSG:${match.crs_epsg}` : String(match.name),
  };
}

export function canvas({ kind = "map", title, note, tools = [], overlays = [], map = null } = {}) {
  const root = el("div", { class: "canvas" });

  if (tools.length) {
    const bar = el("div", { class: "canvas-overlay tl" }, [
      el("div", { class: "canvas-tools" }, tools.map((tool, index) => {
        const button = el("button", {
          class: "tool" + (index === 0 ? " active" : ""),
          title: tool.title,
          text: tool.icon,
        });
        button.onclick = () => {
          bar.querySelectorAll(".tool").forEach((t) => t.classList.remove("active"));
          button.classList.add("active");
          if (tool.onSelect) tool.onSelect();
        };
        return button;
      })),
    ]);
    root.appendChild(bar);
  }

  /* An overlay may carry `html` (a fixed caption -- a legend, a units note) or `node` (an
     element, in practice a live() that asks the application). The distinction matters:
     every overlay that carried a NUMBER used to carry it as `html`, which is how the
     mission map came to print "214 captures, 3.1 km, 18 min, GSD 1.8 cm" over whatever
     project was open. A figure in the corner of the canvas reads as a readout of what is
     on screen, so a fixed one is a fabricated measurement in the most credible position
     the UI has. Fixed `html` is for text that is true of every project or none. */
  for (const overlay of overlays) {
    const box = el("div", { class: `canvas-overlay ${overlay.at || "tr"}` });
    if (overlay.node) box.appendChild(overlay.node);
    else box.innerHTML = overlay.html;
    root.appendChild(box);
  }

  // What the current view mode actually has to show. A map canvas keeps its map; every
  // other view either renders a real product or says which one is missing and how to
  // produce it -- an empty canvas is indistinguishable from a broken one.
  queueMicrotask(async () => {
    const { currentView, currentImage } = await import("./viewstate.js");

    // An image the user picked in a panel wins over everything else: they asked for
    // this exact frame, and showing the basemap instead is the behaviour that made
    // clicking a photograph feel like clicking nothing.
    const picked = currentImage();
    if (picked) {
      root.querySelectorAll(".placeholder").forEach((node) => node.remove());
      const shown = el("img", { class: "canvas-product" });
      shown.setAttribute("src", picked.data_uri);
      shown.setAttribute("alt", picked.name);
      root.appendChild(shown);
      root.appendChild(el("div", { class: "canvas-overlay bl" }, [
        // A survey photograph knows its own pixel size; a rendered map layer does not,
        // and "undefined×undefined" under the picture is worse than no caption at all.
        el("span", {
          class: "chip",
          text: picked.source_width
            ? `${picked.name} — ${picked.source_width}×${picked.source_height}`
            : String(picked.name),
        }),
      ]));
      return;
    }

    const view = currentView();
    // A map canvas keeps its map ONLY while the view is the map. Returning early for
    // every map canvas meant the thermal workspace -- which is a map -- ignored its own
    // RGB, Thermal and Fused buttons entirely.
    if (!view || view === "map") return;

    // The project's OWN products come first.
    //
    // This used to read straight from DATA.products, the bundled example, so the canvas
    // showed the same picture whatever project was open -- and on the Digital Twin, where
    // the example has no entry, it showed nothing at all while a real reconstruction with
    // an orthomosaic, DSM, DTM and mesh sat on disk. raster_preview renders a layer to a
    // PNG the canvas can display, which is exactly what it exists for.
    const product = await projectProduct(view);
    if (!product) {
      // Named, not blank. "Nothing here" and "this has not been produced yet" look the
      // same on a dark canvas, and only one of them tells the user what to do.
      if (view && view !== "map") {
        root.querySelectorAll(".placeholder").forEach((node) => node.remove());
        root.appendChild(el("div", { class: "placeholder" }, [
          el("div", {}, [
            el("div", { class: "big", text: `No ${view} product yet` }),
            el("div", { class: "small", text: "Import a dataset and run Process to produce one." }),
          ]),
        ]));
      }
      return;
    }
    root.querySelectorAll(".placeholder").forEach((node) => node.remove());
    const image = el("img", { class: "canvas-product" });
    image.setAttribute("src", product.src);
    image.setAttribute("alt", product.note);
    root.appendChild(image);
    root.appendChild(el("div", { class: "canvas-overlay bl" }, [
      el("span", { class: "chip", text: product.note }),
    ]));
  });

  root.appendChild(el("div", { class: "placeholder" }, [
    el("div", {}, [
      el("div", { class: "big", text: title || kind }),
      el("div", { class: "small", text: note || "" }),
    ]),
  ]));

  if (map) {
    // Deferred: MapLibre measures its container, and the canvas has no size until the
    // dock has put it in the document. Mounting synchronously gives a map 0px wide.
    queueMicrotask(async () => {
      if (!root.isConnected) return;
      const { mountMap, showMission } = await import("./mapview.js");
      const { DATA } = await import("./demo.js");
      const options = typeof map === "object" ? { ...map } : {};

      // Centre on the site the data actually describes rather than a hardcoded city.
      const mission = DATA.mission || {};
      const footprint = DATA.footprint || {};
      if (!options.centre && footprint.centre) options.centre = footprint.centre;
      if (!options.zoom && footprint.centre) options.zoom = 17;

      const instance = mountMap(root, options);
      if (!instance) return;

      // Draw the planned flight. showMission has existed since the map view was
      // written and nothing ever called it, which is why every canvas was an empty
      // basemap -- the path was computed and then thrown away.
      if (options.mission !== false && (mission.line || []).length > 1) {
        const draw = () => showMission(instance, {
          line: mission.line,
          captures: footprint.captures || [],
        });
        if (instance.isStyleLoaded()) draw();
        else instance.once("load", draw);
      }
    });
  }
  return root;
}

/** Several synchronised viewers side by side -- the AI inspection pattern, where a
 *  finding must be visible as an image, a zoom and a 3D location at the same time. */
export function splitCanvas(views) {
  const root = el("div", { class: "canvas", style: "display:flex;gap:1px;background:var(--line)" });
  for (const view of views) {
    root.appendChild(el("div", {
      class: "canvas",
      style: "margin:0;border:0;border-radius:0;flex:1 1 0",
    }, [
      el("div", { class: "canvas-overlay tl", html: `<strong>${view.title}</strong>` }),
      el("div", { class: "placeholder" }, [
        el("div", {}, [el("div", { class: "small", text: view.note || "" })]),
      ]),
    ]));
  }
  return root;
}
