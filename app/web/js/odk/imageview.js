/* The image-space viewer: one dataset photograph, in its own pixels.
 *
 * Review (findings on a frame), GCP marking and measuring on a photo all work on the
 * photograph rather than the map, so this is an <img> with an SVG overlay, not MapLibre
 * (REBUILD_PLAN 2.5). The picture is the Api's `image_preview` -- a downscaled whole
 * frame, or a close-up window of it (`crop`) -- drawn with object-fit: contain.
 *
 * The contract the cockpit's tests and actions rely on is kept: the picture is
 * `img.canvas-product`, and a click is mapped through the letterboxing to the preview's
 * pixels and then to the source frame's full-resolution pixels, so a ground control mark
 * is a pixel of the photograph, never of the screen. The same mapping places marks:
 * the overlay's viewBox is the source window itself, so a mark at (x, y) in the frame is
 * drawn at (x, y).
 *
 * While a point is being picked a loupe magnifies the preview under the pointer four
 * times. It shows the preview's pixels (what the click maps from); for a finer mark the
 * caller shows a close-up window and picks again, as Mark GCP does.
 */

import { el } from "./components.js";
import { tryCall } from "../workspace/api.js";

const LOUPE_ZOOM = 4;

/* The SVG namespace, taken from a parsed <svg> rather than written out: the sources
   under js/odk name no URL of any kind (tests/test_ui_v1_gallery.py). */
let svgNamespace = null;
function svg(name, attrs = {}) {
  if (!svgNamespace) {
    const holder = document.createElement("div");
    holder.innerHTML = "<svg></svg>";
    svgNamespace = holder.firstChild.namespaceURI;
  }
  const node = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

/**
 * Where the preview is drawn inside the <img> box (object-fit: contain), in CSS px
 * relative to the box: {left, top, width, height, fit}.
 */
export function letterbox(box, naturalWidth, naturalHeight) {
  const fit = Math.min(box.width / naturalWidth, box.height / naturalHeight);
  const width = naturalWidth * fit;
  const height = naturalHeight * fit;
  return { left: (box.width - width) / 2, top: (box.height - height) / 2, width, height, fit };
}

/**
 * A point on the screen (clientX, clientY) as a pixel of the source frame, or null when
 * it is off the picture. `shown` is the current image_preview answer.
 */
export function screenToSource(image, shown, clientX, clientY) {
  const box = image.getBoundingClientRect();
  const nw = image.naturalWidth;
  const nh = image.naturalHeight;
  if (!nw || !nh) return null;
  const drawn = letterbox(box, nw, nh);
  const x = (clientX - box.left - drawn.left) / drawn.fit;
  const y = (clientY - box.top - drawn.top) / drawn.fit;
  if (x < 0 || y < 0 || x > nw || y > nh) return null;
  const width = Number(shown && shown.source_width) || nw;
  const height = Number(shown && shown.source_height) || nh;
  // A close-up covers a window of the frame, not all of it.
  const [cx, cy, cw, ch] = (shown && shown.crop) || [0, 0, width, height];
  return { x: cx + (x * cw) / nw, y: cy + (y * ch) / nh, width, height, previewX: x, previewY: y };
}

/**
 * Mount the viewer into `container`.
 *
 *   ask      Api reader (default tryCall).
 *   notify   (text, kind) for a toast; kind in ok | warn | danger.
 *
 * Returns {node, image, show(name, crop), pickPixel(message), setMarks(marks),
 * setPath(points, {closed}), current(), clear(), destroy()}.
 */
export function mountImageView(container, { ask = tryCall, notify = null } = {}) {
  const say = (text, kind = "ok") => { if (notify) notify(text, kind); };
  let shown = null;
  let marks = [];
  let path = null;

  const image = el("img", { class: "canvas-product", alt: "", draggable: "false" });
  image.hidden = true;
  const overlay = svg("svg", { class: "odk-imageview-overlay", "aria-hidden": "true", preserveAspectRatio: "none" });
  const nameChip = el("span", { class: "odk-map-chip" });
  const sizeChip = el("span", { class: "odk-map-chip odk-num" });
  const cropChip = el("span", { class: "odk-map-chip odk-num" });
  const head = el("div", { class: "odk-imageview-head" }, [nameChip, sizeChip, cropChip]);
  head.hidden = true;
  const hint = el("div", { class: "odk-imageview-hint", role: "status" });
  hint.hidden = true;
  const loupe = el("canvas", { class: "odk-imageview-loupe", width: "144", height: "144", "aria-hidden": "true" });
  loupe.hidden = true;
  const empty = el("div", { class: "odk-imageview-empty", text: "No photograph shown. Choose one from the list." });
  const node = el("div", { class: "odk-imageview", "data-imageview": "" }, [image, overlay, head, hint, loupe, empty]);
  container.appendChild(node);

  /* -- overlay: sits exactly on the drawn picture, in source-frame pixels ---- */
  function layoutOverlay() {
    if (!shown || !image.naturalWidth) { overlay.style.display = "none"; return; }
    const box = image.getBoundingClientRect();
    const host = node.getBoundingClientRect();
    const drawn = letterbox(box, image.naturalWidth, image.naturalHeight);
    overlay.style.display = "";
    overlay.style.left = `${box.left - host.left + drawn.left}px`;
    overlay.style.top = `${box.top - host.top + drawn.top}px`;
    overlay.style.width = `${drawn.width}px`;
    overlay.style.height = `${drawn.height}px`;
    const [cx, cy, cw, ch] = shown.crop || [0, 0, shown.source_width, shown.source_height];
    overlay.setAttribute("viewBox", `${cx} ${cy} ${cw} ${ch}`);
    drawOverlay(drawn, cw);
  }

  function drawOverlay(drawn, windowWidth) {
    overlay.replaceChildren();
    // One screen pixel in source pixels: marks keep a constant on-screen size.
    const unit = windowWidth / Math.max(1, drawn.width);
    if (path && path.points.length > 1) {
      const points = path.points.map(([x, y]) => `${x},${y}`).join(" ");
      overlay.appendChild(svg(path.closed ? "polygon" : "polyline",
        { points, class: "odk-imageview-path", ...(path.closed ? {} : { fill: "none" }) }));
    }
    for (const mark of marks) {
      const g = svg("g", { class: `odk-imageview-mark${mark.tone ? ` odk-imageview-mark--${mark.tone}` : ""}`,
        "data-mark": mark.label || "" });
      g.appendChild(svg("circle", { cx: mark.x, cy: mark.y, r: 5 * unit }));
      if (mark.label) {
        const text = svg("text", { x: mark.x + 9 * unit, y: mark.y - 7 * unit, "font-size": 11 * unit });
        text.textContent = mark.label;
        g.appendChild(text);
      }
      overlay.appendChild(g);
    }
  }

  const resize = typeof ResizeObserver === "function" ? new ResizeObserver(() => layoutOverlay()) : null;
  if (resize) resize.observe(node);
  image.addEventListener("load", layoutOverlay);

  /* ------------------------------------------------------------------ show */

  /** Draw an image_preview answer already in hand. Resolves once the picture loaded. */
  function display(preview) {
    shown = preview;
    const loaded = new Promise((resolve) => {
      image.addEventListener("load", resolve, { once: true });
      image.addEventListener("error", resolve, { once: true });
    });
    image.src = preview.data_uri;
    image.alt = `Photograph ${preview.name}`;
    image.hidden = false;
    empty.hidden = true;
    head.hidden = false;
    nameChip.textContent = preview.name;
    sizeChip.hidden = !(preview.source_width && preview.source_height);
    sizeChip.textContent = `${preview.source_width} × ${preview.source_height} px`;
    cropChip.hidden = !preview.crop;
    if (preview.crop) {
      const [cx, cy, cw, ch] = preview.crop;
      cropChip.textContent = `close-up ${cw} × ${ch} at ${cx}, ${cy}`;
    }
    node.dataset.image = preview.name;
    return loaded.then(() => { layoutOverlay(); return preview; });
  }

  /** Fetch one dataset photograph (or a close-up window of it) and draw it. */
  async function show(name, crop = null) {
    const preview = crop ? await ask("image_preview", name, 1400, crop) : await ask("image_preview", name);
    if (!preview || preview.ok === false || !preview.data_uri) {
      say(`${name}: ${(preview && preview.error) || "no preview available."}`, "warn");
      return null;
    }
    return display(preview);
  }

  /* ------------------------------------------------------------------ pick */

  function drawLoupe(event) {
    const at = screenToSource(image, shown, event.clientX, event.clientY);
    if (!at) { loupe.hidden = true; return; }
    const ctx = loupe.getContext("2d");
    const size = loupe.width;
    const span = size / LOUPE_ZOOM;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(image, at.previewX - span / 2, at.previewY - span / 2, span, span, 0, 0, size, size);
    const host = node.getBoundingClientRect();
    const side = loupe.offsetWidth || size;
    let left = event.clientX - host.left + 24;
    let top = event.clientY - host.top + 24;
    if (left + side > host.width) left = event.clientX - host.left - side - 24;
    if (top + side > host.height) top = event.clientY - host.top - side - 24;
    loupe.style.left = `${Math.max(0, left)}px`;
    loupe.style.top = `${Math.max(0, top)}px`;
    loupe.hidden = false;
  }

  /**
   * Wait for one click on the photograph: {x, y, width, height} in the source frame's
   * pixels, or null on Esc or a click off the picture.
   */
  async function pickPixel(message) {
    for (let i = 0; i < 200 && !(shown && image.naturalWidth); i += 1) {
      await new Promise((r) => setTimeout(r, 25));
    }
    if (!shown || !image.naturalWidth) return null;
    hint.textContent = message || "Click the point on the photograph (Esc to cancel).";
    hint.hidden = false;
    node.setAttribute("data-picking", "");
    return new Promise((resolve) => {
      const done = (value) => {
        image.removeEventListener("click", onClick);
        image.removeEventListener("mousemove", drawLoupe);
        image.removeEventListener("mouseleave", hideLoupe);
        document.removeEventListener("keydown", onKey);
        node.removeAttribute("data-picking");
        hint.hidden = true;
        loupe.hidden = true;
        resolve(value);
      };
      const hideLoupe = () => { loupe.hidden = true; };
      const onKey = (event) => { if (event.key === "Escape") done(null); };
      const onClick = (event) => {
        const at = screenToSource(image, shown, event.clientX, event.clientY);
        done(at ? { x: at.x, y: at.y, width: at.width, height: at.height } : null);
      };
      image.addEventListener("click", onClick);
      image.addEventListener("mousemove", drawLoupe);
      image.addEventListener("mouseleave", hideLoupe);
      document.addEventListener("keydown", onKey);
    });
  }

  return {
    node,
    image,
    show,
    display,
    pickPixel,
    /** Marks in source-frame pixels: [{x, y, label, tone}] (tone: ok | warn | danger). */
    setMarks(next) { marks = Array.isArray(next) ? next : []; layoutOverlay(); },
    /** A measured line or outline on the photograph, in source-frame pixels. */
    setPath(points, { closed = false } = {}) {
      path = Array.isArray(points) && points.length ? { points, closed } : null;
      layoutOverlay();
    },
    /** The image_preview answer on screen, or null. */
    current: () => shown,
    clear() {
      shown = null; marks = []; path = null;
      image.removeAttribute("src");
      image.hidden = true; head.hidden = true; empty.hidden = false;
      delete node.dataset.image;
      layoutOverlay();
    },
    destroy() {
      if (resize) resize.disconnect();
      node.remove();
    },
  };
}

/**
 * The shell's image-view provider (js/odk/shell.js: shell.provide("imageView", ...)):
 * (host, preview) => node, drawing a preview the shell already fetched. The view is
 * kept on the node as `odkImageView` for marks and paths.
 */
export function imageViewProvider(options = {}) {
  return (_host, preview) => {
    const holder = document.createElement("div");
    holder.className = "odk-imageview-host";
    const view = mountImageView(holder, options);
    if (preview && preview.data_uri) view.display(preview);
    holder.odkImageView = view;
    return holder;
  };
}
