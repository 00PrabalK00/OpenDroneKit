/* The command palette (Ctrl/⌘ K, or type in the header search).
 *
 * Five groups, in this order: Workspaces, Projects, Actions, Places, Shortcuts.
 *
 *   Actions    every key in the merged ACTIONS registry. This is the guarantee that no
 *              verb is ever unreachable, whatever a screen chooses to draw: a feature
 *              with no button yet still runs from here (REBUILD_PLAN 1.1).
 *   Projects   list_projects, read when the palette opens; picking one opens it.
 *   Places     search_places, asked only when the operator chooses "Search places":
 *              geocoding is the one part of the toolkit that may reach outward, so it
 *              never fires on a keystroke. Picking a result flies the map.
 *
 * Every row that runs a verb carries data-action=<key>; every other row data-control.
 */

import { el, icon } from "./components.js";
import { tryCall, lastError, connected } from "../workspace/api.js";
import { actionKeys, describe } from "./registry.js";

const GROUPS = ["Workspaces", "Projects", "Actions", "Places", "Shortcuts"];

/** How well `text` matches `query`: 0 = no match, higher is better. */
export function score(text, query) {
  if (!query) return 1;
  const t = String(text || "").toLowerCase();
  const q = query.toLowerCase();
  if (t === q) return 100;
  if (t.startsWith(q)) return 80;
  if (new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`).test(t)) return 60;
  if (t.includes(q)) return 40;
  // Every word of the query somewhere in the text, in any order ("export kml").
  const words = q.split(/\s+/).filter(Boolean);
  if (words.length > 1 && words.every((w) => t.includes(w))) return 20;
  return 0;
}

export class Palette {
  constructor(shell) {
    this.shell = shell;
    this.node = null;
    this.projects = null;
    this.places = null;
    this.placeNote = "";
    this.searching = false;
    this.active = 0;
  }

  get isOpen() { return Boolean(this.node); }

  async open(initial = "") {
    if (this.node) {
      this.input.value = initial || this.input.value;
      this.draw();
      this.input.focus();
      return this;
    }
    this.restoreFocus = document.activeElement;
    this.places = null;
    this.placeNote = "";
    this.active = 0;

    this.input = el("input", {
      class: "odk-palette-input", type: "text", spellcheck: "false", autocomplete: "off",
      role: "combobox", "aria-expanded": "true", "aria-controls": "odk-palette-list",
      "aria-autocomplete": "list", "aria-label": "Search commands, workspaces, projects and places",
      placeholder: "Type a command, workspace, project or place",
    });
    this.input.value = initial || "";
    this.list = el("div", { class: "odk-palette-list", id: "odk-palette-list", role: "listbox",
      "aria-label": "Results" });
    this.status = el("div", { class: "odk-palette-foot", "aria-live": "polite" });
    const panel = el("div", { class: "odk-palette", role: "dialog", "aria-modal": "true",
      "aria-label": "Command palette" }, [
      el("div", { class: "odk-palette-head" }, [
        icon("search", { size: 16 }), this.input,
        el("kbd", { class: "odk-kbd", text: "Esc" }),
      ]),
      this.list,
      this.status,
    ]);
    this.node = el("div", { class: "odk-palette-backdrop", "data-palette": "" }, [panel]);
    this.node.addEventListener("mousedown", (event) => { if (event.target === this.node) this.close(); });
    this.input.addEventListener("input", () => { this.active = 0; this.places = null; this.draw(); });
    this.input.addEventListener("keydown", (event) => this.onKey(event));
    // Mounted inside the shell root so it takes the root's mode (desk or field).
    (this.shell.root || document.body).appendChild(this.node);
    this.draw();
    this.input.focus();
    this.input.select();

    // Projects are read once per opening, so the list is what is on disk now.
    if (connected()) {
      const listing = await tryCall("list_projects");
      if (!this.node) return this;
      this.projects = listing && Array.isArray(listing.projects) ? listing.projects : [];
      this.projectsError = listing ? "" : (lastError.get("list_projects") || "");
      this.draw();
    }
    return this;
  }

  close() {
    if (!this.node) return;
    this.node.remove();
    this.node = null;
    if (this.restoreFocus && this.restoreFocus.focus && this.restoreFocus.isConnected) this.restoreFocus.focus();
  }

  /** Everything the palette can offer for the current query, grouped and ranked. */
  items() {
    const query = this.input.value.trim();
    const shell = this.shell;
    const out = [];
    const add = (group, item) => {
      const s = Math.max(score(item.label, query), item.also ? score(item.also, query) * 0.5 : 0);
      if (s > 0) out.push({ ...item, group, score: s });
    };

    shell.screens().forEach((screen, index) => add("Workspaces", {
      label: screen.label, hint: screen.key ? `${screen.key}` : "", control: `palette:open:${screen.id}`,
      also: screen.id, run: () => shell.open(screen.id),
    }));

    if (this.projects) {
      for (const p of this.projects.slice(0, 60)) {
        add("Projects", {
          label: String(p.name || p.id), hint: p.id === (shell.project && shell.project.id) ? "open now" : "open",
          also: p.root_dir || "", control: `palette:project:${p.id}`,
          run: () => shell.openProject(p.id, p.name),
        });
      }
    }

    if (query) {
      if (this.places) {
        for (const place of this.places) {
          out.push({
            group: "Places", label: place.name, hint: [place.kind, place.source].filter(Boolean).join(" · "),
            control: "palette:place", score: 50, run: () => shell.flyTo(place),
          });
        }
      }
      out.push({
        group: "Places", label: this.searching ? `Searching places for “${query}”…` : `Search places for “${query}”`,
        hint: "Enter", control: "palette:search-places", score: 1, keep: true,
        run: () => this.searchPlaces(query),
      });
    }

    for (const key of actionKeys()) {
      add("Actions", { label: key, hint: describe(key), prose: true, also: describe(key), action: key,
        run: () => shell.runAction(key) });
    }

    for (const s of shell.shortcutList()) {
      if (!s.name) continue;
      add("Shortcuts", { label: s.what, hint: s.keys, control: `palette:shortcut:${s.name}`,
        run: () => shell.shortcut(s.name) });
    }

    // Grouped in GROUPS order; best match first inside a group.
    return GROUPS.flatMap((g) => out.filter((i) => i.group === g).sort((a, b) => b.score - a.score));
  }

  draw() {
    if (!this.node) return;
    const items = this.items();
    this.visible = items;
    this.active = Math.max(0, Math.min(this.active, items.length - 1));
    this.list.replaceChildren();
    if (!items.length) {
      this.list.appendChild(el("div", { class: "odk-palette-empty", text: "Nothing matches." }));
    }
    let group = "";
    items.forEach((item, index) => {
      if (item.group !== group) {
        group = item.group;
        this.list.appendChild(el("div", { class: "odk-palette-group odk-label", role: "presentation", text: group }));
      }
      const attrs = {
        type: "button", class: "odk-palette-row", role: "option", id: `odk-palette-opt-${index}`,
        "aria-selected": index === this.active ? "true" : "false", tabindex: "-1",
      };
      if (item.action) attrs["data-action"] = item.action;
      else attrs["data-control"] = item.control;
      const row = el("button", attrs, [
        el("span", { class: "odk-palette-label", text: item.label }),
        item.hint ? el("span", { class: `odk-palette-hint${item.prose ? " odk-palette-hint--prose" : ""}`, text: item.hint }) : null,
      ]);
      row.addEventListener("mousemove", () => { if (this.active !== index) { this.active = index; this.mark(); } });
      row.addEventListener("click", () => this.choose(index));
      this.list.appendChild(row);
    });
    this.mark();
    const count = items.filter((i) => i.group === "Actions").length;
    const parts = [`${count} action${count === 1 ? "" : "s"}`];
    if (this.projectsError) parts.push(`projects: ${this.projectsError}`);
    if (this.placeNote) parts.push(this.placeNote);
    this.status.textContent = parts.join(" · ");
  }

  mark() {
    if (!this.list) return;
    this.list.querySelectorAll(".odk-palette-row").forEach((row, index) => {
      const on = index === this.active;
      row.setAttribute("aria-selected", on ? "true" : "false");
      if (on) {
        this.input.setAttribute("aria-activedescendant", row.id);
        row.scrollIntoView({ block: "nearest" });
      }
    });
  }

  onKey(event) {
    const n = (this.visible || []).length;
    if (event.key === "ArrowDown") { this.active = Math.min(this.active + 1, n - 1); this.mark(); event.preventDefault(); }
    else if (event.key === "ArrowUp") { this.active = Math.max(this.active - 1, 0); this.mark(); event.preventDefault(); }
    else if (event.key === "PageDown") { this.active = Math.min(this.active + 8, n - 1); this.mark(); event.preventDefault(); }
    else if (event.key === "PageUp") { this.active = Math.max(this.active - 8, 0); this.mark(); event.preventDefault(); }
    else if (event.key === "Enter") { event.preventDefault(); this.choose(this.active); }
    else if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); this.close(); }
    else if (event.key === "Tab") { event.preventDefault(); }
  }

  choose(index) {
    const item = (this.visible || [])[index];
    if (!item) return;
    if (item.keep) { item.run(); return; }
    this.close();
    item.run();
  }

  /** Ask search_places, explicitly. Results become the Places group. */
  async searchPlaces(query) {
    if (this.searching) return;
    if (!connected()) { this.placeNote = "places: the application is not connected"; this.draw(); return; }
    this.searching = true;
    this.draw();
    const result = await tryCall("search_places", query);
    this.searching = false;
    if (!this.node) return;
    if (!result) {
      this.places = [];
      this.placeNote = `places: ${lastError.get("search_places") || "the search failed"}`;
    } else {
      this.places = result.results || [];
      this.placeNote = result.error || result.provider_note || "";
      if (result.provider && result.provider !== "offline") this.shell.noteNetworkUse(`place search (${result.provider})`);
      if (!this.places.length) this.placeNote = `No places match “${query}”. ${this.placeNote}`.trim();
    }
    this.draw();
    const first = this.visible.findIndex((i) => i.group === "Places" && !i.keep);
    if (first >= 0) { this.active = first; this.mark(); }
  }
}
