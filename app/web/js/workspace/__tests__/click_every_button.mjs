/*
 * Click every button in every workspace, and record what each one actually did.
 *
 * The previous tests read actions.js as text and checked that a method name appeared in
 * it. That would have passed while the shell threw on the first click, and it did pass
 * while a syntax error in shell.js left the whole window blank -- text in a file is not
 * evidence that a button works.
 *
 * So this mounts the real Shell against a stub DOM and a FAKE pywebview bridge that
 * records every call, then clicks all seventy-five toolbar actions across all fourteen
 * workspaces and reports, per button:
 *
 *     ok      it called the Api, changed the view, armed a tool or switched workspace
 *     said    it declared itself unavailable and named what is missing
 *     threw   it raised, which is the failure this file exists to catch
 *
 * The bridge answers every method with a plausible success, because the point here is
 * that the wiring reaches Python and survives the round trip -- not that Python is
 * correct, which is what the Python tests are for.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/* ------------------------------------------------------------------ stub DOM */

class ClassList {
  constructor() { this.items = new Set(); }
  add(...names) { names.forEach((n) => this.items.add(n)); }
  remove(...names) { names.forEach((n) => this.items.delete(n)); }
  toggle(name, force) {
    const on = force === undefined ? !this.items.has(name) : Boolean(force);
    if (on) this.items.add(name); else this.items.delete(name);
    return on;
  }
  contains(name) { return this.items.has(name); }
}

class Node {
  constructor(tag = "div") {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.classList = new ClassList();
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this._text = "";
    this.onclick = null;
  }
  get className() { return [...this.classList.items].join(" "); }
  set className(value) { this.classList = new ClassList(); String(value).split(/\s+/).filter(Boolean).forEach((n) => this.classList.add(n)); }
  get textContent() { return this._text; }
  set textContent(value) { this._text = String(value); }
  set innerHTML(value) { this._text = String(value); this.children = []; }
  get innerHTML() { return this._text; }
  appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
  append(...kids) { kids.forEach((k) => k && this.appendChild(k)); }
  prepend(child) { child.parentElement = this; this.children.unshift(child); return child; }
  remove() {
    if (!this.parentElement) return;
    const index = this.parentElement.children.indexOf(this);
    if (index >= 0) this.parentElement.children.splice(index, 1);
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  addEventListener() {}
  removeEventListener() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 1600, height: 900, right: 1600, bottom: 900 }; }
  focus() {}
  descendants() {
    const out = [];
    const walk = (node) => node.children.forEach((child) => { out.push(child); walk(child); });
    walk(this);
    return out;
  }
  querySelectorAll(selector) {
    const wanted = selector.replace(/^\./, "");
    return this.descendants().filter((node) => node.classList.contains(wanted));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

const documentStub = {
  createElement: (tag) => new Node(tag),
  createTextNode: (text) => { const n = new Node("#text"); n.textContent = text; return n; },
  body: new Node("body"),
  documentElement: new Node("html"),
  addEventListener() {},
  querySelectorAll: () => [],
  querySelector: () => null,
};

const storage = new Map();

/* --------------------------------------------------------------- fake bridge */

const calls = [];
const bridge = new Proxy({}, {
  get(_target, name) {
    if (typeof name !== "string") return undefined;
    return (...args) => {
      calls.push({ method: name, args });
      // Shapes the shell reads. Everything else gets a bare success.
      if (name === "list_projects") return Promise.resolve({ ok: true, projects: [{ id: 1, name: "Demo", root_dir: "/tmp/demo" }] });
      if (name === "get_state") return Promise.resolve({ ok: true, project: { id: 1, name: "Demo" }, dataset: "/tmp/images" });
      if (name === "list_jobs") return Promise.resolve({ ok: true, jobs: [] });
      if (name === "pick_folder") return Promise.resolve({ ok: true, path: "/tmp/images" });
      if (name === "pick_file") return Promise.resolve({ ok: true, path: "/tmp/points.csv" });
      if (name === "create_project") return Promise.resolve({ ok: true, project_id: 2 });
      if (name === "run_reconstruction" || name === "run_pipeline") return Promise.resolve({ ok: true, job_id: "job-1" });
      if (name === "job_status") return Promise.resolve({ ok: true, state: "done" });
      if (name === "capabilities") return Promise.resolve({ ok: true, capabilities: {} });
      if (name === "mission_geojson") return Promise.resolve({ ok: true, geojson: { features: [] } });
      return Promise.resolve({ ok: true });
    };
  },
});

globalThis.window = {
  pywebview: { api: bridge },
  location: { search: "" },
  localStorage: { getItem: (k) => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, v) },
  addEventListener() {},
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  // Answer every prompt and confirmation, so a button that asks is still exercised.
  prompt: () => "Demo",
  confirm: () => true,
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  setTimeout: (fn) => { void fn; return 0; },   // no timers: the run must be deterministic
  requestAnimationFrame: () => 0,
};
globalThis.document = documentStub;
// primitives.js tests `value instanceof Node` to decide whether a cell holds an
// element or text, so the stub has to be the global Node for that branch to work.
globalThis.Node = Node;
globalThis.localStorage = window.localStorage;
globalThis.getComputedStyle = window.getComputedStyle;
globalThis.setTimeout = (fn, ms) => (ms === undefined || ms === 0 ? fn() : 0);
globalThis.confirm = window.confirm;
globalThis.prompt = window.prompt;
globalThis.alert = () => {};

/* -------------------------------------------------------------------- the run */

// pathToFileURL because a Windows absolute path is not a valid ESM specifier.
const { Shell } = await import(pathToFileURL(resolve(HERE, "../shell.js")).href);
const { WORKSPACES } = await import(pathToFileURL(resolve(HERE, "../workspaces.js")).href);

const root = new Node("div");
const shell = new Shell(root);

const results = [];
for (const workspace of WORKSPACES) {
  shell.open(workspace.id);
  const buttons = (workspace.toolbar || []).filter((a) => a !== "|");
  for (const action of buttons) {
    const before = calls.length;
    const viewBefore = shell.canvasView;
    const toolBefore = shell.activeTool;
    let outcome = "silent";
    let detail = "";
    try {
      await shell.runAction(action);
      const made = calls.slice(before).map((c) => c.method);
      if (made.length) { outcome = "ok"; detail = made.join(","); }
      else if (shell.canvasView !== viewBefore) { outcome = "ok"; detail = `view=${shell.canvasView}`; }
      else if (shell.activeTool !== toolBefore) { outcome = "ok"; detail = `tool=${shell.activeTool}`; }
      else {
        // Everything else must at least have told the user something.
        const toast = root.querySelectorAll("toast").pop();
        const text = toast ? toast.textContent : "";
        if (/not available|no handler|Open a project|Select a dataset|Not connected/.test(text)) {
          outcome = "said";
          detail = text.slice(0, 60);
        } else if (text) {
          outcome = "ok";
          detail = text.slice(0, 60);
        }
      }
    } catch (error) {
      outcome = "threw";
      detail = String(error && error.message ? error.message : error);
    }
    results.push({ workspace: workspace.id, action, outcome, detail });
  }
}

const threw = results.filter((r) => r.outcome === "threw");
const silent = results.filter((r) => r.outcome === "silent");
const ok = results.filter((r) => r.outcome === "ok");
const said = results.filter((r) => r.outcome === "said");

for (const row of threw) console.log(`THREW  ${row.workspace}/${row.action}: ${row.detail}`);
for (const row of silent) console.log(`SILENT ${row.workspace}/${row.action}`);

console.log(`buttons clicked: ${results.length}`);
console.log(`ok: ${ok.length}`);
console.log(`declared unavailable: ${said.length}`);
console.log(`threw: ${threw.length}`);
console.log(`silent: ${silent.length}`);
if (!threw.length && !silent.length) console.log("every button responded");
process.exit(threw.length || silent.length ? 1 : 0);
