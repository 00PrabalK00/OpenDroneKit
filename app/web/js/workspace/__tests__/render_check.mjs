/* Executes every workspace under a stub DOM.
 *
 * A UI that only "looks right" in a screenshot is not verified. This mounts all
 * fourteen workspaces through the real dock, renders every panel, and asserts the
 * selection bus delivers -- including that one throwing subscriber cannot stop the
 * others, which is the failure that would leave half the panels stale during a
 * flight while nothing looked broken.
 */
// Execute the workspace modules under a minimal DOM so real errors surface.
class El {
  constructor(tag){ this.tagName=tag.toUpperCase(); this.children=[]; this.style={}; this.dataset={};
    this.classList={ _s:new Set(), add:(...c)=>c.forEach(x=>this.classList._s.add(x)),
      remove:(...c)=>c.forEach(x=>this.classList._s.delete(x)),
      toggle:(c,f)=>{ f===undefined? (this.classList._s.has(c)?this.classList._s.delete(c):this.classList._s.add(c)) : (f?this.classList._s.add(c):this.classList._s.delete(c)); return f; },
      contains:(c)=>this.classList._s.has(c) };
  }
  set className(v){ this._cn=v; String(v).split(/\s+/).filter(Boolean).forEach(c=>this.classList._s.add(c)); }
  get className(){ return this._cn||""; }
  appendChild(c){ this.children.push(c); return c; }
  append(...cs){ cs.forEach(c=>this.children.push(c)); }
  addEventListener(){} removeEventListener(){}
  setAttribute(k,v){ this[k]=v; } getAttribute(k){ return this[k]; }
  querySelectorAll(){ return { forEach(){}, length:0 }; }
  querySelector(){ return null; }
  remove(){}
  focus(){} select(){}
  set innerHTML(v){ this._html=v; this.children=[]; } get innerHTML(){ return this._html||""; }
  set textContent(v){ this._text=v; } get textContent(){ return this._text||""; }
}
global.document = {
  createElement:(t)=>new El(t),
  createTextNode:(t)=>({ nodeType:3, text:t }),
  addEventListener(){}, body:new El("body"),
  querySelector(){ return null; }, querySelectorAll(){ return { forEach(){} }; },
  head:new El("head"), activeElement:null,
};
global.Node = El;
global.window = { open:()=>null, addEventListener(){} };
global.localStorage = { _d:{}, getItem(k){ return this._d[k]??null; }, setItem(k,v){ this._d[k]=v; }, removeItem(k){ delete this._d[k]; } };
global.setInterval = () => 0;
global.console = console;

const { WORKSPACES, WORKSPACE_BY_ID } = await import("../workspaces.js");
const { Dock } = await import("../dock.js");
const { selection } = await import("../primitives.js");

let panels = 0, failures = [];
for (const w of WORKSPACES) {
  if (!w.id || !w.title) failures.push(`workspace missing id/title: ${JSON.stringify(w).slice(0,60)}`);
  for (const region of ["left","right","bottom"]) {
    for (const def of (w[region] || [])) {
      panels++;
      if (!def.id) failures.push(`${w.id}/${region}: panel without id`);
      if (!def.title) failures.push(`${w.id}/${def.id}: panel without title`);
      try {
        if (def.tabs) def.tabs.forEach(t => t.render());
        else if (def.render) def.render();
        else failures.push(`${w.id}/${def.id}: renders nothing`);
      } catch (e) { failures.push(`${w.id}/${def.id} render threw: ${e.message}`); }
    }
  }
  try { if (w.canvas) w.canvas(); } catch (e) { failures.push(`${w.id} canvas threw: ${e.message}`); }
}

// Every workspace must actually mount through the dock.
const root = new El("div");
const dock = new Dock(root);
for (const w of WORKSPACES) {
  try { dock.render(w); } catch (e) { failures.push(`dock.render(${w.id}) threw: ${e.message}`); }
}

// Selection bus must fan out.
let got = null;
selection.on("finding", v => { got = v; });
selection.select("finding", { id: "F-118" });
if (!got || got.id !== "F-118") failures.push("selection bus did not deliver");

// A throwing subscriber must not stop the others.
selection.on("asset", () => { throw new Error("boom"); });
let second = false;
selection.on("asset", () => { second = true; });
selection.select("asset", { id: "a" });
if (!second) failures.push("a throwing subscriber blocked the next one");

console.log(`workspaces: ${WORKSPACES.length}`);
console.log(`panels rendered: ${panels}`);
console.log(failures.length ? "FAILURES:\n" + failures.join("\n") : "all workspaces render");
process.exit(failures.length ? 1 : 0);
