/* Field Fly (SysFieldFly): the Field Daylight renderer for the Fly workspace. It reads
   the same ../fly/model.js as the desk renderer and draws the aircraft with the same
   FlyMap, so both modes show the same Api answers.

   Map on the left with the mission card; on the right a status strip, four big tiles
   (altitude, speed, battery, link), progress, and the screen's one yellow action, which
   follows the flight: Connect, Upload, Start, Pause or Resume. Return home sits beside
   it in red, and Abort is held, as on the desk. Every touch target is at least 64 px. */

export { CALLS } from "../fly/model.js";

import { el, simulatedTag, dot } from "../../components.js";
import { live } from "../../../workspace/live.js";
import {
  DRIVERS, FAST_CALLS, SLOW_CALLS, FAST_MS, SLOW_MS, read, derive, messageFeed,
  fixed, distance, duration, clock, isNum,
} from "../fly/model.js";
import { FlyMap, HOLD_TO_ABORT_MS, glyph } from "../fly/desk.js";
import { ensureFlyStyles } from "../fly/styles.js";

export const SCREEN = "fly";
export const MODE = "field";

const recall = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const remember = (key, value) => { try { localStorage.setItem(key, value); } catch { /* private window */ } };

export async function render(regions, context) {
  ensureFlyStyles();
  const view = new FlyField(regions, context);
  view.mount();
  await view.refresh(true);
  view.start();
  return view;
}

class FlyField {
  constructor(regions, context) {
    this.regions = regions;
    this.context = context;
    this.shell = context.shell;
    this.model = derive({});
    this.track = [];
    this.messages = [];
    this.driver = recall("odk.fly.driver") || "mock";
    this.feed = messageFeed((all) => { this.messages = all; });
  }

  run(key, payload) {
    return this.shell.runAction(key, payload).then(async (r) => { await this.refresh(true); return r; });
  }

  mount() {
    const { left, main } = this.regions;
    this.status = el("div", { class: "flyf-status", role: "status" });
    this.mapHost = el("div", { class: "fly-map odk-map" });
    this.card = el("div", { class: "flyf-map-card" });
    this.banner = el("div", { class: "fly-banner", role: "status", hidden: "" });
    main.replaceChildren(el("div", { class: "fly-stage" }, [
      this.status,
      el("div", { class: "fly-map-wrap" }, [this.mapHost, this.card, this.banner]),
    ]));
    this.body = el("div", { class: "flyf-body" });
    this.actions = el("div", { class: "flyf-actions" });
    // The checklist's own answer, re-read with the screen: what still stands between
    // the aircraft and a start.
    this.checks = live({
      calls: ["run_preflight"],
      every: SLOW_MS,
      empty: "The checklist could not be read.",
      render: ([answer]) => this.checkList(answer),
    });
    this.root = el("div", { class: "flyf-panel" }, [this.body, this.actions]);
    left.replaceChildren(this.root);
    this.flyMap = new FlyMap(this.mapHost, this.shell);
    this.flyMap.mount();
    this.onRefresh = () => this.refresh(true);
    window.addEventListener("odk:refresh", this.onRefresh);
  }

  start() {
    const tick = async () => {
      if (!this.root.isConnected) {
        window.removeEventListener("odk:refresh", this.onRefresh);
        this.flyMap.destroy();
        return;
      }
      await this.refresh(Date.now() - (this.slowAt || 0) >= SLOW_MS);
      this.timer = setTimeout(tick, FAST_MS);
    };
    this.timer = setTimeout(tick, FAST_MS);
  }

  async refresh(slow) {
    if (this.reading) return;
    this.reading = true;
    try {
      const results = await read(slow ? [...FAST_CALLS, ...SLOW_CALLS] : FAST_CALLS);
      if (slow) { this.slowResults = results; this.slowAt = Date.now(); }
      this.model = derive({ ...(this.slowResults || {}), ...results });
      await this.feed();
      const t = this.model.telemetry;
      if (t && t.connected && isNum(t.latitude) && (t.latitude || t.longitude)) this.track.push([t.longitude, t.latitude]);
      this.draw(slow);
    } finally {
      this.reading = false;
    }
  }

  draw(slow) {
    this.drawStatus();
    this.drawCard();
    this.drawBanner();
    this.flyMap.update(this.model, this.track);
    const c = this.model.commands;
    const phase = !this.model.link || !this.model.link.connected ? "connect"
      : !c.onVehicle && !c.flying ? "upload" : !c.flying ? "start" : "flying";
    if (phase !== this.phase || phase !== "connect" || slow) {
      this.phase = phase;
      this.drawBody(phase);
      this.drawActions(phase);
    }
    const planned = this.model.planned;
    this.context.setSubtitle(planned ? [planned.template, this.model.link && this.model.link.driver].filter(Boolean).join(" · ") : null);
  }

  /* -------------------------------------------------------------- status strip */

  drawStatus() {
    const m = this.model;
    const t = m.telemetry;
    const state = m.linkState;
    const words = { ok: "LINK UP", degraded: "LINK WEAK", lost: "LINK LOST", none: "NO LINK" };
    const parts = [
      el("span", { class: `flyf-link is-${state}` }, [glyph("follow", 24), el("span", { text: words[state] })]),
    ];
    const sim = simulatedTag(t || m.link);
    if (sim) parts.push(el("span", {}, [sim, el("span", { text: "telemetry" })]));
    if (t && isNum(t.gps_fix)) parts.push(el("span", {}, [el("b", { text: "GPS" }), el("span", { text: `${t.satellites ?? "?"} sats` })]));
    parts.push(el("span", {}, [el("b", { text: "Mode" }), el("span", { text: (t && t.flight_mode) || "not reported" })]));
    if (state === "lost") parts.push(el("span", { text: `${clock(t.telemetry_age_s)} since heard` }));
    this.status.replaceChildren(...parts);
  }

  drawCard() {
    const p = this.model.planned;
    const prog = this.model.progress;
    if (!p) { this.card.replaceChildren(el("span", { class: "flyf-label", text: "Mission" }), el("strong", { text: "No mission planned" })); return; }
    const facts = [p.waypoints != null ? `${p.waypoints} wp` : null, distance(p.path_distance_m),
      isNum(p.altitude_m) ? `${fixed(p.altitude_m, 0)} m` : null].filter(Boolean).join(" · ");
    const rows = [el("span", { class: "flyf-label", text: "Mission" }), el("strong", { text: p.template || "mission" }), el("span", { text: facts })];
    if (prog && this.model.mission && this.model.mission.on_vehicle) {
      rows.push(el("div", { class: "flyf-key" }, [
        el("span", {}, [el("i", { class: "k-flown" }), "Flown"]),
        el("span", {}, [el("i", { class: "k-left" }), "Remaining"]),
      ]));
    } else {
      rows.push(el("span", { text: this.model.connected ? "Not on the aircraft yet" : "No vehicle connected" }));
    }
    this.card.replaceChildren(...rows);
  }

  drawBanner() {
    const m = this.model;
    const lost = m.linkState === "lost";
    const latest = [...this.messages].reverse().find((x) => x.severity <= 2 && x.source === "vehicle");
    const recent = latest && Date.now() - new Date(latest.time_utc).getTime() < SLOW_MS * 12;
    if (!lost && !recent) { this.banner.hidden = true; return; }
    this.banner.hidden = false;
    this.banner.className = "fly-banner is-danger";
    this.banner.replaceChildren(dot("danger"), el("div", { class: "fly-banner-body" }, [
      el("span", { class: "fly-banner-title", text: lost ? `Link lost · ${clock(m.telemetry.telemetry_age_s)}` : latest.text }),
      el("span", { class: "fly-banner-detail" }, [el("span", { text: lost ? "Last known values" : latest.level }), simulatedTag(m.telemetry || m.link, { small: true })].filter(Boolean)),
    ]));
  }

  /* ---------------------------------------------------------------- side panel */

  drawBody(phase) {
    if (phase === "connect") { this.body.replaceChildren(this.connectForm(), this.checks); return; }
    if (phase !== "flying") {
      this.body.replaceChildren(this.tiles(), this.checks);
      return;
    }
    this.body.replaceChildren(this.tiles());
  }

  connectForm() {
    const seg = el("div", { class: "fly-seg", role: "radiogroup", "aria-label": "Driver" }, DRIVERS.map((d) => {
      const b = el("button", { type: "button", class: "fly-seg-btn", role: "radio", "data-control": `driver:${d.id}`,
        "aria-checked": d.id === this.driver ? "true" : "false", text: d.label, title: d.hint });
      b.addEventListener("click", () => { this.driver = d.id; remember("odk.fly.driver", d.id); this.drawBody("connect"); });
      return b;
    }));
    const fallback = DRIVERS.find((d) => d.id === this.driver) || DRIVERS[2];
    const uri = recall(`odk.fly.uri.${fallback.id}`) || fallback.uri;
    return el("div", { class: "flyf-connect" }, [
      el("span", { class: "flyf-tile-head", text: "Connect" }), seg,
      el("span", { class: "flyf-sub", text: `${fallback.hint} · ${uri}` }),
    ]);
  }

  tiles() {
    const m = this.model;
    const t = m.telemetry || {};
    const tile = (label, value, unit, sub, record = t) => el("div", { class: "flyf-tile" }, [
      el("div", { class: "flyf-tile-head" }, [el("span", { text: label }), simulatedTag(record, { small: true })].filter(Boolean)),
      value === null ? el("span", { class: "flyf-value is-missing", text: "not reported" })
        : el("span", { class: "flyf-value" }, [value, unit ? el("small", { text: unit }) : null].filter(Boolean)),
      sub ? el("span", { class: "flyf-sub", text: sub }) : null,
    ].filter(Boolean));
    const p = m.planned;
    const prog = m.progress;
    const tiles = [
      tile("Altitude", fixed(t.altitude_rel_m, 0), "m", p && isNum(p.altitude_m) ? `Plan target ${fixed(p.altitude_m, 0)} m` : "relative to home"),
      tile("Speed", fixed(t.speed_mps), "m/s", p && isNum(p.speed_m_s) ? `Plan cruise ${fixed(p.speed_m_s)} m/s` : null),
      tile("Battery", isNum(t.battery_pct) ? fixed(t.battery_pct, 0) : null, "%", isNum(t.battery_v) ? `${fixed(t.battery_v, 1)} V` : null),
      tile("Link", isNum(t.link_quality_pct) ? fixed(t.link_quality_pct, 0) : null, "%",
        isNum(t.telemetry_age_s) ? `heard ${fixed(t.telemetry_age_s)} s ago` : null),
    ];
    const wide = el("div", { class: "flyf-tile flyf-tile--wide" }, [
      el("div", { class: "flyf-tile-head" }, [el("span", { text: "Progress" }), simulatedTag(t, { small: true })].filter(Boolean)),
    ]);
    if (prog && m.mission && m.mission.on_vehicle) {
      const speed = isNum(t.speed_mps) ? t.speed_mps : null;
      const eta = speed && speed > 0.5 ? duration(prog.leftMetres / speed) : null;
      wide.append(
        el("span", { class: "flyf-value", text: prog.current !== null ? `Item ${prog.current}` : "not reported" }),
        el("span", { class: "flyf-sub", text: `${distance(prog.flownMetres)} of ${distance(prog.routeLength)}${eta ? ` · ~${eta} left` : ""}` }),
        el("div", { class: "flyf-bar" }, [el("span", { style: `width:${((prog.fraction || 0) * 100).toFixed(1)}%` })]),
      );
    } else {
      wide.append(el("span", { class: "flyf-sub", text: m.connected ? "No mission on the aircraft yet." : "No vehicle connected." }));
    }
    return el("div", { class: "flyf-tiles" }, [...tiles, wide]);
  }

  checkList(answer) {
    const rows = (answer.checks || []).filter((c) => (c.severity === "blocking" && !c.confirmed) || (c.requires_manual && !c.confirmed));
    if (!rows.length) return el("ul", { class: "flyf-checks" }, [el("li", {}, [dot("ok"), el("span", { text: "Preflight ready" })])]);
    return el("ul", { class: "flyf-checks", "aria-label": "Preflight" }, rows.slice(0, 6).map((c) => {
      const li = el("li", { class: c.severity === "blocking" && !c.confirmed ? "is-danger" : "" }, [
        dot(c.severity === "blocking" && !c.confirmed ? "danger" : "warn"),
        el("span", { class: "flyf-check-name", text: c.name || c.id, title: c.message }),
      ]);
      if (c.requires_manual && !c.confirmed) {
        const b = el("button", { type: "button", class: "odk-btn odk-btn--secondary", "data-action": "Confirm Check", text: "Confirm" });
        b.addEventListener("click", () => this.run("Confirm Check", { id: c.id, name: c.name }));
        li.appendChild(b);
      }
      return li;
    }));
  }

  big(label, sub, action, cls, onClick, disabled = false) {
    const b = el("button", { type: "button", class: `flyf-big ${cls}`.trim(), "data-action": action }, [
      el("strong", { text: label }), sub ? el("span", { text: sub }) : null,
    ].filter(Boolean));
    b.disabled = disabled;
    b.addEventListener("click", onClick || (() => this.run(action)));
    return b;
  }

  drawActions(phase) {
    const c = this.model.commands;
    const rth = this.big("RTH", "Return home", "RTL", "flyf-big--danger", null, !c.canReturn);
    if (phase === "connect") {
      const d = DRIVERS.find((x) => x.id === this.driver) || DRIVERS[2];
      const uri = recall(`odk.fly.uri.${d.id}`) || d.uri;
      this.actions.replaceChildren(this.big("Connect", d.label, "Connect Vehicle", "", () => this.run("Connect Vehicle", { driver: d.id, uri })),
        this.big("Disconnect", "", "Disconnect", "flyf-big--commit", null, true));
      return;
    }
    if (phase === "upload") {
      this.actions.replaceChildren(this.big("Upload", "Send the mission", "Upload", "", null, !c.canUpload),
        this.big("Disconnect", "", "Disconnect", "flyf-big--commit"));
      return;
    }
    if (phase === "start") {
      this.actions.replaceChildren(this.big("Start", "Arm and fly", "Start Mission", "", null, !c.canStart), rth);
      return;
    }
    const pause = c.canResume ? this.big("Resume", "Continue the route", "Resume", "") : this.big("Pause", "Hold position", "Pause", "", null, !c.canPause);
    const abort = this.holdToAbort();
    this.actions.replaceChildren(pause, rth, abort);
  }

  holdToAbort() {
    const fill = el("span", { class: "fly-hold-fill", "aria-hidden": "true" });
    const b = el("button", { type: "button", class: "fly-hold flyf-hold", "data-action": "Abort", style: "grid-column: 1 / -1; height: 64px; font-size: 18px;" },
      [fill, el("span", { class: "fly-hold-label", text: "Hold to abort and land" })]);
    let started = 0;
    let frame = 0;
    const reset = () => { cancelAnimationFrame(frame); started = 0; fill.style.width = "0%"; };
    const step = async () => {
      const p = Math.min(1, (performance.now() - started) / HOLD_TO_ABORT_MS);
      fill.style.width = `${(p * 100).toFixed(1)}%`;
      if (p < 1) { frame = requestAnimationFrame(step); return; }
      reset();
      const before = this.shell.confirmAll;
      this.shell.confirmAll = true;
      try { await this.run("Abort"); } finally { this.shell.confirmAll = before; }
    };
    b.addEventListener("pointerdown", () => { if (!started) { started = performance.now(); frame = requestAnimationFrame(step); } });
    ["pointerup", "pointerleave", "pointercancel"].forEach((n) => b.addEventListener(n, reset));
    b.addEventListener("keydown", (e) => { if ((e.key === " " || e.key === "Enter") && !started) { e.preventDefault(); started = performance.now(); frame = requestAnimationFrame(step); } });
    b.addEventListener("keyup", reset);
    return b;
  }
}
