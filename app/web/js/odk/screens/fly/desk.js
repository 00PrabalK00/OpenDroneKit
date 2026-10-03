/* Fly: desk renderer (SysFly artboard). Left: CONNECTION, PREFLIGHT, MISSION, SESSION.
   Centre: the project map with the aircraft, the flown and remaining route, the link
   banner, MISSION PROGRESS and the command bar. Inspector: TELEMETRY / MESSAGES / EVENTS
   and the plan the aircraft is flying.

   Every value is read from the Api through ./model.js. The screen polls the aircraft
   once a second and the rest every five seconds, only while it is on screen. Commands
   go through the shell's runAction, so the palette, the native menu and these buttons
   run the same verbs with the same confirmations; Abort is confirmed by holding. */

import {
  el, button, iconButton, section, simulatedTag, tag, dot, panelTabs, input, select,
} from "../../components.js";
import { live } from "../../../workspace/live.js";
import { mountProjectMap } from "../../map/furniture.js";
import {
  DRIVERS, FAST_CALLS, SLOW_CALLS, FAST_MS, SLOW_MS, read, derive, eventsPanel, sessionPanel,
  messageFeed, fixed, distance, duration, clock, latitude, longitude, heading, GPS_FIX, timeOfDay,
  severityTone, isNum,
} from "./model.js";
import { ensureFlyStyles } from "./styles.js";

export const SCREEN = "fly";

/** Hold Abort this long to send it: the confirmation is the hold. */
export const HOLD_TO_ABORT_MS = 1500;
/** How many link samples (one per poll) the LINK strip shows. */
const LINK_SAMPLES = 30;

const STORE = { driver: "odk.fly.driver", uri: "odk.fly.uri", tab: "odk.fly.tab", follow: "odk.fly.follow" };
const remember = (key, value) => { try { localStorage.setItem(key, value); } catch { /* private window */ } };
const recall = (key) => { try { return localStorage.getItem(key); } catch { return null; } };

/* Stroke icons for the command bar (24 px grid, drawn here; nothing fetched). */
const PATHS = {
  lock: '<rect x="5" y="11" width="14" height="9"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path>',
  unlock: '<rect x="5" y="11" width="14" height="9"></rect><path d="M8 11V8a4 4 0 0 1 7.5-2"></path>',
  play: '<path d="M7 5l12 7-12 7z"></path>',
  pause: '<path d="M8 5v14M16 5v14"></path>',
  home: '<path d="M4 11l8-7 8 7"></path><path d="M6 10v10h12V10"></path>',
  stop: '<circle cx="12" cy="12" r="9"></circle><path d="M9 9l6 6M15 9l-6 6"></path>',
  more: '<circle cx="5" cy="12" r="1.5"></circle><circle cx="12" cy="12" r="1.5"></circle><circle cx="19" cy="12" r="1.5"></circle>',
  follow: '<circle cx="12" cy="12" r="3"></circle><path d="M12 2v4M12 18v4M2 12h4M18 12h4"></path>',
  check: '<path d="M5 12l5 5 9-10"></path>',
  warn: '<path d="M12 3l10 18H2z"></path><path d="M12 10v5M12 18v.5"></path>',
  block: '<circle cx="12" cy="12" r="9"></circle><path d="M5.6 5.6l12.8 12.8"></path>',
  pending: '<circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 2"></path>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"></path><path d="M20 4v7h-7"></path>',
};
export function glyph(name, size = 16) {
  const span = document.createElement("span");
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" `
    + `stroke-width="2" stroke-linecap="square" aria-hidden="true">${PATHS[name]}</svg>`;
  return span.firstChild;
}

const NOT_REPORTED = "not reported";
const text = (value, unit = "") => (value === null || value === undefined || value === "" ? NOT_REPORTED : `${value}${unit}`);
const kv = (label, value, { tone, title, node } = {}) => [
  el("dt", { text: label }),
  el("dd", {
    class: [value === null || value === undefined ? "is-missing" : "", tone ? `is-${tone}` : ""].filter(Boolean).join(" "),
    title: title || (value === null || value === undefined ? "" : String(value)),
  }, node ? [node] : [text(value)]),
];

/* ===================================================================== render */

export async function render(regions, context) {
  ensureFlyStyles();
  const view = new FlyDesk(regions, context);
  view.mount();
  await view.refresh({ slow: true });
  view.start();
  return view;
}

class FlyDesk {
  constructor(regions, context) {
    this.regions = regions;
    this.context = context;
    this.shell = context.shell;
    this.model = derive({});
    this.messages = [];
    this.samples = [];
    this.track = [];
    this.driver = recall(STORE.driver) || "mock";
    this.uris = {};
    this.tab = recall(STORE.tab) || "telemetry";
    this.slowAt = 0;
    this.feed = messageFeed((all) => { this.messages = all; this.renderMessages(); this.renderBanner(); this.renderPreArm(); });
  }

  run(key, payload) {
    return this.shell.runAction(key, payload).then(async (result) => {
      await this.refresh({ slow: true });
      return result;
    });
  }

  /* ---------------------------------------------------------------- frame */

  mount() {
    const { left, main, inspector } = this.regions;
    left.replaceChildren(this.buildLeft());
    main.replaceChildren(this.buildMain());
    inspector.replaceChildren(this.buildInspector());
    this.flyMap = new FlyMap(this.mapHost, this.shell);
    this.flyMap.mount();
    this.onRefresh = () => this.refresh({ slow: true });
    window.addEventListener("odk:refresh", this.onRefresh);
  }

  start() {
    const tick = async () => {
      if (!this.root.isConnected) { this.stop(); return; }
      const slow = Date.now() - this.slowAt >= SLOW_MS;
      await this.refresh({ slow });
      this.timer = setTimeout(tick, FAST_MS);
    };
    this.timer = setTimeout(tick, FAST_MS);
  }

  stop() {
    clearTimeout(this.timer);
    window.removeEventListener("odk:refresh", this.onRefresh);
    if (this.flyMap) this.flyMap.destroy();
    if (this.more) { this.more.close(); this.more.node.remove(); }
    if (this.alerts) { this.alerts.close(); this.alerts.node.remove(); }
  }

  async refresh({ slow = false } = {}) {
    if (this.reading) return;
    this.reading = true;
    try {
      const calls = slow ? [...FAST_CALLS, ...SLOW_CALLS] : FAST_CALLS;
      const results = await read(calls);
      if (slow) { this.slowResults = results; this.slowAt = Date.now(); }
      this.model = derive({ ...(this.slowResults || {}), ...results });
      await this.feed();
      this.sample();
      this.update(slow);
    } finally {
      this.reading = false;
    }
  }

  /** One link sample per poll: what this station heard, for the LINK strip. */
  sample() {
    const t = this.model.telemetry;
    const state = t ? this.model.linkState : "none";
    this.samples.push(state);
    if (this.samples.length > LINK_SAMPLES) this.samples.splice(0, this.samples.length - LINK_SAMPLES);
    if (t && t.connected && isNum(t.latitude) && (t.latitude || t.longitude)) {
      const last = this.track[this.track.length - 1];
      if (!last || last[0] !== t.longitude || last[1] !== t.latitude) this.track.push([t.longitude, t.latitude]);
      if (this.track.length > 4000) this.track.splice(0, this.track.length - 4000);
    }
  }

  update(slow) {
    this.renderConnection();
    if (slow) { this.renderPreflight(); this.renderMission(); }
    this.renderPreArm();
    this.renderCommands();
    this.renderProgress();
    this.renderBanner();
    this.renderTelemetry();
    this.renderLegend();
    if (this.flyMap) this.flyMap.update(this.model, this.track);
    const s = this.model.session;
    this.context.setFooter(s && s.state ? `session ${s.state.status}` : null);
  }

  /* ================================================================== LEFT */

  buildLeft() {
    this.connSim = el("span", { class: "fly-aside" });
    this.segments = el("div", { class: "fly-seg", role: "radiogroup", "aria-label": "Driver" },
      DRIVERS.map((d) => {
        const b = el("button", {
          type: "button", class: "fly-seg-btn", role: "radio", "data-control": `driver:${d.id}`,
          title: d.hint, text: d.label,
        });
        b.addEventListener("click", () => this.chooseDriver(d.id));
        return b;
      }));
    const uriField = input({ label: "URI", mono: true, name: "fly-uri" });
    this.uriInput = uriField.querySelector("input");
    this.uriInput.setAttribute("spellcheck", "false");
    this.uriInput.addEventListener("input", () => {
      this.uris[this.driver] = this.uriInput.value;
      remember(`${STORE.uri}.${this.driver}`, this.uriInput.value);
    });
    this.connFacts = el("dl", { class: "fly-kv" });
    this.connectBtn = button({
      label: "Connect", action: "Connect Vehicle", variant: "primary",
      onClick: () => this.run("Connect Vehicle", { driver: this.driver, uri: this.uriInput.value.trim() }),
    });
    this.disconnectBtn = button({ label: "Disconnect", action: "Disconnect", onClick: () => this.run("Disconnect") });
    this.chooseDriver(this.driver, { quiet: true });

    const connection = section({
      label: "CONNECTION", aside: this.connSim, className: "fly-connection",
      children: [this.segments, uriField, this.connFacts,
        el("div", { class: "fly-row-btns" }, [this.connectBtn, this.disconnectBtn])],
    });

    this.pfCounts = el("span", { class: "fly-counts" });
    this.pfList = el("ul", { class: "fly-checks", "aria-label": "Preflight checks" });
    this.preArm = el("div", { class: "fly-prearm", hidden: "" });
    const rerun = iconButton({ label: "Re-run preflight", action: "Preflight", onClick: () => this.run("Preflight") });
    rerun.appendChild(glyph("refresh", 14));
    rerun.classList.add("fly-mini");
    const preflight = section({
      label: "PREFLIGHT", aside: el("span", { class: "fly-aside" }, [this.pfCounts, rerun]), className: "fly-preflight",
      children: [this.preArm, this.pfList],
    });

    this.missionRow = el("div", { class: "fly-mission-row" });
    this.versionSelect = el("div", { class: "fly-version" });
    this.uploadBtn = button({ label: "Upload", action: "Upload", variant: "secondary", onClick: () => this.run("Upload") });
    const mission = section({
      label: "MISSION", className: "fly-mission",
      children: [this.missionRow, this.versionSelect, el("div", { class: "fly-row-btns" }, [this.uploadBtn])],
    });

    this.interruptedBox = el("div", { class: "fly-interrupted", hidden: "" });
    const sessionLine = sessionPanel((session, interrupted) => this.sessionSummary(session, interrupted));
    const sessionBtns = el("div", { class: "fly-btn-grid" }, [
      button({ label: "Export log", action: "Export Log", variant: "secondary", onClick: () => this.run("Export Log") }),
      button({ label: "Load flight", action: "Load Flight", variant: "secondary", onClick: () => this.run("Load Flight") }),
      button({ label: "Linked progress", action: "Linked Progress", variant: "secondary", onClick: () => this.run("Linked Progress") }),
      button({ label: "Mark corner", action: "Mark Corner", variant: "secondary", onClick: () => this.run("Mark Corner") }),
      button({ label: "Close boundary", action: "Close Boundary", variant: "ghost", onClick: () => this.run("Close Boundary") }),
      button({ label: "Clear corners", action: "Clear Corners", variant: "ghost", onClick: () => this.run("Clear Corners") }),
    ]);
    const session = section({
      label: "SESSION", className: "fly-session", children: [sessionLine, this.interruptedBox, sessionBtns],
    });

    const note = el("p", { class: "fly-note", text: "Advisories only. The operator decides to fly." });
    this.root = el("div", { class: "fly-left" }, [connection, preflight, mission, session, note]);
    return this.root;
  }

  chooseDriver(id, { quiet = false } = {}) {
    if (!DRIVERS.some((d) => d.id === id)) id = "mock";
    this.driver = id;
    remember(STORE.driver, id);
    this.segments.querySelectorAll(".fly-seg-btn").forEach((b) => {
      b.setAttribute("aria-checked", b.getAttribute("data-control") === `driver:${id}` ? "true" : "false");
    });
    const fallback = DRIVERS.find((d) => d.id === id).uri;
    const stored = this.uris[id] ?? recall(`${STORE.uri}.${id}`);
    this.uriInput.value = stored || fallback;
    if (!quiet) this.uriInput.focus();
  }

  renderConnection() {
    const m = this.model;
    const t = m.telemetry;
    const link = m.link || {};
    const on = Boolean(link.connected);
    this.connSim.replaceChildren(...[simulatedTag(link, { small: true })].filter(Boolean));
    // The form follows the link that is open, so Reconnect reopens exactly it.
    if (on && !this.syncedLink) {
      this.syncedLink = true;
      const known = DRIVERS.find((d) => d.id === link.driver);
      if (known) { this.chooseDriver(known.id, { quiet: true }); this.uriInput.value = link.uri || this.uriInput.value; }
    }
    if (!on) this.syncedLink = false;
    this.connectBtn.lastChild.textContent = on ? "Reconnect" : "Connect";
    this.connectBtn.className = `odk-btn odk-btn--${on ? "secondary" : "primary"}`;
    this.disconnectBtn.disabled = !on;
    let heartbeat = null;
    let tone = "";
    if (m.linkState === "lost") { heartbeat = `lost · ${clock(t.telemetry_age_s)}`; tone = "danger"; }
    else if (on && t && t.connected) {
      heartbeat = `${m.linkState === "degraded" ? "degraded" : "ok"} · ${fixed(t.telemetry_age_s)} s ago`;
      tone = m.linkState === "degraded" ? "warn" : "ok";
    } else if (on) heartbeat = "waiting for the vehicle";
    const rows = [
      ...kv("VEHICLE", on ? `${link.driver} · ${link.uri}` : "not connected", { title: link.uri || "" }),
      ...kv("HEARTBEAT", heartbeat, { tone }),
      ...kv("HARDWARE", on ? (link.is_simulated ? "none: simulator" : "autopilot on link") : null),
    ];
    if (link.last_error) rows.push(...kv("LAST ERROR", link.last_error, { tone: "danger", title: link.last_error }));
    this.connFacts.replaceChildren(...rows);
  }

  renderPreflight() {
    const p = this.model.preflight;
    if (!p) {
      this.pfCounts.replaceChildren(el("span", { class: "is-missing", text: "not read" }));
      this.pfList.replaceChildren(el("li", { class: "fly-empty", text: "The checklist could not be read." }));
      return;
    }
    this.pfCounts.replaceChildren(
      el("span", { class: "is-ok", text: `${p.ok} ok` }),
      el("span", { class: "is-warn", text: `${p.manual} man` }),
      el("span", { class: "is-danger", text: `${p.blocking} blk` }),
    );
    this.pfList.replaceChildren(...p.rows.map((row) => {
      const icon = row.tone === "danger" ? "block" : row.manual && !row.confirmed ? "pending" : row.tone === "warn" ? "warn" : "check";
      const detail = row.confirmed && row.manual ? `confirmed · ${row.note || "by the operator"}` : row.message;
      const li = el("li", { class: `fly-check is-${row.tone}`, "data-check": row.id }, [
        el("span", { class: "fly-check-icon" }, [glyph(icon, 14)]),
        el("div", { class: "fly-check-body" }, [
          el("span", { class: "fly-check-name", text: row.name }),
          el("span", { class: "fly-check-msg", text: detail, title: row.fix || detail }),
        ]),
      ]);
      if (row.manual && !row.confirmed) {
        li.appendChild(button({
          label: "Confirm", action: "Confirm Check", variant: "ghost",
          title: row.fix || `Confirm ${row.name}`,
          onClick: () => this.run("Confirm Check", { id: row.id, name: row.name }),
        }));
      }
      return li;
    }));
  }

  /** Pre-arm and other critical messages from the vehicle, above the checklist. */
  renderPreArm() {
    if (!this.preArm) return;
    const recent = this.messages.filter((m) => m.source === "vehicle" && (m.prearm || m.severity <= 3)).slice(-3);
    this.preArm.hidden = !recent.length;
    this.preArm.replaceChildren(...recent.map((m) => el("div", { class: `fly-prearm-row is-${severityTone(m.severity)}` }, [
      dot(severityTone(m.severity)),
      el("span", { class: "fly-prearm-text", text: m.text }),
      simulatedTag(m, { small: true }),
    ].filter(Boolean))));
  }

  renderMission() {
    const m = this.model;
    const mission = m.mission;
    const planned = m.planned;
    const rows = [];
    if (!planned) {
      rows.push(el("span", { class: "fly-mission-name is-missing", text: "No mission planned" }),
        el("span", { class: "fly-mission-sub", text: "Plan one in Plan, or load a saved version below." }));
    } else {
      rows.push(el("span", { class: "fly-mission-name", text: `${planned.template || "mission"}${planned.source_version ? ` · v${planned.source_version}` : ""}` }),
        el("span", { class: "fly-mission-sub", text: [
          planned.waypoints != null ? `${planned.waypoints} wp` : null,
          distance(planned.path_distance_m),
          isNum(planned.altitude_m) ? `${fixed(planned.altitude_m)} m` : null,
        ].filter(Boolean).join(" · ") }));
    }
    let state = "not uploaded";
    let tone = "warn";
    if (mission && mission.on_vehicle) { state = `on the aircraft · ${mission.item_count} items`; tone = "ok"; }
    else if (!m.connected) { state = "no vehicle connected"; tone = "idle"; }
    rows.push(el("span", { class: "fly-mission-state" }, [dot(tone), el("span", { text: state })]));
    this.missionRow.replaceChildren(...rows);
    this.uploadBtn.disabled = !m.commands.canUpload;
    this.uploadBtn.title = m.commands.canUpload ? "Send the mission, geofence and rally points" : "Connect a vehicle and plan a mission first";

    const versions = m.versions || [];
    if (!versions.length) {
      this.versionSelect.replaceChildren(el("span", { class: "fly-hint", text: "No saved versions. Save one in Plan to queue it here." }));
      return;
    }
    const current = this.chosenVersion ?? String((planned && planned.source_version) || versions[versions.length - 1].version_num);
    const picker = select({
      label: "SAVED VERSION",
      value: current,
      options: [...versions].reverse().map((v) => ({
        value: String(v.version_num),
        label: `v${v.version_num} · ${v.mission_name || v.template || "mission"}`,
      })),
      onChange: (value) => { this.chosenVersion = value; },
    });
    picker.querySelector("select").setAttribute("data-control", "mission-version");
    const load = button({
      label: "Load", action: "Load Mission Version", variant: "ghost",
      onClick: () => this.run("Load Mission Version", { version: Number(picker.querySelector("select").value) }),
    });
    this.versionSelect.replaceChildren(picker, load);
  }

  sessionSummary(sessionAnswer, interrupted) {
    const session = sessionAnswer && sessionAnswer.session;
    const box = el("dl", { class: "fly-kv" });
    if (session && session.state) {
      box.append(...kv("STATE", session.state.status, { tone: ["aborted", "link_lost"].includes(session.state.status) ? "danger" : "" }),
        ...kv("COMMANDS", String(session.command_count ?? (session.commands || []).length)),
        ...kv("STARTED", timeOfDay(session.started_at) || null));
      if ((session.state.warnings || []).length) box.append(...kv("WARNING", session.state.warnings.slice(-1)[0], { tone: "warn" }));
    } else {
      box.append(...kv("STATE", (sessionAnswer && sessionAnswer.reason) || "no session"));
    }
    this.renderInterrupted(interrupted);
    return box;
  }

  renderInterrupted(answer) {
    const box = this.interruptedBox;
    if (!answer || !answer.recovered) { box.hidden = true; box.replaceChildren(); return; }
    box.hidden = false;
    const state = answer.state || {};
    box.replaceChildren(
      el("div", { class: "fly-interrupted-head" }, [dot("warn"), el("span", { text: "Interrupted flight" })]),
      el("p", { class: "fly-interrupted-text", text: answer.summary || `Recorded phase: ${state.phase || "unknown"}.` }),
      el("div", { class: "fly-row-btns" }, [
        button({ label: "From images", action: "Resume From Images", variant: "secondary", onClick: () => this.run("Resume From Images") }),
        button({ label: "Discard", action: "Discard Interrupted Flight", variant: "danger", onClick: () => this.run("Discard Interrupted Flight") }),
      ]),
    );
  }

  /* ================================================================== MAIN */

  buildMain() {
    this.mapHost = el("div", { class: "fly-map odk-map" });
    this.legend = el("div", { class: "fly-legend", "aria-label": "Route key" });
    this.banner = el("div", { class: "fly-banner", role: "status", hidden: "" });
    this.mapWrap = el("div", { class: "fly-map-wrap" }, [this.mapHost, this.legend, this.banner]);

    this.progress = el("div", { class: "fly-progress", "aria-label": "Mission progress" });
    this.abortCard = this.buildAbortCard();

    this.modeReadout = el("div", { class: "fly-mode" });
    this.armBtn = this.commandButton("Arm", "Arm", "lock");
    this.startBtn = this.commandButton("Start mission", "Start Mission", "play", "fly-cmd--start");
    this.pauseBtn = this.commandButton("Pause", "Pause", "pause");
    const moreBtn = el("button", {
      type: "button", class: "fly-cmd fly-cmd--icon", "data-control": "fly-more", "aria-label": "More commands",
      title: "Land, manual control, capture, disarm", "aria-haspopup": "menu",
    }, [glyph("more")]);
    this.more = null;
    moreBtn.addEventListener("click", () => this.openMore(moreBtn));
    this.rthBtn = this.commandButton("Return to home", "RTL", "home", "fly-cmd--rth");
    this.abortBtn = el("button", {
      type: "button", class: "fly-cmd fly-cmd--abort", "data-control": "abort-confirm",
      "aria-haspopup": "dialog", "aria-expanded": "false",
      "aria-label": "Abort",
    }, [glyph("stop"), el("span", { class: "fly-cmd-label", text: "Abort" })]);
    this.abortBtn.addEventListener("click", () => this.toggleAbort());
    this.commandBar = el("div", { class: "fly-commandbar", role: "toolbar", "aria-label": "Flight commands" }, [
      this.modeReadout, el("span", { class: "fly-divider", "aria-hidden": "true" }),
      this.armBtn, this.startBtn, this.pauseBtn, moreBtn,
      el("span", { class: "fly-spacer" }), this.rthBtn, this.abortBtn,
    ]);
    this.stage = el("div", { class: "fly-stage" }, [this.mapWrap, this.progress, this.abortCard, this.commandBar]);
    return this.stage;
  }

  commandButton(label, action, iconName, extra = "") {
    const b = el("button", { type: "button", class: `fly-cmd ${extra}`.trim(), "data-action": action, "aria-label": label },
      [glyph(iconName), el("span", { class: "fly-cmd-label", text: label })]);
    b.addEventListener("click", () => this.run(b.getAttribute("data-action")));
    return b;
  }

  openMore(anchor) {
    if (!this.more) {
      const items = [
        { label: "Land now", action: "Land" },
        { label: "Take manual control", action: "Manual Override" },
        { label: "Capture now", action: "Capture Now" },
        { label: "Disarm", action: "Disarm" },
      ];
      const body = el("div", { class: "odk-popover-body" }, items.map((item) => {
        const b = el("button", { type: "button", class: "odk-menu-item", "data-action": item.action, text: item.label });
        b.addEventListener("click", () => { this.more.close(); this.run(item.action); });
        return b;
      }));
      const node = el("div", { class: "odk-popover fly-more", role: "menu", "aria-label": "More commands", hidden: "" }, [body]);
      document.body.appendChild(node);
      const close = (e) => { if (!node.contains(e.target) && !anchor.contains(e.target)) api.close(); };
      const key = (e) => { if (e.key === "Escape") api.close(); };
      const api = {
        node,
        open: () => {
          node.hidden = false;
          const r = anchor.getBoundingClientRect();
          node.style.left = `${r.left}px`;
          node.style.top = `${r.top - node.offsetHeight - 4}px`;
          anchor.setAttribute("aria-expanded", "true");
          setTimeout(() => document.addEventListener("mousedown", close), 0);
          document.addEventListener("keydown", key);
        },
        close: () => {
          node.hidden = true;
          anchor.setAttribute("aria-expanded", "false");
          document.removeEventListener("mousedown", close);
          document.removeEventListener("keydown", key);
        },
      };
      this.more = api;
    }
    if (this.more.node.hidden) this.more.open(); else this.more.close();
  }

  buildAbortCard() {
    this.abortText = el("p", { class: "fly-abort-text" });
    this.holdFill = el("span", { class: "fly-hold-fill", "aria-hidden": "true" });
    this.holdBtn = el("button", {
      type: "button", class: "fly-hold", "data-action": "Abort",
      title: "Hold to send Abort to the aircraft",
    }, [this.holdFill, el("span", { class: "fly-hold-label", text: "Hold to abort" })]);
    const cancel = el("button", { type: "button", class: "odk-btn odk-btn--secondary fly-abort-cancel", "data-control": "abort-cancel", text: "Cancel" });
    cancel.addEventListener("click", () => this.toggleAbort(false));
    this.bindHold(this.holdBtn);
    return el("div", { class: "fly-abort", role: "dialog", "aria-label": "Confirm abort", hidden: "" }, [
      el("div", { class: "fly-abort-head" }, [dot("danger"), el("span", { text: "Abort mission and land in place?" })]),
      this.abortText,
      el("div", { class: "fly-abort-btns" }, [cancel, this.holdBtn]),
    ]);
  }

  toggleAbort(open) {
    const show = open === undefined ? this.abortCard.hidden : open;
    this.abortCard.hidden = !show;
    this.abortBtn.setAttribute("aria-expanded", show ? "true" : "false");
    if (show) {
      const t = this.model.telemetry;
      const wp = t && isNum(t.waypoint_index) ? ` at item ${t.waypoint_index}` : "";
      this.abortText.textContent = `The aircraft stops the route${wp} and descends at its ${this.model.linkState === "lost" ? "last known " : ""}position. `
        + "What it does exactly is the autopilot's abort behaviour.";
      this.holdBtn.focus();
    }
  }

  bindHold(buttonNode) {
    let started = 0;
    let frame = 0;
    const reset = () => { cancelAnimationFrame(frame); started = 0; this.holdFill.style.width = "0%"; };
    const step = () => {
      const p = Math.min(1, (performance.now() - started) / HOLD_TO_ABORT_MS);
      this.holdFill.style.width = `${(p * 100).toFixed(1)}%`;
      if (p >= 1) { reset(); this.sendAbort(); return; }
      frame = requestAnimationFrame(step);
    };
    const begin = (e) => {
      if (started || (e.type === "keydown" && ![" ", "Enter"].includes(e.key))) return;
      if (e.type === "keydown") e.preventDefault();
      started = performance.now();
      frame = requestAnimationFrame(step);
    };
    const end = (e) => { if (e && e.type === "keyup" && ![" ", "Enter"].includes(e.key)) return; reset(); };
    buttonNode.addEventListener("pointerdown", begin);
    buttonNode.addEventListener("keydown", begin);
    ["pointerup", "pointerleave", "pointercancel", "keyup", "blur"].forEach((n) => buttonNode.addEventListener(n, end));
    // A click without a hold (mouse tap, assistive "press") does nothing but say so.
    buttonNode.addEventListener("click", (e) => {
      e.preventDefault();
      if (!started) this.holdFill.parentElement.classList.add("is-nudged");
      setTimeout(() => this.holdFill.parentElement.classList.remove("is-nudged"), 400);
    });
  }

  /** The hold is the confirmation, so the action's own dialog is not shown again. */
  async sendAbort() {
    this.toggleAbort(false);
    const before = this.shell.confirmAll;
    this.shell.confirmAll = true;
    try {
      await this.run("Abort");
    } finally {
      this.shell.confirmAll = before;
    }
  }

  renderCommands() {
    const c = this.model.commands;
    const t = this.model.telemetry;
    const sim = simulatedTag(t || this.model.link, { small: true });
    this.modeReadout.replaceChildren(
      el("span", { class: "fly-mode-head" }, [el("span", { class: "odk-label", text: "MODE" }), sim].filter(Boolean)),
      el("span", { class: "fly-mode-value", text: c.mode ? `${c.mode} · ${c.armed ? "ARMED" : "DISARMED"}` : (c.lost ? "link lost" : "no vehicle") }),
    );
    this.modeReadout.title = this.model.control ? this.model.control.description || "" : "";
    // Arm / Disarm on one button: the state is shown, the next step is offered.
    const armKey = c.armed ? "Disarm" : "Arm";
    this.armBtn.setAttribute("data-action", armKey);
    this.armBtn.replaceChildren(glyph(c.armed ? "unlock" : "lock"), el("span", { class: "fly-cmd-label", text: c.armed ? "Armed" : "Arm" }));
    this.armBtn.setAttribute("aria-label", c.armed ? "Armed: disarm" : "Arm");
    this.armBtn.disabled = !(c.armed ? c.canDisarm : c.canArm);
    this.armBtn.title = c.armed ? (c.flying ? "Armed and flying" : "Armed. Press to disarm.") : "Arm the motors";
    this.startBtn.disabled = !c.canStart;
    this.startBtn.title = c.canStart ? "Start the uploaded mission (held to the preflight checklist)"
      : !c.connected ? "Connect a vehicle first" : !c.onVehicle ? "Upload the mission first" : "The mission is running";
    const resume = c.canResume;
    this.pauseBtn.setAttribute("data-action", resume ? "Resume" : "Pause");
    this.pauseBtn.replaceChildren(glyph(resume ? "play" : "pause"), el("span", { class: "fly-cmd-label", text: resume ? "Resume" : "Pause" }));
    this.pauseBtn.setAttribute("aria-label", resume ? "Resume" : "Pause");
    this.pauseBtn.disabled = !(c.canPause || c.canResume);
    this.rthBtn.disabled = !c.canReturn;
    this.abortBtn.disabled = !c.canAbort;
    if (!c.canAbort && !this.abortCard.hidden) this.toggleAbort(false);
  }

  renderProgress() {
    const m = this.model;
    const p = m.progress;
    const t = m.telemetry;
    const head = el("div", { class: "fly-progress-head" }, [
      el("span", { class: "odk-label", text: "MISSION PROGRESS" }),
    ]);
    if (!p || !m.mission || !m.mission.on_vehicle) {
      head.appendChild(el("span", { class: "fly-progress-wp is-missing", text: m.connected ? "no mission on the aircraft" : "no vehicle connected" }));
      this.progress.replaceChildren(head, el("div", { class: "fly-bar" }), el("div", { class: "fly-progress-foot" }, [
        el("span", { text: m.connected ? "Upload a mission from MISSION to follow it here." : "Connect a vehicle in CONNECTION." }),
      ]));
      return;
    }
    head.appendChild(el("span", { class: "fly-progress-wp", text: p.current !== null && p.total !== null ? `ITEM ${p.current} / ${p.total}` : "item not reported" }));
    const sim = simulatedTag(t, { small: true });
    if (sim) { sim.classList.add("fly-push"); head.appendChild(sim); }
    const fill = el("span", { class: "fly-bar-fill", style: `width:${((p.fraction || 0) * 100).toFixed(1)}%` });
    const bar = el("div", {
      class: "fly-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
      "aria-valuenow": ((p.fraction || 0) * 100).toFixed(0), "aria-label": "Route flown",
    }, [fill]);
    const speed = t && isNum(t.speed_mps) ? t.speed_mps : null;
    const eta = speed && speed > 0.5 && p.leftMetres > 0 ? duration(p.leftMetres / speed) : null;
    this.progress.replaceChildren(head, bar, el("div", { class: "fly-progress-foot" }, [
      el("span", { text: `${distance(p.flownMetres)} flown · ${distance(p.leftMetres)} left` }),
      el("span", { class: "fly-push", text: eta ? `ETA ${eta} @ ${fixed(speed)} m/s (est.)` : speed !== null ? "ETA: not moving" : "ETA not reported" }),
    ]));
  }

  renderBanner() {
    if (!this.banner) return;
    const m = this.model;
    const t = m.telemetry;
    const lost = m.linkState === "lost";
    const latest = [...this.messages].reverse().find((x) => x.severity <= 2 && x.source === "vehicle");
    const warnings = (m.session && m.session.state && m.session.state.warnings) || [];
    let title = null;
    let detail = null;
    let tone = "danger";
    if (lost) {
      title = `Link lost · ${clock(t.telemetry_age_s) || "unknown"} since last heartbeat`;
      detail = t.reason || "The values shown are the last ones the vehicle reported.";
    } else if (latest && Date.now() - new Date(latest.time_utc).getTime() < SLOW_MS * 12) {
      title = `Vehicle: ${latest.text}`;
      detail = `${latest.level} · ${timeOfDay(latest.time_utc)}`;
    } else if (warnings.length) {
      title = warnings[warnings.length - 1];
      detail = "flight session warning";
      tone = "warn";
    } else if (m.unread) {
      title = `${m.unread} unread alert${m.unread === 1 ? "" : "s"}`;
      detail = "notifications for this project";
      tone = "warn";
    }
    if (!title) { this.banner.hidden = true; return; }
    this.banner.hidden = false;
    this.banner.className = `fly-banner is-${tone}`;
    const details = el("button", { type: "button", class: "odk-btn odk-btn--secondary fly-banner-btn", "data-control": "alerts", text: "Details" });
    details.addEventListener("click", () => this.openAlerts(details));
    this.banner.replaceChildren(
      dot(tone),
      el("div", { class: "fly-banner-body" }, [
        el("span", { class: "fly-banner-title", text: title }),
        el("span", { class: "fly-banner-detail" }, [el("span", { text: detail }), simulatedTag(t || m.link, { small: true })].filter(Boolean)),
      ]),
      details,
    );
  }

  openAlerts(anchor) {
    if (this.alerts) { this.alerts.close(); this.alerts.node.remove(); this.alerts = null; return; }
    const m = this.model;
    const rows = [];
    for (const w of (m.session && m.session.state && m.session.state.warnings) || []) {
      rows.push(el("li", {}, [dot("warn"), el("span", { text: w })]));
    }
    for (const n of m.notifications || []) {
      rows.push(el("li", {}, [dot(n.level === "error" || n.severity === "error" ? "danger" : "warn"),
        el("span", { text: n.title ? `${n.title}${n.message ? `: ${n.message}` : ""}` : n.message || n.text || "notification" })]));
    }
    for (const x of this.messages.filter((y) => y.severity <= 3).slice(-5)) {
      rows.push(el("li", {}, [dot(severityTone(x.severity)), el("span", { text: `${timeOfDay(x.time_utc)} ${x.text}` })]));
    }
    const clear = button({
      label: "Clear alerts", action: "Clear Alerts", variant: "secondary",
      onClick: async () => { this.alerts.close(); this.alerts.node.remove(); this.alerts = null; await this.run("Clear Alerts"); },
    });
    const node = el("div", { class: "odk-popover fly-alerts", role: "dialog", "aria-label": "Alerts" }, [
      el("div", { class: "odk-popover-head odk-label", text: "ALERTS" }),
      el("ul", { class: "fly-alert-list" }, rows.length ? rows : [el("li", { class: "fly-empty", text: "No unread alerts." })]),
      el("div", { class: "fly-alerts-foot" }, [clear]),
    ]);
    document.body.appendChild(node);
    const r = anchor.getBoundingClientRect();
    node.style.left = `${Math.max(8, r.right - node.offsetWidth)}px`;
    node.style.top = `${r.bottom + 4}px`;
    const close = (e) => { if (!node.contains(e.target) && !anchor.contains(e.target) && this.alerts) { this.alerts.close(); node.remove(); this.alerts = null; } };
    setTimeout(() => document.addEventListener("mousedown", close), 0);
    this.alerts = { node, close: () => document.removeEventListener("mousedown", close) };
  }

  renderLegend() {
    const p = this.model.progress;
    const items = [];
    if (p && this.model.mission && this.model.mission.on_vehicle) {
      if (p.flown.length) items.push(["flown", `Flown · items ${p.firstSeq}–${p.flown[p.flown.length - 1].seq}`]);
      if (p.remaining.length) items.push(["planned", `To fly · items ${p.remaining[0].seq}–${p.lastSeq}`]);
    } else if (this.model.planned) {
      items.push(["plan", "Planned route (not on the aircraft)"]);
    }
    if (this.track.length > 1) items.push([this.model.simulated ? "track-sim" : "track", this.model.simulated ? "Track · SIMULATED" : "Track · as reported"]);
    this.legend.hidden = !items.length;
    this.legend.replaceChildren(...items.map(([kind, label]) => el("div", { class: "fly-legend-row" }, [
      el("span", { class: `fly-swatch fly-swatch--${kind}`, "aria-hidden": "true" }), el("span", { text: label }),
    ])));
  }

  /* ============================================================= INSPECTOR */

  buildInspector() {
    this.telemetryTab = el("div", { class: "fly-tab", "data-tab": "telemetry" });
    this.messagesTab = el("div", { class: "fly-tab", "data-tab": "messages" });
    this.eventsTab = el("div", { class: "fly-tab", "data-tab": "events" });
    this.tabs = panelTabs([
      { id: "telemetry", label: "Telemetry" },
      { id: "messages", label: "Messages" },
      { id: "events", label: "Events" },
    ], { active: this.tab, label: "Flight inspector", onSelect: (id) => this.showTab(id) });
    this.planBlock = live({
      calls: ["vehicle_mission"],
      every: SLOW_MS,
      empty: "No plan.",
      isEmpty: ([answer]) => !answer || !answer.planned,
      render: ([answer]) => this.planSummary(answer),
    });
    this.inspectorRoot = el("div", { class: "fly-inspector" }, [
      this.tabs,
      el("div", { class: "fly-tabs-body" }, [this.telemetryTab, this.messagesTab, this.eventsTab]),
      el("div", { class: "fly-plan" }, [this.planBlock]),
    ]);
    this.showTab(this.tab);
    return this.inspectorRoot;
  }

  showTab(id) {
    this.tab = id;
    remember(STORE.tab, id);
    for (const node of [this.telemetryTab, this.messagesTab, this.eventsTab]) node.hidden = node.getAttribute("data-tab") !== id;
    this.tabs.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", t.getAttribute("data-control") === `panel-tab:${id}` ? "true" : "false"));
    if (id === "events" && !this.eventsTab.firstChild) this.eventsTab.appendChild(eventsPanel((events) => this.eventRows(events)));
    if (id === "messages") this.renderMessages();
  }

  renderTelemetry() {
    const m = this.model;
    const t = m.telemetry;
    const reported = t && t.connected !== undefined;
    const lost = m.linkState === "lost";
    const head = el("div", { class: "fly-tab-head" }, [
      el("span", { class: "odk-label", text: lost ? "TELEMETRY · LAST KNOWN" : "TELEMETRY" }),
      simulatedTag(t, { small: true }),
    ].filter(Boolean));
    if (!reported) {
      this.telemetryTab.replaceChildren(head, el("p", { class: "fly-empty", text: (t && t.reason) || "No vehicle connected. Telemetry appears here once a link is open." }));
      return;
    }
    const big = (label, value, unit, size = "hero") => el("div", { class: `fly-stat fly-stat--${size}${lost ? " is-stale" : ""}` }, [
      el("span", { class: "fly-stat-label", text: label }),
      value === null ? el("span", { class: "fly-stat-value is-missing", text: NOT_REPORTED })
        : el("span", { class: "fly-stat-value" }, [value, unit ? el("span", { class: "fly-stat-unit", text: unit }) : null].filter(Boolean)),
    ]);
    const stats = el("div", { class: "fly-stats" }, [
      big("Altitude rel. home", fixed(t.altitude_rel_m), "m"),
      big("Ground speed", fixed(t.speed_mps), "m/s"),
      big("Heading", heading(t.heading_deg), "", "value"),
      big("Battery", isNum(t.battery_pct) ? fixed(t.battery_pct, 0) : null, "%", "value"),
    ]);
    const gps = isNum(t.gps_fix) ? `${GPS_FIX[t.gps_fix] || `fix ${t.gps_fix}`} · ${t.satellites ?? "?"} sats${isNum(t.hdop) ? ` · hdop ${fixed(t.hdop, 2)}` : ""}` : null;
    const facts = el("dl", { class: "fly-kv fly-kv--wide" }, [
      ...kv("LAT", latitude(t.latitude)),
      ...kv("LON", longitude(t.longitude)),
      ...kv("ALT AMSL", isNum(t.altitude_abs_m) ? `${fixed(t.altitude_abs_m)} m` : null),
      ...kv("GPS", gps, { tone: isNum(t.gps_fix) && t.gps_fix < 3 ? "warn" : "" }),
      ...kv("VOLTAGE", isNum(t.battery_v) ? `${fixed(t.battery_v, 2)} V` : null),
      ...kv("CONTROL", m.control ? m.control.control : null, { title: m.control ? m.control.description : "" }),
      ...kv("HOME", t.home_set === true ? "set" : t.home_set === false ? "not set" : null, { tone: t.home_set === false ? "warn" : "" }),
      ...kv("GIMBAL PITCH", null),
      ...kv("WIND", null),
    ]);
    // LINK: this station's own observations, one per poll.
    const state = m.linkState;
    const seg = el("div", { class: "fly-linkseg", role: "img", "aria-label": `Link ${state}` },
      ["ok", "degraded", "lost"].map((s) => el("span", { class: state === s ? `is-on is-${s}` : "", text: s.toUpperCase() })));
    const strip = el("div", { class: "fly-samples", "aria-hidden": "true" },
      Array.from({ length: LINK_SAMPLES }, (_, i) => {
        const s = this.samples[this.samples.length - LINK_SAMPLES + i];
        return el("span", { class: s ? `is-${s}` : "is-empty" });
      }));
    const rth = m.planned && isNum(m.planned.rth_altitude_m) ? `RTH at ${fixed(m.planned.rth_altitude_m, 0)} m (plan)` : null;
    const link = el("div", { class: "fly-link" }, [
      el("div", { class: "fly-tab-head" }, [
        el("span", { class: "odk-label", text: "LINK" }),
        el("span", { class: `fly-link-state is-${state}` }, [dot(state === "ok" ? "ok" : state === "degraded" ? "warn" : state === "lost" ? "danger" : "idle"), el("span", { text: state.toUpperCase() })]),
      ]),
      seg, strip,
      el("div", { class: "fly-samples-key" }, [
        el("span", { text: `station polls · last ${this.samples.length} s` }),
        simulatedTag(t, { small: true }),
      ].filter(Boolean)),
      el("dl", { class: "fly-kv fly-kv--wide" }, [
        ...kv("LINK QUALITY", isNum(t.link_quality_pct) ? `${fixed(t.link_quality_pct, 0)} % packets` : null),
        ...kv("LAST HEARD", isNum(t.telemetry_age_s) ? `${fixed(t.telemetry_age_s)} s ago` : null, { tone: lost ? "danger" : "" }),
        ...kv("FAILSAFE", rth, { tone: rth ? "warn" : "" }),
      ]),
    ]);
    this.telemetryTab.replaceChildren(head, stats, facts, link);
  }

  renderMessages() {
    if (!this.messagesTab || this.messagesTab.hidden) {
      const count = this.messages.length;
      const tab = this.tabs && this.tabs.querySelector("[data-control='panel-tab:messages']");
      if (tab) tab.textContent = count ? `Messages ${count}` : "Messages";
      return;
    }
    const rows = [...this.messages].reverse().map((x) => el("li", { class: `fly-msg is-${severityTone(x.severity)}` }, [
      el("span", { class: "fly-msg-time", text: timeOfDay(x.time_utc) }),
      el("span", { class: "fly-msg-level", text: x.level }),
      el("span", { class: "fly-msg-text", text: x.text }),
      el("span", { class: "fly-msg-src" }, [el("span", { text: x.source === "station" ? "station" : x.driver || "vehicle" }), simulatedTag(x, { small: true })].filter(Boolean)),
    ]));
    this.messagesTab.replaceChildren(
      el("div", { class: "fly-tab-head" }, [el("span", { class: "odk-label", text: "VEHICLE MESSAGES" }), el("span", { class: "fly-aside-num", text: String(this.messages.length) })]),
      rows.length ? el("ul", { class: "fly-msgs" }, rows)
        : el("p", { class: "fly-empty", text: this.model.connected ? "The vehicle has sent no status text yet." : "No vehicle connected." }),
    );
  }

  eventRows(events) {
    return el("ul", { class: "fly-events" }, events.slice(0, 80).map((e) => {
      const detail = e.details || e.detail || e.payload || {};
      const summary = typeof detail === "string" ? detail
        : Object.entries(detail).slice(0, 4).map(([k, v]) => `${k} ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ");
      return el("li", { class: "fly-event" }, [
        el("span", { class: "fly-msg-time", text: timeOfDay(e.created_at || e.timestamp || e.time_utc) }),
        el("span", { class: "fly-event-type", text: String(e.event_type || e.action || e.event || "").replace(/_/g, " ") }),
        el("span", { class: "fly-event-detail", text: summary, title: summary }),
      ]);
    }));
  }

  planSummary(answer) {
    const p = answer.planned;
    const overlap = isNum(p.front_overlap_pct) && isNum(p.side_overlap_pct) ? `${fixed(p.front_overlap_pct, 0)} % front · ${fixed(p.side_overlap_pct, 0)} % side` : null;
    const plan = [p.template, isNum(p.altitude_m) ? `${fixed(p.altitude_m, 0)} m` : null, isNum(p.estimated_gsd_cm) ? `${fixed(p.estimated_gsd_cm, 2)} cm GSD` : null].filter(Boolean).join(" · ");
    return el("dl", { class: "fly-kv fly-kv--plan" }, [
      ...kv("PLAN", plan || null),
      ...kv("OVERLAP", overlap),
      ...kv("SPEED", isNum(p.speed_m_s) ? `${fixed(p.speed_m_s)} m/s` : null),
      ...kv("ALT FRAME", p.altitude_frame ? String(p.altitude_frame).replace(/_/g, " ") : null, { tone: "warn" }),
    ]);
  }
}

/* ===================================================================== map */

/**
 * The aircraft and its route on the project map (desk and Field): the flown and the
 * remaining part of the mission the vehicle acknowledged, the track it reported, and
 * a cased arrow at its position that turns with its heading. Used by the Field Fly
 * screen too, so both modes draw the same answer.
 */
export class FlyMap {
  constructor(host, shell) {
    this.host = host;
    this.shell = shell;
    this.follow = recall(STORE.follow) === "on";
    this.model = null;
    this.track = [];
  }


  async mount() {
    const mounted = await mountProjectMap(this.host, {
      screen: "fly",
      run: (key, payload) => this.shell.runAction(key, payload),
      notify: (message, kind) => this.shell.toast(message, kind),
    });
    if (!mounted) return;
    this.map = mounted.map;
    const followBtn = iconButton({
      text: "", label: "Follow aircraft", control: "follow-aircraft", pressed: this.follow,
      onClick: () => {
        this.follow = !this.follow;
        remember(STORE.follow, this.follow ? "on" : "off");
        followBtn.setAttribute("aria-pressed", this.follow ? "true" : "false");
        this.renderMap();
      },
    });
    followBtn.appendChild(glyph("follow"));
    followBtn.classList.add("fly-follow");
    const stack = this.host.querySelector(".odk-mapf-tr");
    (stack || this.host).appendChild(el("div", { class: "odk-icon-group fly-follow-group" }, [followBtn]));
    const ready = () => { this.ensureLayers(); this.renderMap(); };
    if (this.map.isStyleLoaded()) ready(); else this.map.once("load", ready);
    this.map.on("styledata", () => this.ensureLayers());
  }

  ensureLayers() {
    const map = this.map;
    if (!map || !map.isStyleLoaded()) return;
    const field = document.documentElement.getAttribute("data-mode") === "field";
    const css = getComputedStyle(document.documentElement);
    const col = (name, fallback) => (css.getPropertyValue(name) || fallback).trim();
    const empty = { type: "FeatureCollection", features: [] };
    for (const id of ["odk-fly-flown", "odk-fly-remaining", "odk-fly-track"]) {
      if (!map.getSource(id)) map.addSource(id, { type: "geojson", data: empty });
    }
    const line = (id, paint, layout = {}) => {
      if (map.getLayer(id)) return;
      map.addLayer({ id, type: "line", source: id.replace(/-casing$/, ""), layout: { "line-cap": "round", "line-join": "round", ...layout }, paint });
    };
    if (field) {
      const ink = col("--field-ink", "#0A0A0A");
      line("odk-fly-remaining-casing", { "line-color": ink, "line-width": 8 });
      line("odk-fly-remaining", { "line-color": "#FFFFFF", "line-width": 4 });
      line("odk-fly-flown-casing", { "line-color": ink, "line-width": 8 });
      line("odk-fly-flown", { "line-color": col("--action", "#FFD400"), "line-width": 4 });
      line("odk-fly-track", { "line-color": col("--sim", "#C084FC"), "line-width": 3, "line-dasharray": [1, 1.5] });
    } else {
      line("odk-fly-remaining-casing", { "line-color": col("--map-well", "#0A0D11"), "line-width": 4, "line-opacity": 0.6 });
      line("odk-fly-remaining", { "line-color": col("--accent", "#2F86FF"), "line-width": 2 });
      line("odk-fly-flown", { "line-color": col("--text-2", "#B4BCC8"), "line-width": 1.5, "line-dasharray": [3, 2] });
      line("odk-fly-track", { "line-color": col("--sim", "#C084FC"), "line-width": 1.5, "line-opacity": 0.9 });
    }
  }

  update(model, track) {
    this.model = model;
    this.track = track;
    this.renderMap();
  }

  renderMap() {
    const map = this.map;
    if (!map || !map.isStyleLoaded() || !this.model) return;
    this.ensureLayers();
    const m = this.model;
    const p = m.progress;
    const lineOf = (points) => ({
      type: "FeatureCollection",
      features: points.length > 1 ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: points } }] : [],
    });
    const onVehicle = Boolean(p && m.mission && m.mission.on_vehicle);
    const flown = onVehicle ? p.flown.map((q) => [q.lon, q.lat]) : [];
    const remaining = onVehicle ? p.remaining.map((q) => [q.lon, q.lat]) : [];
    if (onVehicle && p.started && p.position && remaining.length) {
      // The leg in progress: from the last item reached, through the aircraft, onward.
      flown.push([p.position.lon, p.position.lat]);
      remaining.unshift([p.position.lon, p.position.lat]);
    }
    if (onVehicle && this.fittedFor !== m.mission.item_count) this.fitRoute(p);
    map.getSource("odk-fly-flown")?.setData(lineOf(flown));
    map.getSource("odk-fly-remaining")?.setData(lineOf(remaining));
    map.getSource("odk-fly-track")?.setData(lineOf(this.track));
    const trackColour = m.simulated ? null : getComputedStyle(document.documentElement).getPropertyValue("--text").trim();
    if (trackColour && map.getLayer("odk-fly-track")) map.setPaintProperty("odk-fly-track", "line-color", trackColour);
    this.renderAircraft();
  }

  renderAircraft() {
    const t = this.model.telemetry;
    const maplibre = window.maplibregl;
    const has = t && isNum(t.latitude) && isNum(t.longitude) && (t.latitude || t.longitude) && t.connected !== undefined;
    if (!has || !maplibre || !this.map) { if (this.marker) { this.marker.remove(); this.marker = null; } return; }
    if (!this.marker) {
      this.arrow = el("span", { class: "fly-aircraft-arrow" });
      this.arrow.innerHTML = '<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">'
        + '<path class="fly-arrow-casing" d="M14 2L24 25L14 19L4 25Z"></path>'
        + '<path class="fly-arrow-fill" d="M14 2L24 25L14 19L4 25Z"></path></svg>';
      this.callout = el("span", { class: "fly-aircraft-label" });
      this.markerNode = el("div", { class: "fly-aircraft", role: "img" }, [this.arrow, this.callout]);
      this.marker = new maplibre.Marker({ element: this.markerNode, anchor: "center" }).setLngLat([t.longitude, t.latitude]).addTo(this.map);
    }
    this.marker.setLngLat([t.longitude, t.latitude]);
    this.arrow.style.transform = `rotate(${(t.heading_deg || 0) - this.map.getBearing()}deg)`;
    const lost = this.model.linkState === "lost";
    this.markerNode.classList.toggle("is-lost", lost);
    this.markerNode.classList.toggle("is-sim", t.is_simulated === true);
    const wp = isNum(t.waypoint_index) && isNum(t.waypoint_total) && t.waypoint_total ? `item ${t.waypoint_index} / ${t.waypoint_total}` : null;
    this.callout.replaceChildren(
      el("span", { text: [t.driver, wp].filter(Boolean).join(" · ") }),
      el("span", { text: [isNum(t.altitude_rel_m) ? `${fixed(t.altitude_rel_m)} m` : null, heading(t.heading_deg), isNum(t.speed_mps) ? `${fixed(t.speed_mps)} m/s` : null].filter(Boolean).join(" · ") }),
      el("span", { class: "fly-aircraft-tags" }, [
        simulatedTag(t, { small: true }),
        lost ? el("span", { class: "fly-lastknown", text: "LAST KNOWN" }) : null,
      ].filter(Boolean)),
    );
    this.markerNode.setAttribute("aria-label", `Aircraft ${latitude(t.latitude)} ${longitude(t.longitude)}${lost ? ", last known position" : ""}`);
    if (this.follow && !lost) this.map.easeTo({ center: [t.longitude, t.latitude], duration: FAST_MS * 0.8 });
  }


  /** Frame the route the aircraft carries, clear of the map furniture, once per upload. */
  fitRoute(p) {
    const points = [...p.flown, ...p.remaining];
    if (p.position) points.push(p.position);
    if (points.length < 2 || !this.map) return;
    this.fittedFor = this.model.mission.item_count;
    const lons = points.map((q) => q.lon);
    const lats = points.map((q) => q.lat);
    const field = document.documentElement.getAttribute("data-mode") === "field";
    try {
      this.map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], {
        padding: field ? { top: 200, bottom: 110, left: 40, right: 110 } : { top: 110, bottom: 110, left: 72, right: 96 },
        duration: 0, maxZoom: 19,
      });
    } catch { /* a map without a size yet keeps its view */ }
  }

  destroy() {
    if (this.marker) { this.marker.remove(); this.marker = null; }
  }
}
