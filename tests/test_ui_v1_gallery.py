"""The v1 design system, rendered in a real browser, in both modes, with nothing fetched.

odk.html?gallery draws every component of the approved SysTokens board through
js/odk/components.js, once in a desk board and once in a field board. This opens it in
headless Edge/Chromium through tests/ui_live_bridge.py (the harness the live UI tests
use) and checks what a reviewer would otherwise check by eye:

  * every component rendered, with a size, in both boards;
  * Field Daylight's rule that every touch target is at least 64 px, by computed style;
  * text tokens meet WCAG AA (4.5:1) against the surfaces they are used on;
  * the vendored fonts actually load, from vendor/fonts;
  * OFFLINE: no request leaves 127.0.0.1 -- no Google Fonts, no CDN. The artboards load
    their fonts from fonts.googleapis.com; a copy of that line would pass every visual
    check on a connected machine and leave the app in a fallback font in the field.
  * every screen module imports, and odk.html without ?gallery opens the shell's desk frame
    (the shell itself is tested in test_ui_v1_shell.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("websocket")
pytest.importorskip("requests")

from ui_live_bridge import Browser, chromium_path, serve  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "app" / "web"


class RecordingBrowser(Browser):
    """Browser that keeps every network request the page makes (CDP Network domain)."""

    def __init__(self, executable: str):
        self.requests: list[str] = []
        self.failed: list[str] = []
        super().__init__(executable)
        self._send("Network.enable")
        # The bridge serves with no-store already; this keeps a re-run honest too.
        self._send("Network.setCacheDisabled", cacheDisabled=True)

    def _send(self, method: str, **params):  # noqa: D401 - same contract as Browser._send
        self._id += 1
        wanted = self._id
        self.socket.send(json.dumps({"id": wanted, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            event = message.get("method")
            if event == "Network.requestWillBeSent":
                self.requests.append(message["params"]["request"]["url"])
            elif event == "Network.loadingFailed":
                self.failed.append(message["params"].get("errorText", ""))
            if message.get("id") == wanted:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})


@pytest.fixture(scope="module")
def page():
    executable = chromium_path()
    if executable is None:
        pytest.skip("No Chromium-family browser is installed to drive the UI.")
    server, base = serve(object())  # the gallery calls no Api method
    browser = RecordingBrowser(executable)
    try:
        browser._send("Emulation.setDeviceMetricsOverride", width=1440, height=900,
                      deviceScaleFactor=1, mobile=False)
        browser.open(base + "/odk.html?gallery", settle_s=0.5)
        browser.wait_for("window.__odkReady === true || window.__odkBootError")
        assert browser.js("window.__odkBootError || null") is None, browser.js("window.__odkBootError")
        browser.js("document.fonts.ready.then(() => true)")
        browser.wait_for("document.querySelector('[data-specimen=open]') !== null")
        yield browser, base
    finally:
        browser.close()
        server.shutdown()


def _rgb(value: str) -> tuple[float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
    nums = [float(x) for x in re.findall(r"[\d.]+", value)[:3]]
    return tuple(n / 255 for n in nums)  # type: ignore[return-value]


def contrast(a: str, b: str) -> float:
    def lum(rgb):
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    la, lb = sorted((lum(_rgb(a)), lum(_rgb(b))), reverse=True)
    return (la + 0.05) / (lb + 0.05)


class TestEveryComponentRenders:
    def test_both_boards_render_every_component(self, page) -> None:
        browser, _ = page
        result = browser.js("""(() => {
          const out = {};
          const names = window.__odkGallery.components;
          const fieldOnly = window.__odkGallery.fieldOnly;
          for (const mode of ['desk', 'field']) {
            const board = document.querySelector(`[data-board=${mode}]`);
            const want = mode === 'field' ? names.concat(fieldOnly) : names;
            out[mode] = want.filter((n) => {
              const node = board.querySelector(`[data-component="${n}"]`);
              if (!node) return true;
              const r = node.getBoundingClientRect();
              return !(r.width > 0 && r.height > 0);
            });
          }
          return out;
        })()""")
        assert result == {"desk": [], "field": []}, f"components missing or empty: {result}"

    def test_every_action_button_carries_its_key(self, page) -> None:
        browser, _ = page
        unmarked = browser.js("""[...document.querySelectorAll('[data-gallery] button, .odk-popover button')]
            .filter((b) => !b.hasAttribute('data-action') && !b.hasAttribute('data-control'))
            .map((b) => b.outerHTML.slice(0, 120))""")
        assert unmarked == [], unmarked

    def test_field_mode_really_is_field_mode(self, page) -> None:
        browser, _ = page
        styles = browser.js("""(() => {
          const pick = (mode, sel, prop) => getComputedStyle(
            document.querySelector(`[data-board=${mode}] ${sel}`))[prop];
          return {
            deskFont: pick('desk', '.odk-btn', 'fontFamily'),
            fieldFont: pick('field', '.odk-btn', 'fontFamily'),
            deskBg: getComputedStyle(document.querySelector('[data-board=desk]')).backgroundColor,
            fieldBg: getComputedStyle(document.querySelector('[data-board=field]')).backgroundColor,
            fieldPrimary: pick('field', '.odk-btn--primary', 'backgroundColor'),
            deskPrimary: pick('desk', '.odk-btn--primary', 'backgroundColor'),
            fieldHeader: pick('field', '.odk-field-header', 'height'),
            fieldTabbar: pick('field', '.odk-field-tabbar', 'height'),
            activeFieldTab: pick('field', '.odk-field-tab[aria-current=page]', 'backgroundColor'),
          };
        })()""")
        assert "IBM Plex Sans" in styles["deskFont"]
        assert "Atkinson Hyperlegible" in styles["fieldFont"]
        assert styles["deskBg"] == "rgb(14, 17, 22)"
        assert styles["fieldBg"] == "rgb(255, 255, 255)"
        assert styles["deskPrimary"] == "rgb(47, 134, 255)"
        assert styles["fieldPrimary"] == "rgb(255, 212, 0)"
        assert styles["fieldHeader"] == "72px"
        assert styles["fieldTabbar"] == "88px"
        assert styles["activeFieldTab"] == "rgb(255, 212, 0)"


class TestFieldTouchTargets:
    def test_every_field_target_is_at_least_64px(self, page) -> None:
        browser, _ = page
        small = browser.js("""(() => {
          const board = document.querySelector('[data-board=field]');
          const targets = [
            ...board.querySelectorAll('button, a[href], label.odk-check-row, label.odk-layer-row, .odk-input'),
          ].filter((n) => !n.closest('.odk-input') || n.classList.contains('odk-input'));
          const out = [];
          for (const n of targets) {
            const cs = getComputedStyle(n);
            const h = parseFloat(cs.height), w = parseFloat(cs.width);
            if (!(h >= 64 && w >= 64)) {
              out.push(`${n.tagName.toLowerCase()}.${[...n.classList].join('.')} "${(n.textContent || n.getAttribute('aria-label') || '').trim().slice(0, 30)}" ${w}x${h}`);
            }
          }
          return { count: targets.length, small: out };
        })()""")
        assert small["count"] >= 40, small
        assert small["small"] == [], "field targets under 64px:\n" + "\n".join(small["small"])

    def test_the_desk_board_keeps_desk_sizes(self, page) -> None:
        browser, _ = page
        sizes = browser.js("""(() => {
          const h = (sel) => getComputedStyle(document.querySelector(`[data-board=desk] ${sel}`)).height;
          return { button: h('.odk-btn'), icon: h('.odk-icon-btn'), input: h('.odk-input'), row: h('.odk-layer-row') };
        })()""")
        assert sizes == {"button": "36px", "icon": "36px", "input": "30px", "row": "36px"}


class TestContrast:
    # (text token, background token) pairs the components actually use.
    DESK = [("--text", "--bg"), ("--text", "--surface-1"), ("--text", "--surface-2"),
            ("--text", "--surface-3"), ("--text-2", "--bg"), ("--text-2", "--surface-1"),
            ("--text-2", "--surface-2"), ("--muted", "--bg"), ("--muted", "--surface-1"),
            ("--muted", "--surface-2"), ("--on-accent", "--accent"), ("--on-action", "--action"),
            ("--accent", "--bg"), ("--accent", "--surface-1"), ("--accent-hover", "--surface-2"),
            ("--ok", "--surface-1"), ("--warn", "--surface-1"), ("--danger", "--surface-1"),
            ("--danger", "--surface-2"), ("--sim", "--surface-1")]
    FIELD = [("--text", "--bg"), ("--text", "--surface-2"), ("--on-action", "--action"),
             ("--on-selection", "--selection"), ("--on-ok", "--ok"), ("--on-danger", "--danger"),
             ("--sim-text", "--sim"), ("--link", "--bg"), ("--text", "--warn")]

    @pytest.mark.parametrize("mode", ["desk", "field"])
    def test_text_tokens_meet_4_5_to_1(self, page, mode) -> None:
        browser, _ = page
        pairs = self.DESK if mode == "desk" else self.FIELD
        tokens = sorted({t for pair in pairs for t in pair})
        values = browser.js(f"""(() => {{
          const cs = getComputedStyle(document.querySelector('[data-board={mode}]'));
          return Object.fromEntries({json.dumps(tokens)}.map((t) => [t, cs.getPropertyValue(t).trim()]));
        }})()""")
        failures = []
        for fg, bg in pairs:
            ratio = contrast(values[fg], values[bg])
            if ratio < 4.5:
                failures.append(f"{fg} {values[fg]} on {bg} {values[bg]}: {ratio:.2f}")
        assert not failures, f"{mode}: " + "; ".join(failures)

    def test_rendered_text_meets_4_5_to_1(self, page) -> None:
        """The tokens are right; this checks the components use them on the right fills."""
        browser, _ = page
        samples = browser.js("""(() => {
          const opaque = (n) => {
            for (let e = n; e; e = e.parentElement) {
              const bg = getComputedStyle(e).backgroundColor;
              if (bg && bg !== 'transparent' && !/rgba\\(.*,\\s*0\\)$/.test(bg)) {
                const a = bg.match(/rgba\\([^)]*,\\s*([\\d.]+)\\)/);
                if (!a || parseFloat(a[1]) >= 0.99) return bg;
              }
            }
            return 'rgb(255, 255, 255)';
          };
          const sel = '.odk-btn:not(:disabled), .odk-chip, .odk-tag, .odk-tab, .odk-panel-tab, .odk-toast-title, '
            + '.odk-toast-detail, .odk-label, .odk-layer-name, .odk-stat-value, .odk-table td, .odk-status, .odk-field-tab';
          return [...document.querySelectorAll(`[data-board] :is(${sel})`)]
            .filter((n) => n.textContent.trim() && !n.closest('.odk-layer-row--disabled, .odk-check-row--disabled'))
            .map((n) => ({ text: n.textContent.trim().slice(0, 24), mode: n.closest('[data-board]').dataset.board,
                           fg: getComputedStyle(n).color, bg: opaque(n) }));
        })()""")
        assert len(samples) > 80
        low = [f"{s['mode']} '{s['text']}' {s['fg']} on {s['bg']}: {contrast(s['fg'], s['bg']):.2f}"
               for s in samples if contrast(s["fg"], s["bg"]) < 4.5]
        assert not low, "\n".join(low)


class TestOffline:
    def test_no_request_leaves_loopback(self, page) -> None:
        browser, base = page
        outside = [u for u in browser.requests
                   if not (u.startswith(base + "/") or u.startswith("data:") or u.startswith("blob:"))]
        assert browser.requests, "no requests were recorded at all"
        assert not outside, f"requests left 127.0.0.1: {outside}"

    def test_the_vendored_fonts_load(self, page) -> None:
        browser, base = page
        loaded = browser.js("""(async () => {
          const want = [['IBM Plex Sans', 400], ['IBM Plex Sans', 500], ['IBM Plex Sans', 600],
                        ['IBM Plex Mono', 400], ['IBM Plex Mono', 500],
                        ['Atkinson Hyperlegible', 400], ['Atkinson Hyperlegible', 700]];
          await Promise.all(want.map(([f, w]) => document.fonts.load(`${w} 16px "${f}"`, 'Ag1')));
          const faces = [...document.fonts].filter((f) => f.status === 'loaded')
            .map((f) => `${f.family.replace(/"/g, '')}|${f.weight}`);
          return want.filter(([f, w]) => !faces.includes(`${f}|${w}`)).map(([f, w]) => `${f} ${w}`);
        })()""")
        assert loaded == [], f"fonts that did not load: {loaded}"
        fonts = [u for u in browser.requests if u.endswith(".woff2")]
        assert fonts and all(u.startswith(base + "/vendor/fonts/") for u in fonts), fonts

    def test_the_sources_point_nowhere_else(self) -> None:
        files = [WEB / "odk.html", *(WEB / "css" / "odk").glob("*.css"), *(WEB / "js" / "odk").rglob("*.js")]
        for path in files:
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"https?://", text), f"{path.name} names a network URL"
            assert "googleapis" not in text and "gstatic" not in text and "/_blob/" not in text, path.name
        fonts_css = (WEB / "css" / "odk" / "fonts.css").read_text(encoding="utf-8")
        urls = re.findall(r'url\("([^"]+)"\)', fonts_css)
        assert len(urls) == 9
        for url in urls:
            assert url.startswith("../../vendor/fonts/"), url
            assert (WEB / "css" / "odk" / url).resolve().is_file(), url
        for licence in ("OFL-IBM-Plex-Sans.txt", "OFL-IBM-Plex-Mono.txt", "OFL-Atkinson-Hyperlegible.txt"):
            assert "SIL Open Font License" in (WEB / "vendor" / "fonts" / licence).read_text(encoding="utf-8")


class TestTheFrameAndTheModules:
    def test_every_screen_module_imports(self, page) -> None:
        browser, _ = page
        loaded = browser.js("""import('./js/odk/screens/index.js').then(async (m) => {
          const all = await m.loadAll();
          return Object.entries(all).map(([k, mod]) => [k, Object.keys(mod).sort()]);
        })""")
        names = dict((k, v) for k, v in loaded)
        assert len(names) == 9 * 4 + 5
        for key, exports in names.items():
            if key.startswith("field/"):
                assert {"render", "SCREEN", "MODE", "CALLS"} <= set(exports), key
            elif key.endswith("/surface"):
                assert {"SCREEN", "PLACEMENTS"} <= set(exports), key
            elif key.endswith("/model"):
                assert {"CALLS", "derive"} <= set(exports), key
            elif key.endswith("/desk"):
                assert {"render", "SCREEN"} <= set(exports), key
            else:
                assert exports == ["ACTIONS"], key

    def test_a_button_without_a_key_is_refused(self, page) -> None:
        browser, _ = page
        message = browser.js("""import('./js/odk/components.js').then((c) => {
          try { c.button({ label: 'Export' }); return 'accepted'; } catch (e) { return e.message; }
        })""")
        assert "data-action" in message

    def test_tags_follow_the_api_flag(self, page) -> None:
        browser, _ = page
        tags = browser.js("""import('./js/odk/components.js').then((c) => [
          c.simulatedTag({ is_simulated: true })?.textContent ?? null,
          c.simulatedTag({ is_simulated: false }), c.simulatedTag({}), c.simulatedTag(null),
          c.demoTag({ demo: true })?.textContent ?? null, c.demoTag({ is_demo: true })?.textContent ?? null,
          c.demoTag({ demo: 'yes' }), c.demoTag({}),
          c.stat({ label: 'Battery', value: null }).textContent,
        ])""")
        assert tags == ["SIMULATED", None, None, None, "DEMO RECORD", "DEMO RECORD", None, None,
                        "Batterynot reported"]

    def test_the_shell_frame_opens(self, page) -> None:
        browser, base = page
        before = len(browser.requests)
        browser.open(base + "/odk.html?mode=desk", settle_s=0.3)
        browser.wait_for("window.__odkReady === true || window.__odkBootError")
        assert browser.js("window.__odkBootError || null") is None
        frame = browser.js("""(() => {
          const f = document.querySelector('.odk-frame[data-frame=desk]');
          const h = (s) => getComputedStyle(f.querySelector(s)).height;
          const w = (s) => getComputedStyle(f.querySelector(s)).width;
          return { header: h('[data-region=header]'), footer: h('[data-region=footer]'),
                   left: w('.odk-frame-left'), inspector: w('.odk-frame-inspector'),
                   text: f.querySelector('[data-region=main]').textContent };
        })()""")
        # The screens are still stubs: they render nothing rather than anything invented.
        assert frame == {"header": "48px", "footer": "28px", "left": "264px", "inspector": "312px", "text": ""}
        outside = [u for u in browser.requests[before:]
                   if not (u.startswith(base + "/") or u.startswith("data:") or u.startswith("blob:"))]
        assert not outside, outside
