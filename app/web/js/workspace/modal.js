/*
 * Dialogs the application draws itself.
 *
 * Every action that needed a value called window.prompt. In a browser that works, which
 * is why the click harness and the Edge probes passed. In the desktop shell it does not:
 * WebView2 does not implement window.prompt, and pywebview does not route script dialogs
 * either -- the call returns null immediately, so the action read "no name given" and
 * stopped. Twenty-six buttons appeared to do nothing at all, and no error was raised
 * anywhere, because returning null is exactly what a cancelled prompt looks like.
 *
 * So the dialogs are DOM. That also fixes things window.prompt could never do: several
 * fields at once, a real list to choose from, and a reveal box a token can be selected
 * out of rather than read off the screen and typed.
 *
 * Every function resolves rather than throwing on cancel. A cancelled dialog is a normal
 * outcome, not an error, and an action that treats it as one would report a failure the
 * user deliberately chose.
 */

const KEY_ESCAPE = "Escape";
const KEY_ENTER = "Enter";

function backdrop() {
  const element = document.createElement("div");
  element.className = "modal-backdrop";
  return element;
}

function panel(title) {
  const box = document.createElement("div");
  box.className = "modal";
  const heading = document.createElement("div");
  heading.className = "modal-title";
  heading.textContent = title;
  box.appendChild(heading);
  return box;
}

function buttons(box, { okLabel = "OK", cancelLabel = "Cancel" }, onOk, onCancel) {
  const row = document.createElement("div");
  row.className = "modal-actions";

  const cancel = document.createElement("button");
  cancel.className = "tbtn";
  cancel.textContent = cancelLabel;
  cancel.onclick = onCancel;

  const confirm = document.createElement("button");
  confirm.className = "tbtn primary";
  confirm.textContent = okLabel;
  confirm.onclick = onOk;

  row.append(cancel, confirm);
  box.appendChild(row);
  return confirm;
}

function mount(box, onCancel) {
  const shade = backdrop();
  shade.appendChild(box);
  document.body.appendChild(shade);
  shade.addEventListener("click", (event) => {
    if (event.target === shade) onCancel();
  });
  const onKey = (event) => {
    if (event.key === KEY_ESCAPE) onCancel();
  };
  document.addEventListener("keydown", onKey);
  return () => {
    document.removeEventListener("keydown", onKey);
    shade.remove();
  };
}

/**
 * Ask for one or more values.
 *
 * `fields` is [{key, label, value, type, options}]. Resolves to an object keyed by
 * field key, or null if the user cancelled.
 */
export function ask(title, fields, options = {}) {
  return new Promise((resolve) => {
    const box = panel(title);
    const inputs = new Map();

    for (const field of fields) {
      const row = document.createElement("label");
      row.className = "modal-field";
      const label = document.createElement("span");
      label.textContent = field.label || field.key;
      row.appendChild(label);

      const input = field.options
        ? document.createElement("select")
        : document.createElement("input");
      if (field.options) {
        for (const option of field.options) {
          const choice = document.createElement("option");
          choice.value = String(option);
          choice.textContent = String(option);
          if (String(option) === String(field.value)) choice.selected = true;
          input.appendChild(choice);
        }
      } else {
        input.type = field.type || "text";
        input.value = field.value == null ? "" : String(field.value);
        if (field.placeholder) input.placeholder = field.placeholder;
      }
      row.appendChild(input);
      box.appendChild(row);
      inputs.set(field.key, input);
    }

    let close = () => {};
    const finish = (value) => { close(); resolve(value); };
    const submit = () => {
      const answer = {};
      for (const [key, input] of inputs) answer[key] = input.value;
      finish(answer);
    };

    buttons(box, options, submit, () => finish(null));
    box.addEventListener("keydown", (event) => {
      if (event.key === KEY_ENTER && event.target.tagName !== "TEXTAREA") submit();
    });
    close = mount(box, () => finish(null));

    const first = inputs.values().next().value;
    if (first) setTimeout(() => first.focus(), 0);
  });
}

/** A single value, the common case. Resolves to a string or null. */
export async function askOne(title, label, value = "") {
  const answer = await ask(title, [{ key: "value", label, value }]);
  return answer ? answer.value : null;
}

/** Pick from a list. `options` is [{label, value, hint}]. Resolves to a value or null. */
export function choose(title, options) {
  return new Promise((resolve) => {
    const box = panel(title);
    const list = document.createElement("div");
    list.className = "modal-list";

    let close = () => {};
    const finish = (value) => { close(); resolve(value); };

    if (!options.length) {
      const empty = document.createElement("div");
      empty.className = "modal-empty";
      empty.textContent = "Nothing to choose from yet.";
      list.appendChild(empty);
    }

    for (const option of options) {
      const row = document.createElement("button");
      row.className = "modal-row";
      const label = document.createElement("span");
      label.textContent = option.label;
      row.appendChild(label);
      if (option.hint) {
        const hint = document.createElement("span");
        hint.className = "modal-hint";
        hint.textContent = option.hint;
        row.appendChild(hint);
      }
      row.onclick = () => finish(option.value);
      list.appendChild(row);
    }

    box.appendChild(list);
    buttons(box, { okLabel: "Close" }, () => finish(null), () => finish(null));
    close = mount(box, () => finish(null));
  });
}

/** A yes/no question. Resolves true or false, never throws. */
export function confirmAsk(question) {
  return new Promise((resolve) => {
    const box = panel("Confirm");
    const text = document.createElement("div");
    text.className = "modal-text";
    text.textContent = question;
    box.appendChild(text);

    let close = () => {};
    const finish = (value) => { close(); resolve(value); };
    buttons(box, { okLabel: "Yes", cancelLabel: "No" }, () => finish(true), () => finish(false));
    close = mount(box, () => finish(false));
  });
}

/**
 * Show something the application will not show again.
 *
 * Selectable and pre-selected: a share token or a signing secret is stored only as a
 * hash, so if it is not copied out of this box it is gone. A toast would disappear
 * before it could be read.
 */
export function reveal(title, body) {
  return new Promise((resolve) => {
    const box = panel(title);
    const area = document.createElement("textarea");
    area.className = "modal-reveal";
    area.readOnly = true;
    area.value = body;
    area.rows = Math.min(8, String(body).split("\n").length + 2);
    box.appendChild(area);

    let close = () => {};
    const finish = () => { close(); resolve(); };
    buttons(box, { okLabel: "Done", cancelLabel: "Copy" }, finish, () => {
      area.select();
      try {
        document.execCommand("copy");
      } catch {
        /* Selection is left in place so it can be copied by hand. */
      }
    });
    close = mount(box, finish);
    setTimeout(() => { area.focus(); area.select(); }, 0);
  });
}
