"""Generate docs/sdk/reference.md from the SDK's own signatures and docstrings.

    python tools/gen_sdk_reference.py            # rewrite docs/sdk/reference.md
    python tools/gen_sdk_reference.py --check    # exit 1 if the file is out of date

The reference covers exactly the public API: the names in ``opendronekit.__all__`` and in
``opendronekit.plugins.__all__``. ``tests/test_sdk_docs.py`` runs ``--check``, so a change
to a public signature or docstring that is not regenerated fails the suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import inspect
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "sdk" / "reference.md"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _doc(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    if dataclasses.is_dataclass(obj) and doc.startswith(obj.__name__ + "("):
        return ""                                  # the auto-generated dataclass repr
    return doc.strip()


def _annotation(value: Any) -> str:
    if isinstance(value, str):
        return value                                   # postponed annotations are strings
    return inspect.formatannotation(value)


def _signature(obj: Any) -> str:
    """``(a: int, *, b: str = 'x') -> T`` with annotations exactly as written in the code."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    parts: list[str] = []
    star_done = False
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind is param.KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        prefix = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(param.kind, "")
        if param.kind is param.VAR_POSITIONAL:
            star_done = True
        text = prefix + param.name
        if param.annotation is not param.empty:
            text += f": {_annotation(param.annotation)}"
        if param.default is not param.empty:
            text += (" = " if param.annotation is not param.empty else "=") + repr(param.default)
        parts.append(text)
    out = f"({', '.join(parts)})"
    if sig.return_annotation is not sig.empty and _annotation(sig.return_annotation) != "None":
        out += f" -> {_annotation(sig.return_annotation)}"
    return out


def _anchor(name: str) -> str:
    return name.lower().replace(".", "").replace("_", "-")


def _members(cls: type) -> list[tuple[str, Any]]:
    out = []
    for name, value in cls.__dict__.items():
        if name.startswith("_") and name not in ("__init__",):
            continue
        if name == "__init__":
            continue
        if isinstance(value, (staticmethod, classmethod)):
            out.append((name, value.__func__))
        elif isinstance(value, property) or inspect.isfunction(value):
            out.append((name, value))
    return out


def _class_section(qual: str, cls: type) -> list[str]:
    lines = [f"### `{qual}`", ""]
    if issubclass(cls, enum.Enum):
        lines += [_doc(cls) or "An enumeration.", "",
                  "| Member | Value |", "|---|---|"]
        lines += [f"| `{m.name}` | `{m.value!r}` |" for m in cls]
        return lines + [""]
    kind = "dataclass" if dataclasses.is_dataclass(cls) else (
        "protocol" if getattr(cls, "_is_protocol", False) else "class")
    sig = "" if kind == "protocol" else _signature(cls)
    lines += [f"*{kind}*" + (f" `{qual}{sig}`" if sig else ""), ""]
    if _doc(cls):
        lines += [_doc(cls), ""]
    if dataclasses.is_dataclass(cls):
        lines += ["| Field | Type | Default |", "|---|---|---|"]
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                default = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = f"{f.default_factory.__name__}()"      # type: ignore[union-attr]
            else:
                default = "required"
            lines.append(f"| `{f.name}` | `{f.type}` | `{default}` |")
        lines.append("")
    for name, member in _members(cls):
        if isinstance(member, property):
            lines += [f"#### `{qual}.{name}` *(property)*", ""]
            if _doc(member.fget):
                lines += [_doc(member.fget), ""]
            continue
        lines += [f"#### `{qual}.{name}{_signature(member)}`", ""]
        if _doc(member):
            lines += [_doc(member), ""]
    return lines


def _function_section(qual: str, func: Any) -> list[str]:
    lines = [f"### `{qual}{_signature(func)}`", ""]
    if _doc(func):
        lines += [_doc(func), ""]
    return lines


def _module_section(title: str, module: Any, names: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if _doc(module):
        lines += [_doc(module), ""]
    constants, sections = [], []
    for name in sorted(names, key=lambda n: (not inspect.isclass(getattr(module, n)), n.lower())):
        value = getattr(module, name)
        qual = f"{module.__name__}.{name}"
        if inspect.isclass(value):
            sections += _class_section(qual, value)
        elif inspect.isfunction(value):
            sections += _function_section(qual, value)
        elif isinstance(value, (str, int, float)):
            constants.append(f"| `{qual}` | `{value!r}` |")
        else:
            sections += [f"### `{qual}`", "", f"An instance of `{type(value).__name__}`. "
                         + (_doc(type(value)).split("\n\n")[0] if _doc(type(value)) else ""), ""]
    if constants:
        lines += ["| Constant | Value |", "|---|---|", *constants, ""]
    return lines + sections


def render() -> str:
    import opendronekit
    import opendronekit.plugins as plugins

    sdk_names = [n for n in opendronekit.__all__ if n not in ("load_plugins", "plugin_status",
                                                             "PLUGIN_API_LEVEL", "PluginError")]
    plugin_names = [n for n in plugins.__all__ if n != "registry"]
    head = [
        "# OpenDroneKit SDK reference",
        "",
        "<!-- Generated by tools/gen_sdk_reference.py from the code's signatures and",
        "     docstrings. Do not edit by hand: change the docstring and regenerate. -->",
        "",
        f"SDK version **{opendronekit.__version__}**, plugin API level "
        f"**{plugins.PLUGIN_API_LEVEL}**.",
        "",
        "Everything on this page is public API under semantic versioning: a breaking change",
        "bumps the SDK's major version (or, for the plugin contracts, the plugin API major",
        "level). Names not listed here, modules starting with an underscore, and the packages",
        "behind the SDK (`app`, `core`, `mission`, `sdk`) are internal and may change in any",
        "release. Start with the [quickstart](README.md).",
        "",
        "Contents: [opendronekit](#opendronekit) · [opendronekit.plugins](#opendronekitplugins)",
        "",
    ]
    body = _module_section("opendronekit", opendronekit, sdk_names)
    body += _module_section("opendronekit.plugins", plugins, plugin_names)
    text = "\n".join(head + body).rstrip() + "\n"
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUTPUT.relative_to(REPO)} is out of date: run python tools/gen_sdk_reference.py")
            return 1
        print(f"{OUTPUT.relative_to(REPO)} is up to date.")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
