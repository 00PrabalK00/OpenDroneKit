"""The file chooser's filters must be what the toolkit accepts, not what the UI has.

pywebview validates `file_types` against "Description (*.ext;*.ext)" and raises
ValueError on anything else. The UI passed the bare list it naturally holds --
["tif", "tiff", "geojson", ...] -- so pressing New Folder produced

    New Folder: ValueError: tif is not a valid file filter

as the toolbar result. Every layer-import control was dead in that way, and the audit
recorded it as an `api` outcome because something WAS reported back: an exception is a
response, just not a useful one.

These tests pin the conversion. The pywebview validator is a regex over the same shape,
so matching it here is the check that matters.
"""

from __future__ import annotations

import re

import pytest

from app.api import _file_type_filters

# Mirrors pywebview's own grammar for a file filter: a description, then a
# parenthesised semicolon-separated list of *.ext patterns. The wildcard *.* is legal
# there and is what "All files" uses, so it has to be legal here -- an earlier version of
# this pattern rejected it and failed three tests over the checker rather than the code.
#
# Copied rather than imported: pywebview is a desktop-only dependency and is not
# installed in CI, so importing it would skip exactly the tests that matter.
VALID = re.compile(r"^[^(]+\((\*\.(?:\w+|\*))(;\*\.(?:\w+|\*))*\)$")


def test_the_extensions_that_broke_it_now_produce_valid_filters():
    """The exact call the New Folder button makes."""
    filters = _file_type_filters(["tif", "tiff", "geojson", "json", "shp"])
    assert filters
    for entry in filters:
        assert VALID.match(entry), f"pywebview would reject {entry!r}"


def test_a_bare_extension_is_never_passed_through():
    """The bug itself: "tif" is not a filter, it is an extension."""
    for entry in _file_type_filters(["tif"]):
        assert entry != "tif"
        assert VALID.match(entry)


@pytest.mark.parametrize("extensions", [None, []])
def test_no_extensions_means_everything(extensions):
    assert _file_type_filters(extensions) == ("All files (*.*)",)


def test_every_requested_extension_survives():
    """Silently dropping a type would stop a user opening a file they are entitled to.

    Quieter than the ValueError and worse, because nothing reports it.
    """
    wanted = ["tif", "geojson", "las", "xyz"]
    joined = " ".join(_file_type_filters(wanted))
    for extension in wanted:
        assert f"*.{extension}" in joined, f"{extension} was dropped"


def test_an_unknown_extension_gets_its_own_entry():
    filters = _file_type_filters(["xyz"])
    assert any("*.xyz" in f for f in filters)
    for entry in filters:
        assert VALID.match(entry)


def test_related_extensions_are_grouped_under_one_label():
    """A chooser listing eight suffixes separately is worse than one listing two kinds."""
    filters = _file_type_filters(["tif", "tiff", "geojson", "shp"])
    rasters = [f for f in filters if f.startswith("Rasters")]
    assert rasters == ["Rasters (*.tif;*.tiff)"]
    vectors = [f for f in filters if f.startswith("Vectors")]
    assert vectors == ["Vectors (*.geojson;*.shp)"]


def test_a_combined_entry_comes_first_when_there_are_several_kinds():
    """A mixed folder should not force the user to cycle the dropdown."""
    filters = _file_type_filters(["tif", "geojson"])
    assert filters[0] == "Supported files (*.tif;*.geojson)"


def test_all_files_is_always_offered_last():
    """Refusing to show a file the user can see on disk is not our call to make."""
    assert _file_type_filters(["tif", "geojson"])[-1] == "All files (*.*)"


def test_extensions_are_accepted_however_they_are_written():
    for form in (["*.tif"], [".tif"], ["TIF"], ["  tif "]):
        filters = _file_type_filters(form)
        assert any("*.tif" in f for f in filters), form
