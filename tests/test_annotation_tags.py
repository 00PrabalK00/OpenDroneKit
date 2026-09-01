"""Tags, and the bulk operations that make them worth having.

Annotations carried a label, a severity and a status. Those say WHAT a finding is and how
bad it is; neither is how an inspector slices four hundred findings afterwards. "North
elevation", "reflight", "client query" are tags, one finding routinely needs several, and
none of them fits in a single-valued label.

The bulk case is the normal case. An inspector reviews forty roof photographs and wants
them all marked "north elevation"; doing that one at a time is why people stop tagging,
and an untagged set cannot be filtered, reported on or handed over.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.annotations import (
    add_tags,
    all_tags,
    create_annotation,
    list_annotations,
    normalise_tag,
    remove_tags,
)

GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}


@pytest.fixture
def project(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def findings(project) -> list[str]:
    return [
        create_annotation(
            project, project_id="p1", source_type="image", source_id=f"img{i}.jpg",
            annotation_type="rectangle", geometry=GEOMETRY, label="crack",
            severity="medium", status="open",
        ).id
        for i in range(4)
    ]


class TestOneSpellingPerTag:
    @pytest.mark.parametrize("written", [
        "North Elevation", "north elevation", "  NORTH   ELEVATION  ", "North  Elevation",
    ])
    def test_the_same_tag_written_differently_is_one_tag(self, written) -> None:
        """Three spellings is one tag to an inspector and three to a filter, which is how
        a tag list becomes useless by the second survey."""
        assert normalise_tag(written) == "north elevation"

    def test_filtering_ignores_how_it_was_typed(self, project, findings) -> None:
        add_tags(project, findings[:2], ["North Elevation"])
        assert len(list_annotations(project, tag="NORTH   elevation")) == 2


class TestTaggingManyAtOnce:
    def test_it_tags_every_one_it_was_given(self, project, findings) -> None:
        result = add_tags(project, findings[:3], ["north elevation", "reflight"])
        assert len(result["updated"]) == 3
        assert len(list_annotations(project, tag="reflight")) == 3

    def test_a_stale_id_does_not_discard_the_rest(self, project, findings) -> None:
        """A selection built before something was deleted should still tag the other
        thirty-nine, and say which one it could not find."""
        result = add_tags(project, findings[:3] + ["gone"], ["north elevation"])
        assert len(result["updated"]) == 3
        assert result["missing"] == ["gone"]

    def test_reapplying_a_tag_changes_nothing(self, project, findings) -> None:
        add_tags(project, findings[:3], ["north elevation"])
        again = add_tags(project, findings[:3], ["north elevation"])
        assert again["updated"] == []
        assert len(list_annotations(project, tag="north elevation")) == 3

    def test_a_finding_carries_several_tags(self, project, findings) -> None:
        add_tags(project, findings[:1], ["north elevation"])
        add_tags(project, findings[:1], ["reflight", "client query"])
        tags = list_annotations(project)[0].tags if list_annotations(project)[0].id == findings[0] \
            else next(a for a in list_annotations(project) if a.id == findings[0]).tags
        assert set(tags) == {"north elevation", "reflight", "client query"}

    def test_tagging_nothing_is_refused(self, project, findings) -> None:
        with pytest.raises(ValueError):
            add_tags(project, findings, [])
        with pytest.raises(ValueError):
            add_tags(project, findings, ["   "])

    def test_untagged_findings_are_untouched(self, project, findings) -> None:
        add_tags(project, findings[:2], ["north elevation"])
        untouched = next(a for a in list_annotations(project) if a.id == findings[3])
        assert untouched.tags == []


class TestRemovingTags:
    def test_it_removes_from_many(self, project, findings) -> None:
        add_tags(project, findings, ["reflight", "north elevation"])
        remove_tags(project, findings, ["reflight"])
        assert list_annotations(project, tag="reflight") == []
        assert len(list_annotations(project, tag="north elevation")) == 4

    def test_removing_a_tag_that_was_not_there_is_not_an_error(self, project, findings) -> None:
        result = remove_tags(project, findings, ["never applied"])
        assert result["updated"] == []


class TestTheTagList:
    def test_it_counts_use(self, project, findings) -> None:
        """The count is what makes the list usable: a tag on one finding out of four
        hundred is usually a typo, and it appears here beside the one it should have
        been."""
        add_tags(project, findings[:3], ["north elevation"])
        add_tags(project, findings[:1], ["reflight"])
        assert all_tags(project) == [
            {"tag": "north elevation", "count": 3},
            {"tag": "reflight", "count": 1},
        ]

    def test_it_is_empty_before_anything_is_tagged(self, project, findings) -> None:
        assert all_tags(project) == []


class TestOlderRecordsStillLoad:
    def test_an_annotation_written_before_tags_existed_reads_as_untagged(
        self, project, findings
    ) -> None:
        """from_dict filters to known fields, so a store written before this feature has
        no tags key. It must load as an untagged finding rather than failing -- a project
        that cannot be opened because of a new column is the worst kind of migration.
        """
        import json

        store = project / "analysis" / "annotations" / "annotations.json"
        records = json.loads(store.read_text(encoding="utf-8"))
        for record in records:
            record.pop("tags", None)
        store.write_text(json.dumps(records), encoding="utf-8")

        loaded = list_annotations(project)
        assert len(loaded) == len(findings)
        assert all(a.tags == [] for a in loaded)

    def test_tagging_still_works_on_such_a_record(self, project, findings) -> None:
        import json

        store = project / "analysis" / "annotations" / "annotations.json"
        records = json.loads(store.read_text(encoding="utf-8"))
        for record in records:
            record.pop("tags", None)
        store.write_text(json.dumps(records), encoding="utf-8")

        add_tags(project, findings[:2], ["north elevation"])
        assert len(list_annotations(project, tag="north elevation")) == 2
