"""A stable global short index assigned at plan time and carried through
state, so Phase D stops recomputing ``global_index_offset`` from
``len(posts)``.

See docs/pipeline/135-shorts-titles-and-interactive-family.md.
"""

from types import SimpleNamespace

import pytest
import yaml

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _build_social_posts_per_lf
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.social_generator import (
    SocialGenerator,
    assign_global_short_indexes,
    derive_interactive_indexes,
    load_hand_titles,
)


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


# ── step 1: plan-time assignment ────────────────────────────────────────────


def test_contiguous_across_lfs_in_plan_order():
    shorts_per_lf = {1: ["a.mp4", "b.mp4"], 2: ["c.mp4"]}
    assigned = assign_global_short_indexes(shorts_per_lf)
    assert assigned == {"a.mp4": 1, "b.mp4": 2, "c.mp4": 3}


def test_plan_order_is_ascending_lf_index_not_dict_insertion_order():
    shorts_per_lf = {2: ["c.mp4"], 1: ["a.mp4", "b.mp4"]}
    assigned = assign_global_short_indexes(shorts_per_lf)
    assert assigned == {"a.mp4": 1, "b.mp4": 2, "c.mp4": 3}


def test_culled_short_leaves_a_hole_never_reused():
    # First plan-time pass: 3 shorts assigned 1, 2, 3.
    first = assign_global_short_indexes({1: ["a.mp4", "b.mp4"], 2: ["c.mp4"]})
    assert first == {"a.mp4": 1, "b.mp4": 2, "c.mp4": 3}

    # b.mp4 failed/was culled and no longer appears; a new short d.mp4 shows
    # up for LF 2. The existing indexes for a/c must not move, and d.mp4
    # must NOT reuse the freed index 2 — it gets the next fresh one.
    second = assign_global_short_indexes({1: ["a.mp4"], 2: ["c.mp4", "d.mp4"]}, existing=first)
    assert second["a.mp4"] == 1
    assert second["c.mp4"] == 3
    assert second["d.mp4"] == 4
    # index 2 (b.mp4's) is never handed to a different short — the hole.
    assert [k for k, v in second.items() if v == 2] == ["b.mp4"]


def test_assignment_never_mutates_existing_in_place():
    existing = {"a.mp4": 1}
    result = assign_global_short_indexes({1: ["a.mp4", "b.mp4"]}, existing=existing)
    assert existing == {"a.mp4": 1}
    assert result == {"a.mp4": 1, "b.mp4": 2}


# ── step 3: interactive_indexes derived once from load_hand_titles ─────────


def test_derive_interactive_indexes(tmp_path):
    path = tmp_path / "titles.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                1: {"title": "Guess this coastal road", "interactive": True},
                2: {"title": "Ordinary short"},
                3: {"title": "Another guess", "interactive": True},
            }
        )
    )
    hand_titles = load_hand_titles(path)
    assert derive_interactive_indexes(hand_titles) == {1, 3}


def test_derive_interactive_indexes_empty_file(tmp_path):
    assert derive_interactive_indexes(load_hand_titles(tmp_path / "nope.yaml")) == set()


# ── step 2: Phase D consumes the persisted index ────────────────────────────


def _lf(idx):
    return SimpleNamespace(index=idx)


def test_phase_d_uses_persisted_global_index_over_recompute(tmp_path):
    """A hole preceding a short must not shift its persisted index — the
    consumer must use the persisted map, not len(posts)."""
    short_global_index = {"lf1_a.mp4": 1, "lf2_c.mp4": 4}  # a hole at 2-3
    out = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[_lf(1), _lf(2)]),
        shorts_per_lf={1: ["lf1_a.mp4"], 2: ["lf2_c.mp4"]},
        per_lf_locations_override={1: "County Kerry, Ireland", 2: "County Cork, Ireland"},
        project_base="proj",
        output_base=tmp_path,
        short_global_index=short_global_index,
    )
    assert out[1][0].short_index == 1
    assert out[2][0].short_index == 4


def test_phase_d_titles_yaml_lands_on_persisted_index_across_a_hole(tmp_path):
    """A hole before a short must not shift which titles.yaml entry lands on
    it — the whole point of a stable, persisted global index."""
    titles_path = tmp_path / "social" / "titles.yaml"
    titles_path.parent.mkdir(parents=True)
    titles_path.write_text(yaml.safe_dump({4: {"title": "Hand title for the 4th short"}}))

    short_global_index = {"lf1_a.mp4": 1, "lf2_c.mp4": 4}  # 2, 3 are holes
    out = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[_lf(1), _lf(2)]),
        shorts_per_lf={1: ["lf1_a.mp4"], 2: ["lf2_c.mp4"]},
        per_lf_locations_override={1: "County Kerry, Ireland", 2: "County Cork, Ireland"},
        project_base="proj",
        output_base=tmp_path / "out",
        titles_file=titles_path,
        short_global_index=short_global_index,
    )
    assert out[2][0].short_index == 4
    assert out[2][0].youtube_title == "Hand title for the 4th short"


def test_phase_d_falls_back_when_lf_short_uncovered_by_map(caplog, tmp_path):
    """A short whose path isn't in short_global_index (legacy state / a
    genuinely uncovered entry) falls back to the legacy recompute for that
    LF, with a warning naming the fallback — never a crash."""
    with caplog.at_level("WARNING"):
        out = _build_social_posts_per_lf(
            social_gen_cls=SocialGenerator,
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            shorts_per_lf={1: ["unmapped.mp4"]},
            per_lf_locations_override={1: "County Kerry, Ireland"},
            project_base="proj",
            output_base=tmp_path,
            short_global_index={},  # no coverage at all
        )
    assert out[1][0].short_index == 1  # legacy index_offset=0 recompute
    assert any("fall" in r.message.lower() for r in caplog.records)


def test_phase_d_no_map_supplied_is_legacy_recompute_unchanged(tmp_path):
    """No short_global_index at all (older callers) behaves exactly as
    before — pure backward compatibility."""
    out = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[_lf(1), _lf(2)]),
        shorts_per_lf={1: ["a.mp4", "b.mp4"], 2: ["c.mp4"]},
        per_lf_locations_override={1: "County Kerry, Ireland", 2: "County Cork, Ireland"},
        project_base="proj",
        output_base=tmp_path,
    )
    assert [p.short_index for p in out[1]] == [1, 2]
    assert [p.short_index for p in out[2]] == [3]


# ── generate_all(global_indexes=...) / index_offset unchanged ──────────────


def _generator(**kwargs):
    return SocialGenerator(
        location="County Kerry, Ireland",
        project_name="test_project",
        series_part=1,
        **kwargs,
    )


def test_generate_all_index_offset_unchanged_for_existing_callers():
    gen = _generator()
    posts = gen.generate_all(num_shorts=2, index_offset=5)
    assert [p.short_index for p in posts] == [6, 7]


def test_generate_all_global_indexes_wins_over_index_offset():
    gen = _generator()
    posts = gen.generate_all(num_shorts=2, index_offset=5, global_indexes=[10, 20])
    assert [p.short_index for p in posts] == [10, 20]


def test_generate_all_global_indexes_wrong_length_raises():
    gen = _generator()
    with pytest.raises(ValueError, match="global_indexes"):
        gen.generate_all(num_shorts=3, global_indexes=[1, 2])


# ── legacy state without the field: pipeline_state.json round-trip ─────────


def test_legacy_state_without_field_loads_cleanly():
    legacy = {"region": "ireland", "publish_day": "wed"}
    state = PipelineState.from_dict(legacy)
    assert state.short_global_index is None
    assert state.interactive_indexes is None


def test_state_round_trips_new_fields():
    state = PipelineState(short_global_index={"a.mp4": 1}, interactive_indexes=[3, 7])
    payload = state.to_dict()
    restored = PipelineState.from_dict(payload)
    assert restored.short_global_index == {"a.mp4": 1}
    assert restored.interactive_indexes == [3, 7]
