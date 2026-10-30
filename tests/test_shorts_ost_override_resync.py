"""Phase-D override resync — an operator ``--per-lf-location`` correction on
a Phase-D-only resume reaches shorts whose burned label was INHERITED from
their parent LF (2026-09-27 Schull: 'To Goleen, County Cork' corrected to
'Goleen, County Cork', but both shorts' metadata still read 'To Goleen'
because Phase D reads each short's label verbatim from the day's
``shorts_ost_audit_<date>.json`` sidecar, which a Phase-D-only resume never
re-derives).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from test_phase_d import phase_d_state as phase_d_state  # noqa: F401 -- fixture re-export


def _mk_sidecar(tmp_path: Path, date: str, shorts: list) -> list:
    """Write a day-first shorts OST sidecar; return parallel short paths."""
    stages = tmp_path / "_stages"
    shorts_dir = stages / f"day_{date}" / "shorts"
    shorts_dir.mkdir(parents=True)
    (stages / f"shorts_ost_audit_{date}.json").write_text(
        json.dumps({"date": date, "shorts": shorts})
    )
    return [str(shorts_dir / f"short_{s['rank']:02d}.mp4") for s in shorts]


def _run(tmp_path, short_entry, per_lf_overrides):
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
        _build_social_posts_per_lf,
    )
    from gomoveshift.pipeline.social_generator import SocialGenerator

    tmp_path.mkdir(parents=True, exist_ok=True)
    short_paths = _mk_sidecar(tmp_path, "2026-09-27", [short_entry])
    out = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        shorts_per_lf={1: short_paths},
        # The parent-LF label (already corrected via preserve_per_lf_overrides_on_resume,
        # #4040) differs on purpose from the stale sidecar text below.
        per_lf_locations_override={1: "Goleen, County Cork"},
        project_base="proj",
        output_base=tmp_path / "out",
        per_lf_overrides=per_lf_overrides,
    )
    return out[1][0]


_LF_LABEL_ENTRY = {
    "rank": 1,
    "parent_lf_index": 1,
    "ost_text": "To Goleen, County Cork",
    "start_sec": 0.0,
    "end_sec": 30.0,
    "regions": [],
    "place_source": "lf_label",
}


def test_lf_label_short_follows_override_on_resync(tmp_path):
    post = _run(tmp_path, dict(_LF_LABEL_ENTRY), {1: "Goleen, County Cork"})
    # sampled_regions carries the raw resolved location verbatim (F32) — the
    # cleanest surface to assert the override text landed.
    assert post.sampled_regions == ["Goleen, County Cork"]
    assert not post.youtube_title.startswith("To Goleen")
    assert not post.tiktok_title.startswith("To Goleen")
    for tag in post.youtube_tags.split(","):
        assert not tag.strip().startswith("To Goleen")


def test_no_override_leaves_sidecar_text_unchanged(tmp_path):
    """Normal run (no resume, no operator correction) — nothing changes."""
    post_none = _run(tmp_path, dict(_LF_LABEL_ENTRY), None)
    assert post_none.sampled_regions == ["To Goleen, County Cork"]

    post_empty = _run(tmp_path / "second", dict(_LF_LABEL_ENTRY), {})
    assert post_empty.sampled_regions == ["To Goleen, County Cork"]


def test_non_lf_label_source_ignores_override(tmp_path):
    """A short whose burned text came from its OWN clip GPS is never
    rewritten — it's independently verified against that clip's footage,
    not the parent label."""
    entry = dict(_LF_LABEL_ENTRY)
    entry["place_source"] = "clip_gps"
    post = _run(tmp_path, entry, {1: "Goleen, County Cork"})
    assert post.sampled_regions == ["To Goleen, County Cork"]


def test_string_key_override_behaves_like_int_key(tmp_path):
    """``state.per_lf_overrides`` round-trips through JSON with string keys
    in some callers — a string key must resolve the same LF index as int."""
    post = _run(tmp_path, dict(_LF_LABEL_ENTRY), {"1": "Goleen, County Cork"})
    assert post.sampled_regions == ["Goleen, County Cork"]


def test_phase_d_passes_state_overrides_to_builder(phase_d_state, monkeypatch):  # noqa: F811
    """Phase D hands ``state.per_lf_overrides`` to the post builder."""
    import yaml
    from unittest.mock import MagicMock

    import test_global_short_index_wiring as w

    w._install_fake_youtube_uploader(monkeypatch)
    phase_d_state.short_global_index = {"a.mp4": 1, "b.mp4": 2}
    phase_d_state.per_lf_overrides = {1: "Goleen, County Cork"}
    titles_path = phase_d_state["output_base"] / "social" / "titles.yaml"
    titles_path.parent.mkdir(parents=True, exist_ok=True)
    titles_path.write_text(yaml.safe_dump({1: {"title": "Title one"}, 2: {"title": "Title two"}}))

    captured = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return {
            lf.index: [
                MagicMock(short_index=i + 1, caption=f"c{i}")
                for i in range(len(kwargs["shorts_per_lf"].get(lf.index, [])))
            ]
            for lf in kwargs["drive_plan"].regular_lfs
        }

    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.phase_d._build_social_posts_per_lf", _fake_builder
    )
    from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

    patches = w._phase_d_patches()
    with (
        patches[0] as m0,
        patches[1] as m1,
        patches[2] as m2,
        patches[3] as m3,
        patches[4] as m4,
        patches[5] as m5,
        patches[6] as m6,
        patches[7] as m7,
    ):
        w._setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
        PhaseD(w._config()).run(phase_d_state)

    assert captured.get("per_lf_overrides") == {1: "Goleen, County Cork"}
