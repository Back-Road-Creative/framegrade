"""A published short may name ONLY the place its own clip's GPS verified.

Phase D refuses, before chapters and scheduling, any planned short whose
sidecar entry lacks a label, has no clip-GPS regions, or whose label was not
its own clip-GPS vote (``place_source != "clip_gps"``). On a publish run no
short is ever handed the LF label as a stand-in.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
    _build_social_posts_per_lf,
    assert_all_places_verified,
)

from test_phase_d import (
    _install_fake_youtube_uploader,
    _phase_d_patches,
    _setup_mocks,
    get_config,
)
from test_phase_d import phase_d_state as phase_d_state  # noqa: F401 -- fixture re-export

DATE = "2026-09-14"
GOOD = {
    "ost_text": "Wythe County, Virginia",
    "regions": ["Wythe County, Virginia"],
    "start_sec": 10.0,
    "end_sec": 40.0,
}


def _short(stages: Path, rank: int) -> Path:
    return stages / f"day_{DATE}" / "shorts" / f"short_{rank:02d}.mp4"


def _sidecar(stages: Path, entries: dict[int, dict]) -> None:
    """Write the day's shorts_ost_audit sidecar; ``entries`` is rank -> fields."""
    stages.mkdir(parents=True, exist_ok=True)
    (stages / f"shorts_ost_audit_{DATE}.json").write_text(
        json.dumps({"shorts": [{"rank": r, **e} for r, e in entries.items()]})
    )


def _verified(**over) -> dict:
    return {**GOOD, "place_source": "clip_gps", **over}


class TestGateVerdicts:
    def test_all_verified_passes(self, tmp_path):
        _sidecar(tmp_path, {1: _verified(), 2: _verified()})
        shorts = {1: [_short(tmp_path, 1), _short(tmp_path, 2)]}
        assert assert_all_places_verified(shorts) == []

    def test_refuses_short_without_clip_gps_place_source(self, tmp_path):
        _sidecar(tmp_path, {1: _verified(), 2: _verified(place_source="lf_label")})
        shorts = {1: [_short(tmp_path, 1), _short(tmp_path, 2)]}
        with pytest.raises(RuntimeError, match=r"short_02\.mp4.*place_source") as exc:
            assert_all_places_verified(shorts)
        assert "short_01.mp4" not in str(exc.value)

    def test_refuses_a_sidecar_that_predates_place_source(self, tmp_path):
        _sidecar(tmp_path, {1: dict(GOOD)})
        with pytest.raises(RuntimeError, match="place_source"):
            assert_all_places_verified({1: [_short(tmp_path, 1)]})

    def test_refuses_short_with_empty_regions(self, tmp_path):
        _sidecar(tmp_path, {1: _verified(regions=[])})
        with pytest.raises(RuntimeError, match=r"short_01\.mp4.*regions"):
            assert_all_places_verified({1: [_short(tmp_path, 1)]})

    def test_refuses_short_with_no_label_and_names_every_offender(self, tmp_path):
        _sidecar(tmp_path, {1: _verified(ost_text=""), 3: _verified()})
        shorts = {1: [_short(tmp_path, 1), _short(tmp_path, 2)], 2: [_short(tmp_path, 3)]}
        with pytest.raises(RuntimeError) as exc:
            assert_all_places_verified(shorts)
        msg = str(exc.value)
        assert "short_01.mp4" in msg and "short_02.mp4" in msg  # 2 has no entry at all
        assert "short_03.mp4" not in msg

    def test_audit_pass_warns_not_raises(self, tmp_path, caplog):
        _sidecar(tmp_path, {1: _verified(place_source="lf_label")})
        with caplog.at_level(logging.WARNING):
            problems = assert_all_places_verified({1: [_short(tmp_path, 1)]}, enforce=False)
        assert len(problems) == 1
        warned = [r for r in caplog.records if "UNVERIFIED-PLACES" in r.getMessage()]
        assert len(warned) == 1 and warned[0].levelno == logging.WARNING
        assert "short_01.mp4" in warned[0].getMessage()


class _RecordingGen:
    """Stand-in SocialGenerator: records what per-short locations it is handed."""

    calls: list = []

    def __init__(self, **kw):
        pass

    def generate_all(self, **kw):
        type(self).calls.append(kw)
        return []


def _build(tmp_path, shorts, **kw):
    _RecordingGen.calls = []
    kw.setdefault("short_global_index", {str(p): i for i, p in enumerate(shorts, 1)})
    plan = MagicMock(regular_lfs=[MagicMock(index=1)])
    return _build_social_posts_per_lf(
        social_gen_cls=_RecordingGen,
        drive_plan=plan,
        shorts_per_lf={1: shorts},
        per_lf_locations_override={1: "Gettysburg Drive, Pennsylvania"},
        project_base="p",
        output_base=tmp_path / "out",
        **kw,
    )


class TestNoBorrowedLabelOnPublish:
    def test_refuses_gps_less_short_on_publish_run(self, tmp_path):
        """A GPS-less short has no label of its own: it must never be handed the LF label."""
        _sidecar(tmp_path, {1: _verified(), 2: {}})
        shorts = [_short(tmp_path, 1), _short(tmp_path, 2)]
        with pytest.raises(RuntimeError, match=r"short_02\.mp4"):
            _build(tmp_path, shorts, require_verified_places=True)
        assert _RecordingGen.calls == []  # refused before any per-short label was built

    def test_publish_run_uses_only_sidecar_labels(self, tmp_path):
        _sidecar(tmp_path, {1: _verified(), 2: _verified(ost_text="Smyth County, Virginia")})
        shorts = [_short(tmp_path, 1), _short(tmp_path, 2)]
        _build(tmp_path, shorts, require_verified_places=True)
        locs = _RecordingGen.calls[0]["per_short_locations"]
        assert locs == ["Wythe County, Virginia", "Smyth County, Virginia"]

    def test_audit_run_keeps_the_legacy_fallback(self, tmp_path):
        _sidecar(tmp_path, {1: _verified(), 2: {}})
        shorts = [_short(tmp_path, 1), _short(tmp_path, 2)]
        _build(tmp_path, shorts)  # require_verified_places defaults to False
        locs = _RecordingGen.calls[0]["per_short_locations"]
        assert locs[1] == "Gettysburg Drive, Pennsylvania"

    @pytest.mark.parametrize("index", [None, {}, "only_first"])
    def test_publish_run_refuses_short_missing_from_short_global_index(self, tmp_path, index):
        """A short in shorts_per_lf is posted whether or not the index knows it (old
        checkpoint, missing key): it must not slip past the gate into the LF label."""
        _sidecar(tmp_path, {1: _verified(), 2: {}})
        shorts = [_short(tmp_path, 1), _short(tmp_path, 2)]
        if index == "only_first":
            index = {str(shorts[0]): 1}
        with pytest.raises(RuntimeError, match=r"short_02\.mp4"):
            _build(tmp_path, shorts, require_verified_places=True, short_global_index=index)
        assert _RecordingGen.calls == []

    def test_publish_run_ignores_stale_index_entry_for_a_dropped_short(self, tmp_path):
        """The resumed-plan direction: an index entry for a short no longer in
        shorts_per_lf is not posted, so it never trips the gate."""
        _sidecar(tmp_path, {1: _verified(), 2: {}})
        shorts = [_short(tmp_path, 1)]
        stale = {str(shorts[0]): 1, str(_short(tmp_path, 2)): 2}
        _build(tmp_path, shorts, require_verified_places=True, short_global_index=stale)
        assert _RecordingGen.calls[0]["per_short_locations"] == ["Wythe County, Virginia"]


class TestPhaseDWiring:
    @staticmethod
    def _plan(state, tmp_path, *, sidecar: dict[int, dict] | None):
        shorts = {
            1: [_short(tmp_path, r) for r in (1, 2, 3)],
            2: [_short(tmp_path, r) for r in (4, 5, 6)],
        }
        state.shorts_per_lf = shorts
        state.short_global_index = {str(p): i for i, p in enumerate(sum(shorts.values(), []), 1)}
        if sidecar is not None:
            _sidecar(tmp_path, sidecar)
        titles = state["output_base"] / "social" / "titles.yaml"
        titles.parent.mkdir(parents=True, exist_ok=True)
        titles.write_text(yaml.safe_dump({i: {"title": f"T{i}"} for i in range(1, 7)}))

    def test_gate_runs_before_chapters_and_schedule(self, phase_d_state, tmp_path, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
        # Hand titles cover every short, so only the place gate can refuse.
        self._plan(phase_d_state, tmp_path, sidecar={1: _verified()})
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_all_lfs"
            ) as chapters_mock,
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._build_and_persist_drive_schedule"
            ) as schedule_mock,
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[6] as m6,
            _phase_d_patches()[7] as m7,
        ):
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            with pytest.raises(RuntimeError, match="place") as exc_info:
                PhaseD(get_config()).run(phase_d_state)
        for n in range(2, 7):  # shorts 2..6 have no sidecar entry: all named at once
            assert f"short_{n:02d}.mp4" in str(exc_info.value)
        chapters_mock.assert_not_called()
        schedule_mock.assert_not_called()

    def test_audit_pass_warns_and_continues_to_chapters(
        self, phase_d_state, tmp_path, monkeypatch, caplog
    ):
        _install_fake_youtube_uploader(monkeypatch)
        phase_d_state.youtube_upload = False
        phase_d_state.skip_upload = True
        self._plan(phase_d_state, tmp_path, sidecar=None)

        class _StopError(Exception):
            pass

        with (
            _phase_d_patches()[0] as meta_cls,
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_all_lfs",
                side_effect=_StopError,
            ),
            caplog.at_level(logging.WARNING),
        ):
            meta = MagicMock()
            meta.generate.side_effect = lambda **k: {"title": k["project_name"], "tags": []}
            meta_cls.return_value = meta
            with pytest.raises(_StopError):  # reached chapters: the audit pass was not refused
                PhaseD(get_config()).run(phase_d_state)
        assert any("UNVERIFIED-PLACES" in r.getMessage() for r in caplog.records)

    def test_gate_covers_shorts_missing_from_short_global_index(
        self, phase_d_state, tmp_path, monkeypatch
    ):
        """Universe is shorts_per_lf, not the index: a legacy None index cannot skip it."""
        _install_fake_youtube_uploader(monkeypatch)
        self._plan(phase_d_state, tmp_path, sidecar={})
        phase_d_state.short_global_index = None
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_all_lfs"
            ) as chapters_mock,
            _phase_d_patches()[0],
        ):
            with pytest.raises(RuntimeError, match="place") as exc_info:
                PhaseD(get_config()).run(phase_d_state)
        for n in range(1, 7):
            assert f"short_{n:02d}.mp4" in str(exc_info.value)
        chapters_mock.assert_not_called()
