"""Activate the stable global short indexing mechanism at its two live
call sites.

``assign_global_short_indexes`` and ``derive_interactive_indexes`` exist and
are unit-tested (``tests/test_global_short_index.py``), but nothing calls
them: Phase BC never assigns ``state.short_global_index`` and Phase D never
passes ``short_global_index=`` into ``_build_social_posts_per_lf``. These
tests lock the wiring at both call sites.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from test_phase_bc_shorts_per_lf import _build_state_with_drive_plan, _run_phase  # noqa: E402
from test_phase_d import (  # noqa: E402
    _install_fake_youtube_uploader,
    _phase_d_patches,
    _setup_mocks,
)
from test_phase_d import phase_d_state as phase_d_state  # noqa: E402,F401 -- fixture re-export


# ── Phase BC: state.short_global_index populated after the plan step ───────


class TestPhaseBCAssignsGlobalShortIndex:
    def test_short_global_index_populated_in_plan_order(self, tmp_path):
        state, _, _ = _build_state_with_drive_plan(
            tmp_path,
            days_to_lf_indices={
                "2026-02-20": [1],
                "2026-02-21": [2],
            },
            shorts_per_day=2,
            pre_record_days=True,
        )

        _run_phase(state)

        shorts_per_lf = state["shorts_per_lf"]
        index_map = state.get("short_global_index")
        assert isinstance(index_map, dict), (
            "phase_bc must assign state.short_global_index after state.shorts_per_lf is set"
        )
        # Every planned short is covered, values are the ascending plan-order
        # sequence (LF 1 before LF 2, list order within each LF).
        expected_paths = [str(p) for p in shorts_per_lf[1]] + [str(p) for p in shorts_per_lf[2]]
        assert set(index_map.keys()) == set(expected_paths)
        assert [index_map[p] for p in expected_paths] == list(range(1, len(expected_paths) + 1))

    def test_resume_preserves_existing_indexes(self, tmp_path):
        state, _, _ = _build_state_with_drive_plan(
            tmp_path,
            days_to_lf_indices={"2026-02-20": [1]},
            shorts_per_day=2,
            pre_record_days=True,
        )
        # Simulate a prior run's persisted map for one of the two shorts.
        shorts_per_lf_preview = state.get("shorts_per_lf") or {}
        state.short_global_index = {"already-assigned.mp4": 1}

        _run_phase(state)

        index_map = state["short_global_index"]
        assert index_map["already-assigned.mp4"] == 1, (
            "a resume must not reassign a previously-persisted index"
        )


# ── Phase D: _build_social_posts_per_lf receives short_global_index= ───────


class TestPhaseDPassesGlobalShortIndex:
    def test_calls_builder_with_persisted_short_global_index(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
        phase_d_state.short_global_index = {"a.mp4": 1, "b.mp4": 2}
        # Phase D's hand-title coverage gate now runs upstream of the
        # (mocked) social-post build this test is exercising — give it a
        # titles.yaml covering both required indices so this wiring test
        # isn't tripped by an unrelated gate.
        titles_path = phase_d_state["output_base"] / "social" / "titles.yaml"
        titles_path.parent.mkdir(parents=True, exist_ok=True)
        titles_path.write_text(
            yaml.safe_dump({1: {"title": "Title one"}, 2: {"title": "Title two"}})
        )

        captured = {}

        def _fake_builder(**kwargs):
            captured.update(kwargs)
            out = {}
            for lf in kwargs["drive_plan"].regular_lfs:
                shorts = kwargs["shorts_per_lf"].get(lf.index, [])
                out[lf.index] = [
                    MagicMock(short_index=i + 1, caption=f"c{i}") for i in range(len(shorts))
                ]
            return out

        monkeypatch.setattr(
            "gomoveshift.pipeline.drive_phases.phase_d._build_social_posts_per_lf",
            _fake_builder,
        )

        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        with (
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
            PhaseD(_config()).run(phase_d_state)

        assert captured.get("short_global_index") == {"a.mp4": 1, "b.mp4": 2}

    def test_legacy_state_with_no_map_still_runs(self, phase_d_state, monkeypatch):
        """state.short_global_index is None (older checkpoint) — must not crash."""
        _install_fake_youtube_uploader(monkeypatch)
        phase_d_state.short_global_index = None

        captured = {}
        real_builder = _import_real_builder()

        def _spy_builder(**kwargs):
            captured.update(kwargs)
            return real_builder(**kwargs)

        monkeypatch.setattr(
            "gomoveshift.pipeline.drive_phases.phase_d._build_social_posts_per_lf",
            _spy_builder,
        )

        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        with (
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
            PhaseD(_config()).run(phase_d_state)

        assert captured.get("short_global_index") is None


# ── Phase D: state.interactive_indexes derived from titles.yaml ────────────


class TestPhaseDDerivesInteractiveIndexes:
    def test_derived_when_titles_yaml_present(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
        titles_path = phase_d_state["output_base"] / "social" / "titles.yaml"
        titles_path.parent.mkdir(parents=True, exist_ok=True)
        titles_path.write_text(
            yaml.safe_dump(
                {
                    1: {"title": "Guess this coastal road", "interactive": True},
                    2: {"title": "Ordinary short"},
                }
            )
        )
        phase_d_state.interactive_indexes = None

        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        with (
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
            PhaseD(_config()).run(phase_d_state)

        assert phase_d_state["interactive_indexes"] == [1]

    def test_stays_none_when_titles_yaml_absent(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
        phase_d_state.interactive_indexes = None

        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        with (
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
            PhaseD(_config()).run(phase_d_state)

        assert phase_d_state["interactive_indexes"] is None


def _config():
    from gomoveshift.config import get_config

    return get_config()


def _import_real_builder():
    from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf

    return _build_social_posts_per_lf
