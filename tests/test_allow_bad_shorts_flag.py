"""``--allow-bad-shorts`` is a real, reachable CLI flag.

Phase D's shorts quality gate hard-fails a run whose shorts are not
1080x1920 and tells the operator, in the failure message itself, to "pass
--allow-bad-shorts to override". That flag did not exist: the state field
``skip_shorts_quality_gate`` was declared (``drive_state_model``) and read
(``phase_d_gates``), but nothing wired it to argparse or the flag registry —
so the only escape from a documented escape hatch was hand-editing
``pipeline_state.json``.

These tests drive the REAL parser and the REAL apply/merge layers rather
than setting ``state.skip_shorts_quality_gate`` directly (as the two
pre-existing gate tests do), because the wiring — not the gate — is what
was missing. The gate's own logic is unchanged and covered elsewhere.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d_gates import _assert_shorts_quality_gate

from test_flag_registry import _fresh_state, _make_orch, _resume

_FIELD = "skip_shorts_quality_gate"


def _parse_drive(*argv: str) -> argparse.Namespace:
    """Parse a real ``gomoveshift drive`` command line."""
    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    return parser.parse_args(["drive", "/tmp/src", *argv])


def _empty_shorts(state):
    """The shape the gate hard-fails on: regular LFs but no shorts."""
    state.regular_lf_paths = ["/x/lf_01.mp4"]
    state.shorts_per_lf = {}
    return state


class TestAllowBadShortsReachesState:
    def test_flag_parses_to_the_state_field_dest(self):
        """1. ``--allow-bad-shorts`` parses and lands on the state dest."""
        args = _parse_drive("--allow-bad-shorts")
        assert getattr(args, _FIELD) is True

    def test_flag_absent_parses_to_the_none_sentinel(self):
        """Argparse-sentinel: absent flag is None, not False — a concrete
        False would re-stomp a saved opt-in on every resume."""
        args = _parse_drive()
        assert getattr(args, _FIELD) is None

    def test_flag_reaches_the_built_state(self, tmp_path):
        """1 (cont.). The parsed value survives ``_build_initial_state``."""
        args = _parse_drive("--allow-bad-shorts")
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, **{_FIELD: getattr(args, _FIELD)})
        assert state.get(_FIELD) is True


class TestAllowBadShortsResumeSemantics:
    def test_saved_optin_survives_a_resume_without_the_flag(self, tmp_path):
        """2. The sentinel behaviour: a saved True is preserved when the
        operator does not re-pass the flag on a resume."""
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)  # flag NOT passed this invocation
        _resume(orch, state, tmp_path, {_FIELD: True})
        assert state.get(_FIELD) is True

    def test_fresh_run_without_the_flag_leaves_the_gate_active(self, tmp_path):
        """3. Omitting the flag must not silently disable the gate."""
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)
        assert not state.get(_FIELD)
        with pytest.raises(RuntimeError, match="shorts_per_lf"):
            _assert_shorts_quality_gate(_empty_shorts(state), validator_cls=MagicMock())


class TestAllowBadShortsReachesTheGate:
    def test_flag_makes_the_gate_return_early(self, tmp_path):
        """4. End to end: the CLI flag actually reaches the gate's escape,
        so the operator instruction in the failure message is truthful."""
        args = _parse_drive("--allow-bad-shorts")
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, **{_FIELD: getattr(args, _FIELD)})
        # No raise: the escape at phase_d_gates._assert_shorts_quality_gate.
        _assert_shorts_quality_gate(_empty_shorts(state), validator_cls=MagicMock())
