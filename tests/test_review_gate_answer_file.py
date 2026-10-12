"""Gate C/D non-interactive answer-file path.

Before prompting, ``gate_cd`` checks ``<output_dir>/_gates/<gate-id>.answer``.
If present, it is consumed (renamed to ``.answer.used``) and its decision is
used instead of prompting. Otherwise ``<gate-id>.question.json`` is written
describing the gate, and the existing interactive prompt path runs as before
(unblocking a later browser console that polls the question file and writes
the answer).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from gomoveshift.pipeline.review_gate import ReviewFailedError, ReviewGate

from _state_helpers import make_state

import pytest


def _stub_ffprobe(monkeypatch):
    monkeypatch.setattr(ReviewGate, "_stream_count", lambda self, p: 2)
    monkeypatch.setattr(ReviewGate, "_av_drift_pct", lambda self, p: 0.5)
    monkeypatch.setattr(ReviewGate, "_probe_av_durations", lambda self, p: (600.0, 600.0))


def _write_lf_set(output_dir: Path, lf_count: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)
    lf_paths = []
    for idx in range(1, lf_count + 1):
        (output_dir / f"lf_{idx:02d}_youtube.json").write_text(
            json.dumps({"title": f"LF {idx} Title", "description": "x"}, sort_keys=True)
        )
        (output_dir / f"lf_{idx:02d}.mp4").write_bytes(b"")
        lf_paths.append(str(output_dir / f"lf_{idx:02d}.mp4"))
    return make_state(regular_lf_paths=lf_paths)


class TestGateAnswerFile:
    def test_answer_file_approve_skips_prompt_and_is_marked_used(self, tmp_path, monkeypatch):
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        gates_dir = tmp_path / "_gates"
        gates_dir.mkdir()
        answer_path = gates_dir / "gate_cd_2026-08-25.answer"
        answer_path.write_text(
            json.dumps(
                {"decision": "approve", "note": "looks good", "answered_at": "2026-08-25T00:00:00Z"}
            )
        )

        with patch("builtins.input") as mi:
            ReviewGate(interactive=True).gate_cd(
                state=state, output_dir=tmp_path, date_key="2026-08-25"
            )

        assert not mi.called
        assert state.gate_cd_confirmed is True
        assert not answer_path.exists()
        assert (gates_dir / "gate_cd_2026-08-25.answer.used").exists()

    def test_answer_file_reject_raises_without_prompt(self, tmp_path, monkeypatch):
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        gates_dir = tmp_path / "_gates"
        gates_dir.mkdir()
        answer_path = gates_dir / "gate_cd_d.answer"
        answer_path.write_text(json.dumps({"decision": "reject", "note": "bad OST"}))

        with patch("builtins.input") as mi:
            with pytest.raises(ReviewFailedError):
                ReviewGate(interactive=True).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert not mi.called
        assert (gates_dir / "gate_cd_d.answer.used").exists()

    def test_no_answer_file_writes_question_and_falls_back_to_prompt(self, tmp_path, monkeypatch):
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch("builtins.input", return_value="y") as mi:
            ReviewGate(interactive=True).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert mi.called
        question_path = tmp_path / "_gates" / "gate_cd_d.question.json"
        assert question_path.exists()
        payload = json.loads(question_path.read_text())
        assert payload["gate_id"] == "gate_cd_d"
        assert "question" in payload
        assert payload["choices"] == ["approve", "reject"]
        assert "artifacts" in payload
        assert state.gate_cd_confirmed is True

    def test_non_interactive_no_answer_writes_question_and_skips(self, tmp_path, monkeypatch):
        """A console (--no-review) run must still write question.json so a
        later console can discover and answer the gate -- the old code
        returned at ``if not self.interactive`` before ever reaching the
        answer-file mechanism, so headless runs never wrote it.
        """
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)

        with patch("builtins.input") as mi:
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert not mi.called
        question_path = tmp_path / "_gates" / "gate_cd_d.question.json"
        assert question_path.exists()
        payload = json.loads(question_path.read_text())
        assert payload["gate_id"] == "gate_cd_d"
        # The waiver is NOT persisted -- no answer was ever supplied.
        assert state.gate_cd_confirmed is not True

    def test_non_interactive_default_timeout_never_sleeps(self, tmp_path, monkeypatch):
        """The answer poll is opt-in (``GMS_GATE_ANSWER_TIMEOUT_SEC`` unset or
        0, the default): a headless cron/CLI --no-review run with nobody to
        answer it must return immediately, with zero calls to ``time.sleep``
        -- never block up to the 6h a console operator's opt-in gets."""
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.delenv("GMS_GATE_ANSWER_TIMEOUT_SEC", raising=False)

        def fail_if_called(_seconds):
            raise AssertionError("must not sleep when the answer poll is off by default")

        monkeypatch.setattr("gomoveshift.pipeline.review_gate.time.sleep", fail_if_called)

        with patch("builtins.input") as mi:
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert not mi.called
        assert state.gate_cd_confirmed is not True

    def test_non_interactive_polls_and_consumes_a_late_answer(self, tmp_path, monkeypatch):
        """A console-answered gate is no longer stranded until a re-run --
        with the poll opted in (``GMS_GATE_ANSWER_TIMEOUT_SEC`` > 0, as the
        console container sets it) the non-interactive path polls for the
        answer file it just prompted for, instead of returning ``None`` on
        the very same call."""
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.setenv("GMS_GATE_ANSWER_TIMEOUT_SEC", "30")
        gates_dir = tmp_path / "_gates"
        answer_path = gates_dir / "gate_cd_d.answer"
        calls = {"n": 0}

        def fake_sleep(_seconds):
            calls["n"] += 1
            gates_dir.mkdir(parents=True, exist_ok=True)
            answer_path.write_text(json.dumps({"decision": "approve", "note": "console"}))

        monkeypatch.setattr("gomoveshift.pipeline.review_gate.time.sleep", fake_sleep)

        with patch("builtins.input") as mi:
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert not mi.called
        assert calls["n"] == 1
        assert (gates_dir / "gate_cd_d.answer.used").exists()
        assert state.gate_cd_confirmed is True

    def test_malformed_answer_file_is_left_in_place_not_destroyed(self, tmp_path, monkeypatch):
        """A malformed ``.answer`` used to be silently renamed ``.used`` and
        discarded, destroying the operator's decision. It must now stay put
        with a ``.answer.rejected`` sidecar recording the parse error."""
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        gates_dir = tmp_path / "_gates"
        gates_dir.mkdir()
        answer_path = gates_dir / "gate_cd_d.answer"
        answer_path.write_text("{not valid json")

        with patch("builtins.input") as mi:
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert not mi.called
        assert answer_path.exists()
        assert answer_path.read_text() == "{not valid json"
        assert not (gates_dir / "gate_cd_d.answer.used").exists()
        rejected_path = gates_dir / "gate_cd_d.answer.rejected"
        assert rejected_path.exists()
        assert rejected_path.read_text()
        assert state.gate_cd_confirmed is not True

    def test_non_interactive_honors_pre_staged_answer_file(self, tmp_path, monkeypatch):
        """A pre-staged answer file lets a non-interactive run resolve the
        gate without a TTY and WITH the waiver persisted (a genuine decision,
        unlike the silent no-answer skip above)."""
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        gates_dir = tmp_path / "_gates"
        gates_dir.mkdir()
        answer_path = gates_dir / "gate_cd_d.answer"
        answer_path.write_text(json.dumps({"decision": "approve", "note": "console approved"}))

        with patch("builtins.input") as mi:
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=tmp_path, date_key="d")

        assert not mi.called
        assert not answer_path.exists()
        assert (gates_dir / "gate_cd_d.answer.used").exists()
        assert state.gate_cd_confirmed is True
