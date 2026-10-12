"""Review-gate prompts must work without a TTY and must never
silently pass when nobody answers in time (console jobs with review left
ON have no console attached to stdin -- ``input()`` used to raise
``EOFError``; ``_prompt``/``_prompt_required`` now route through the same
answer-file mechanism Gate C/D already used)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.review_gate import ReviewFailedError, ReviewGate

from _state_helpers import make_state


def _good_ffprobe_side_effect(cmd, **kwargs):
    if "-show_streams" in cmd:
        streams = [
            {"codec_type": "video", "duration": "600.0"},
            {"codec_type": "audio", "duration": "600.0"},
        ]
        return MagicMock(returncode=0, stdout=json.dumps({"streams": streams}))
    if "-show_format" in cmd:
        return MagicMock(returncode=0, stdout=json.dumps({"format": {"bit_rate": "500000000"}}))
    return MagicMock(returncode=0, stdout="{}")


def _write_preview_clip(output_dir: Path) -> None:
    preview_dir = output_dir / "_stages" / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "ep_001_preview.mp4").touch()


def _write_answer(output_dir: Path, gate_id: str, decision: str) -> None:
    gates_dir = output_dir / "_gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    (gates_dir / f"{gate_id}.answer").write_text(json.dumps({"decision": decision}))


def _expire_timeout(monkeypatch, seconds: str = "10") -> None:
    """Make the next poll's deadline expire on its first sleep."""
    monkeypatch.setenv("GMS_GATE_ANSWER_TIMEOUT_SEC", seconds)
    fake_now = {"t": 0.0}
    monkeypatch.setattr("gomoveshift.pipeline.review_gate.time.monotonic", lambda: fake_now["t"])
    monkeypatch.setattr(
        "gomoveshift.pipeline.review_gate.time.sleep",
        lambda s: fake_now.__setitem__("t", fake_now["t"] + s + 1),
    )


class TestGatePreviewNonTty:
    def test_non_tty_approve_answer_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        _write_preview_clip(tmp_path)
        _write_answer(tmp_path, "gate_preview", "approve")
        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            with patch("builtins.input") as mi:
                ReviewGate(interactive=True).gate_preview(output_dir=tmp_path)
        assert not mi.called
        assert (tmp_path / "_gates" / "gate_preview.answer.used").exists()

    def test_non_tty_reject_answer_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        _write_preview_clip(tmp_path)
        _write_answer(tmp_path, "gate_preview", "reject")
        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            with patch("builtins.input") as mi, pytest.raises(ReviewFailedError):
                ReviewGate(interactive=True).gate_preview(output_dir=tmp_path)
        assert not mi.called

    def test_non_tty_timeout_expiry_raises_and_leaves_question(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        _expire_timeout(monkeypatch)
        _write_preview_clip(tmp_path)
        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            with patch("builtins.input") as mi, pytest.raises(ReviewFailedError):
                ReviewGate(interactive=True).gate_preview(output_dir=tmp_path)
        assert not mi.called
        assert (tmp_path / "_gates" / "gate_preview.question.json").exists()

    def test_tty_path_still_uses_input(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        _write_preview_clip(tmp_path)
        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            with patch("builtins.input", return_value="y") as mi:
                ReviewGate(interactive=True).gate_preview(output_dir=tmp_path)
        assert mi.called


class TestGateAAndGateBNonTty:
    def test_gate_a_non_tty_approve_answer_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        state = make_state(
            episode_files=["/tmp/ep_001.mp4"],
            episode_dates=["2026-03-01"],
            file_durations=[3600.0],
            gps_per_episode=[{"lat": 51.77, "lon": -9.98}],
        )
        _write_answer(tmp_path, "gate_a", "approve")
        with patch("builtins.input") as mi:
            ReviewGate(interactive=True).gate_a(
                state=state, output_dir=tmp_path, location="County Kerry, Ireland"
            )
        assert not mi.called
        assert (tmp_path / "_gates" / "gate_a.answer.used").exists()

    def test_gate_b_non_tty_approve_answer_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        final_file = tmp_path / "ep_001_FINAL.mp4"
        final_file.touch()
        state = make_state(processed_files={0: str(final_file)})
        _write_answer(tmp_path, "gate_b", "approve")
        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            with patch("builtins.input") as mi:
                ReviewGate(interactive=True).gate_b(state=state, output_dir=tmp_path)
        assert not mi.called
        assert (tmp_path / "_gates" / "gate_b.answer.used").exists()


class TestGateCdTimeoutExpiryFailsInsteadOfProceeding:
    def test_no_review_timeout_expiry_fails_instead_of_proceeding(self, tmp_path, monkeypatch):
        """A console (--no-review, timeout set) run whose question is
        never answered must FAIL, never proceed to the irreversible
        uploads -- before the fix this returned rc 0 with review skipped."""
        monkeypatch.setattr(ReviewGate, "_stream_count", lambda self, p: 2)
        monkeypatch.setattr(ReviewGate, "_av_drift_pct", lambda self, p: 0.5)
        monkeypatch.setattr(ReviewGate, "_probe_av_durations", lambda self, p: (600.0, 600.0))
        output_dir = tmp_path
        (output_dir / "lf_01_youtube.json").write_text(
            json.dumps({"title": "LF 1 Title", "description": "x"}, sort_keys=True)
        )
        (output_dir / "lf_01.mp4").write_bytes(b"")
        state = make_state(regular_lf_paths=[str(output_dir / "lf_01.mp4")])
        _expire_timeout(monkeypatch)

        with patch("builtins.input") as mi, pytest.raises(ReviewFailedError):
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=output_dir, date_key="d")

        assert not mi.called
        assert (output_dir / "_gates" / "gate_cd_d.question.json").exists()
        assert state.gate_cd_confirmed is not True
