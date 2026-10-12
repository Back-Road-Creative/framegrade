"""Gate C/D ``--gate-cd-confirmed`` flag + content-hash auto-pass."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.review_gate import (
    ReviewFailedError,
    ReviewGate,
    compute_gate_cd_hash,
)

from _state_helpers import make_state


def _stub_ffprobe(monkeypatch):
    monkeypatch.setattr(ReviewGate, "_stream_count", lambda self, p: 2)
    monkeypatch.setattr(ReviewGate, "_av_drift_pct", lambda self, p: 0.5)
    monkeypatch.setattr(ReviewGate, "_probe_av_durations", lambda self, p: (600.0, 600.0))


def _write_lf_set(output_dir: Path, lf_count: int = 2):
    output_dir.mkdir(parents=True, exist_ok=True)
    lf_paths = []
    for idx in range(1, lf_count + 1):
        (output_dir / f"lf_{idx:02d}_youtube.json").write_text(
            json.dumps({"title": f"LF {idx} Title", "description": "x"}, sort_keys=True)
        )
        (output_dir / f"lf_{idx:02d}.mp4").write_bytes(b"")
        lf_paths.append(str(output_dir / f"lf_{idx:02d}.mp4"))
    (output_dir / "proj_shorts_metricool.csv").write_text("c1,c2,c3\nv1,v2,v3\nv4,v5,v6\n")
    return make_state(regular_lf_paths=lf_paths)


class TestComputeGateCdHash:
    def test_hash_excludes_wallclock(self, tmp_path):
        state = _write_lf_set(tmp_path, 2)
        a = compute_gate_cd_hash(state=state, output_dir=tmp_path)
        future = time.time() + 3600
        for p in tmp_path.iterdir():
            os.utime(p, (future, future))
        assert a == compute_gate_cd_hash(state=state, output_dir=tmp_path)

    def test_hash_changes_on_title_edit(self, tmp_path):
        state = _write_lf_set(tmp_path, 2)
        before = compute_gate_cd_hash(state=state, output_dir=tmp_path)
        meta = tmp_path / "lf_01_youtube.json"
        d = json.loads(meta.read_text())
        d["title"] = "Edited"
        meta.write_text(json.dumps(d, sort_keys=True))
        assert before != compute_gate_cd_hash(state=state, output_dir=tmp_path)

    def test_hash_changes_on_csv_edit(self, tmp_path):
        """Header + row-count are both covered: editing CSV invalidates the hash."""
        state = _write_lf_set(tmp_path, 2)
        before = compute_gate_cd_hash(state=state, output_dir=tmp_path)
        (tmp_path / "proj_shorts_metricool.csv").write_text("cA,cB,cC\nv1,v2,v3\n")
        assert before != compute_gate_cd_hash(state=state, output_dir=tmp_path)

    def test_hash_changes_on_publish_plan_edit(self, tmp_path):
        """A publish-plan sidecar's content is covered by the gate hash — a plan
        change (e.g. a different schedule/route) must reopen Gate C/D."""
        state = _write_lf_set(tmp_path, 2)
        (tmp_path / "proj_shorts_publish_plan.json").write_text(
            json.dumps({"shorts": [{"lf_index": 1, "short_index": 0}]}, sort_keys=True)
        )
        before = compute_gate_cd_hash(state=state, output_dir=tmp_path)
        (tmp_path / "proj_shorts_publish_plan.json").write_text(
            json.dumps({"shorts": [{"lf_index": 1, "short_index": 1}]}, sort_keys=True)
        )
        assert before != compute_gate_cd_hash(state=state, output_dir=tmp_path)

    def test_hash_unchanged_when_no_plan_sidecar_present(self, tmp_path):
        """No code path writes a ``*_publish_plan.json`` sidecar today, so this
        fixture (LFs + a metricool CSV, no plan file) must hash to the exact
        digest the ORIGINAL (pre-plan-coverage) function produced — proving the
        new glob is a no-op for every run that exists today."""
        state = _write_lf_set(tmp_path, 2)
        assert (
            compute_gate_cd_hash(state=state, output_dir=tmp_path)
            == "868f9995c1e207639f25dd69d7ea53b5957bde7047edb9547a8fc03ad81da524"  # pragma: allowlist secret
        )

    def test_hash_stable_on_malformed_plan_sidecar(self, tmp_path):
        """A malformed/unreadable plan sidecar degrades to an empty payload,
        like the existing CSV loop -- it must never crash the gate."""
        state = _write_lf_set(tmp_path, 2)
        (tmp_path / "proj_shorts_publish_plan.json").write_text("{not valid json")
        # Must not raise.
        digest = compute_gate_cd_hash(state=state, output_dir=tmp_path)
        assert isinstance(digest, str) and len(digest) == 64


class TestGateCdAutoPass:
    def test_tty_no_flag_prompts(self, tmp_path, monkeypatch):
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        gate = ReviewGate(interactive=True)
        with patch("builtins.input", return_value="n") as mi:
            with pytest.raises(ReviewFailedError):
                gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-05-18")
        assert mi.called

    def test_flag_hash_match_skips(self, tmp_path, monkeypatch):
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        state["gate_cd_confirmed"] = True
        state["gate_cd_confirmation_hash"] = compute_gate_cd_hash(state=state, output_dir=tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        with patch("builtins.input") as mi:
            ReviewGate(interactive=True).gate_cd(state=state, output_dir=tmp_path, date_key="d")
        assert not mi.called

    def test_flag_hash_mismatch_reprompts(self, tmp_path, monkeypatch):
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        state["gate_cd_confirmed"] = True
        state["gate_cd_confirmation_hash"] = "stale_" + "0" * 58
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        with patch("builtins.input", return_value="n") as mi:
            with pytest.raises(ReviewFailedError):
                ReviewGate(interactive=True).gate_cd(state=state, output_dir=tmp_path, date_key="d")
        assert mi.called

    def test_non_tty_interactive_hard_fails_and_does_not_persist(self, tmp_path, monkeypatch):
        """Phase-0.4 hole: an interactive run whose stdin is not a TTY must NOT
        silently auto-pass and must NOT persist a waiver — one headless run used
        to waive the gate for every later run of that state. It now hard-fails.
        """
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("builtins.input") as mi:
            with pytest.raises(ReviewFailedError):
                ReviewGate(interactive=True).gate_cd(state=state, output_dir=tmp_path, date_key="d")
        assert not mi.called
        # No waiver may be persisted — a later human run must still be prompted.
        assert state.gate_cd_confirmed is not True
        assert state.gate_cd_confirmation_hash is None

    def test_no_review_skips_without_persisting_waiver(self, tmp_path, monkeypatch):
        """--no-review (interactive=False) skips the prompt like every gate but
        must NOT persist gate_cd_confirmed/hash (that persist is the poison)."""
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("builtins.input") as mi:
            ReviewGate(interactive=False).gate_cd(state=state, output_dir=tmp_path, date_key="d")
        assert not mi.called
        assert state.gate_cd_confirmed is not True
        assert state.gate_cd_confirmation_hash is None

    def test_tty_force_alone_does_not_skip(self, tmp_path, monkeypatch):
        """TTY + only --force (no gate_cd_confirmed) → prompt still fires."""
        _stub_ffprobe(monkeypatch)
        state = _write_lf_set(tmp_path, 1)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        with patch("builtins.input", return_value="n") as mi:
            with pytest.raises(ReviewFailedError):
                ReviewGate(interactive=True).gate_cd(state=state, output_dir=tmp_path, date_key="d")
        assert mi.called
