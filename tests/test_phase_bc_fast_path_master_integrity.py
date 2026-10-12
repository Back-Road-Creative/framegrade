"""Fast-path #1 must validate master integrity via ffprobe.

Bug: when an encode is killed
mid-master, the partial output may have valid byte length on disk but no
trailing ``moov`` atom, so it's unplayable. ``phase_bc_day_first`` fast-path #1
previously trusted ``Path(master_path).exists()`` alone — it would skip the
re-encode and reconstruct ``DayArtifacts`` from the corrupt master, which
either crashed downstream or shipped bad content.

Fix: in fast-path #1, additionally call ``FFprobeClient(subprocess).probe_duration``
on the master and compare to ``batch.total_duration``. If actual < expected * 0.99,
fall through — preferring the finals-only fast-path #2 when the day_dir has
finals on disk, else appending to ``remaining_batches`` for re-encode. When
``batch.total_duration <= 0`` (synthesised batches), skip the integrity check
and emit a WARNING so the operator sees the gap in logs.

The atomic rename of the master output is a separate fix; this check is the
safety net.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _state_helpers import make_state


def _make_state(tmp_path, dates=("2026-02-20", "2026-02-21"), per_day_total=1500.0):
    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    episodes = []
    episode_files = []
    episode_dates_list = []
    file_durations = []
    file_boundaries = []
    cursor = 0.0
    for i, d in enumerate(dates):
        ep_path = f"/fake/ep_{i:03d}.mp4"
        episode_files.append(ep_path)
        episode_dates_list.append(d)
        file_durations.append(per_day_total)
        file_boundaries.append((cursor, cursor + per_day_total))
        cursor += per_day_total
        episodes.append(
            SimpleNamespace(
                index=i,
                concat_path=Path(ep_path),
                date=d,
                total_duration=per_day_total,
                gpx_path=None,
                creation_time=None,
            )
        )
    ingest_plan = SimpleNamespace(episodes=episodes)

    state = make_state(
        output_base=output_base,
        ingest_plan=ingest_plan,
        episode_files=episode_files,
        episode_dates=episode_dates_list,
        episode_locations=[""] * len(dates),
        file_durations=file_durations,
        region="usa",
        location="Test",
        drive_scores=None,
        file_boundaries=file_boundaries,
    )
    return state, output_base, stages_dir


def _run_phase(state, run_all_days_impl, probe_duration_map=None):
    """Run the phase under the same patches the existing tests use, plus a
    patched ``FFprobeClient`` so we control what fast-path #1 sees.

    ``probe_duration_map`` maps master Path → returned float duration. A
    missing key returns 0.0 (mirrors the real client's failure mode).
    """
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        PhaseBCDayFirst,
    )

    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None

    probe_calls: list[Path] = []

    class _FakeFFprobeClient:
        def __init__(self, _subprocess_module, *, timeout=30):
            pass

        def probe_duration(self, path):
            probe_calls.append(Path(path))
            if probe_duration_map is None:
                return 0.0
            return float(probe_duration_map.get(Path(path), 0.0))

    cfg = Config()
    phase = PhaseBCDayFirst(cfg)
    with (
        patch(
            "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
            return_value=geocoder,
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=run_all_days_impl,
        ) as run_mock,
        patch(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient",
            _FakeFFprobeClient,
        ),
    ):
        phase.run(state, probe_duration_fn=lambda p: 100.0)
    return run_mock, probe_calls


def _write_finals(day_dir: Path) -> None:
    """Create the minimum lf_NN.mp4 + short_NN.mp4 set that
    ``_day_dir_has_finals`` requires."""
    lf_dir = day_dir / "regular_lfs"
    sh_dir = day_dir / "shorts"
    lf_dir.mkdir(parents=True, exist_ok=True)
    sh_dir.mkdir(parents=True, exist_ok=True)
    (lf_dir / "lf_01.mp4").write_bytes(b"final-lf-bytes")
    (sh_dir / "short_01.mp4").write_bytes(b"final-short-bytes")


class TestReconstructExcludesEmptyFinals:
    """A 0-byte / partial final (crashed or recovery-marked day) must NOT be
    reconstructed into the resume upload set, where it would bypass the OST
    audit gate + the apply_ost re-burn loop."""

    def test_zero_byte_finals_are_excluded(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _reconstruct_day_artifacts_from_disk,
        )

        day_dir = tmp_path / "day_2026-02-20"
        lf_dir = day_dir / "regular_lfs"
        sh_dir = day_dir / "shorts"
        lf_dir.mkdir(parents=True)
        sh_dir.mkdir(parents=True)
        (lf_dir / "lf_01.mp4").write_bytes(b"final-lf-bytes")
        (lf_dir / "lf_02.mp4").write_bytes(b"")  # 0-byte partial
        (sh_dir / "short_01.mp4").write_bytes(b"final-short-bytes")
        (sh_dir / "short_02.mp4").write_bytes(b"")  # 0-byte partial

        artifacts = _reconstruct_day_artifacts_from_disk("2026-02-20", None, day_dir)

        assert [p.name for p in artifacts.regular_lf_paths] == ["lf_01.mp4"]
        assert [p.name for p in artifacts.short_paths] == ["short_01.mp4"]


class TestFastPath1MasterIntegrityProbe:
    """Fast-path #1 must ffprobe-validate masters before trusting them."""

    def test_master_passes_duration_probe_fast_path_fires(self, tmp_path):
        """Master exists AND probe_duration >= expected * 0.99 → skip re-encode,
        reconstruct from disk (skipped_artifacts grows, run_all_days sees no batches).
        """
        state, _, stages_dir = _make_state(tmp_path)
        d = "2026-02-20"
        master = stages_dir / f"day_{d}_master.mp4"
        master.write_bytes(b"x" * 1024)  # plausible bytes
        # Only one of the two days is recorded — the other should fall through naturally
        state["per_day_completed"] = {
            d: {"master_path": str(master), "master_sha": "abc"},
        }

        seen_batches: list[str] = []

        def _fake(**kwargs):
            seen_batches.extend(b.date for b in kwargs.get("day_batches") or [])
            artifacts = []
            for batch in kwargs.get("day_batches") or []:
                m = stages_dir / f"day_{batch.date}_master.mp4"
                m.write_bytes(b"freshly-encoded")
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=m,
                        regular_lf_paths=[],
                        short_paths=[],
                    )
                )
            return artifacts

        # Probe returns full duration (1500s, equal to expected) for the recorded master.
        _, probe_calls = _run_phase(
            state,
            _fake,
            probe_duration_map={master: 1500.0},
        )

        # Fast-path #1 fired for the recorded day: no re-encode for it.
        assert d not in seen_batches
        # The probe was actually consulted (otherwise the test isn't proving anything).
        assert master in probe_calls

    def test_master_short_duration_falls_through_to_finals_only_path(self, tmp_path):
        """Master exists + probe < expected*0.99 + finals on disk → fall
        through to fast-path #2 (finals-only reconstruction). No re-encode.
        """
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20",))
        d = "2026-02-20"
        master = stages_dir / f"day_{d}_master.mp4"
        master.write_bytes(b"truncated-bytes")  # partial encode survivor

        # Finals are on disk from a prior successful run.
        day_dir = stages_dir / f"day_{d}"
        _write_finals(day_dir)

        state["per_day_completed"] = {
            d: {"master_path": str(master), "master_sha": "abc"},
        }

        seen_batches: list[str] = []

        def _fake(**kwargs):
            seen_batches.extend(b.date for b in kwargs.get("day_batches") or [])
            return []

        # Probe returns ~half the expected duration → master is corrupt.
        _, probe_calls = _run_phase(
            state,
            _fake,
            probe_duration_map={master: 700.0},  # < 1500 * 0.99
        )

        # Fast-path #1 was rejected; fast-path #2 caught it; no re-encode.
        assert seen_batches == [], (
            "Day with finals on disk must be reconstructed without re-encoding "
            "even when the recorded master fails the duration probe."
        )
        assert master in probe_calls

    def test_master_short_no_finals_goes_to_remaining_batches(self, tmp_path):
        """Master exists + probe < expected*0.99 + NO finals → re-encode."""
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20",))
        d = "2026-02-20"
        master = stages_dir / f"day_{d}_master.mp4"
        master.write_bytes(b"corrupt-master")

        # No finals written — day_dir does not exist (or exists empty).
        state["per_day_completed"] = {
            d: {"master_path": str(master), "master_sha": "abc"},
        }

        seen_batches: list[str] = []

        def _fake(**kwargs):
            seen_batches.extend(b.date for b in kwargs.get("day_batches") or [])
            artifacts = []
            for batch in kwargs.get("day_batches") or []:
                m = stages_dir / f"day_{batch.date}_master.mp4"
                m.write_bytes(b"re-encoded")
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=m,
                        regular_lf_paths=[],
                        short_paths=[],
                    )
                )
            return artifacts

        _, probe_calls = _run_phase(
            state,
            _fake,
            probe_duration_map={master: 200.0},  # well under expected
        )

        assert seen_batches == [d], (
            "Corrupt master with no finals must trigger re-encoding (remaining_batches path)."
        )
        assert master in probe_calls

    def test_zero_total_duration_skips_probe_and_logs_warning(self, tmp_path, caplog):
        """When ``batch.total_duration <= 0`` we cannot compute the threshold,
        so existence-only behaviour is preserved AND a WARNING is emitted so
        the integrity gap is visible in logs.
        """
        state, _, stages_dir = _make_state(tmp_path, dates=("2026-02-20",), per_day_total=0.0)
        d = "2026-02-20"
        master = stages_dir / f"day_{d}_master.mp4"
        master.write_bytes(b"unverifiable-but-existing")
        state["per_day_completed"] = {
            d: {"master_path": str(master), "master_sha": "abc"},
        }

        seen_batches: list[str] = []

        def _fake(**kwargs):
            seen_batches.extend(b.date for b in kwargs.get("day_batches") or [])
            return []

        # If anything probes when expected_duration <= 0, that's a regression.
        called = {"probe": False}

        class _Tripwire:
            def __init__(self, _subprocess_module, *, timeout=30):
                pass

            def probe_duration(self, path):  # pragma: no cover — must not run
                called["probe"] = True
                return 0.0

        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None
        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        with (
            caplog.at_level(
                logging.WARNING, logger="gomoveshift.pipeline.drive_phases.phase_bc_day_first"
            ),
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_fake,
            ),
            patch(
                "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient",
                _Tripwire,
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        # Fast-path #1 still fired (existence-only behaviour preserved):
        assert seen_batches == []
        assert called["probe"] is False, (
            "When expected_duration <= 0 we must NOT call ffprobe — "
            "there's no threshold to compare against."
        )
        # And the operator must see the gap in logs.
        warn_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "integrity" in m.lower() or "unverified" in m.lower() or "skipped" in m.lower()
            for m in warn_msgs
        ), f"Expected a WARNING about skipped integrity validation; got: {warn_msgs!r}"


class TestReconstructedShortfallGate:
    """A resume-reconstructed day is count-matched against the CURRENT
    shorts target, so a raised --shorts-target can't silently ship the old
    lower count."""

    def _phase(self):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

        return PhaseBCDayFirst(Config())

    def _reconstructed(self, n_shorts, n_lfs=1):
        return SimpleNamespace(
            date="2026-02-20",
            regular_lf_paths=[Path(f"lf_{i + 1:02d}.mp4") for i in range(n_lfs)],
            short_paths=[Path(f"short_{i + 1:02d}.mp4") for i in range(n_shorts)],
        )

    @pytest.fixture(autouse=True)
    def _decided(self, monkeypatch):
        # Waypost's decided per-LF count. 20 keeps every --shorts-target in this
        # class at or below the decision, so the flag is the binding number.
        from gomoveshift.pipeline import volume_source

        self.decided = {"value": 20}
        monkeypatch.setattr(
            volume_source,
            "shorts_per_lf",
            lambda db=None: volume_source.ShortsDecision(self.decided["value"], "policy", "TEST"),
        )

    def test_reconstructed_day_below_raised_target_raises(self):
        # Config default: 2 shorts windows/day × 7 = 14/LF required.
        state = SimpleNamespace(shorts_target=14, shorts_per_day=None, allow_short_count=None)
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT.*reconstructed"):
            self._phase()._enforce_reconstructed_shortfall(state, self._reconstructed(9))

    def test_reconstructed_day_meeting_target_passes(self):
        state = SimpleNamespace(shorts_target=14, shorts_per_day=None, allow_short_count=None)
        self._phase()._enforce_reconstructed_shortfall(state, self._reconstructed(14))

    def test_allow_short_count_escapes_the_gate(self):
        state = SimpleNamespace(shorts_target=14, shorts_per_day=None, allow_short_count=True)
        self._phase()._enforce_reconstructed_shortfall(state, self._reconstructed(9))

    def test_capped_day_resumes_without_raising(self):
        # A capacity-capped day: 4 LFs, 4 x 14 = 56 planned, capacity-capped to 49
        # and recorded as such — the CURRENT target (14) matches what the
        # day actually ran under, so the recorded post-cap expectation wins.
        state = SimpleNamespace(shorts_target=14, shorts_per_day=None, allow_short_count=None)
        entry = {"shorts_target_in_force": 14, "shorts_expected": 49}
        self._phase()._enforce_reconstructed_shortfall(
            state, self._reconstructed(49, n_lfs=4), entry
        )

    def test_capped_day_still_gates_a_raised_target(self):
        # Same recorded (capped) day, but the operator has since RAISED
        # --shorts-target to 20 — the recorded expectation no longer applies
        # (it was recorded for 14), so the strict 20*4=80 re-derivation binds
        # and 49 < 80 must still raise.
        state = SimpleNamespace(shorts_target=20, shorts_per_day=None, allow_short_count=None)
        entry = {"shorts_target_in_force": 14, "shorts_expected": 49}
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT.*reconstructed"):
            self._phase()._enforce_reconstructed_shortfall(
                state, self._reconstructed(49, n_lfs=4), entry
            )

    def test_no_target_in_force_disables_the_gate(self):
        # No explicit --shorts-target -> the gate is inert regardless of the
        # recorded entry or how few shorts are on disk.
        state = SimpleNamespace(shorts_target=None, shorts_per_day=None, allow_short_count=None)
        self._phase()._enforce_reconstructed_shortfall(state, self._reconstructed(0, n_lfs=1))

    def test_no_state_target_returns_none_even_if_config_carries_retired_key(self):
        # state.shorts_target is the only explicit target. A leftover
        # config attribute named like the retired override is never read.
        phase = self._phase()
        phase.config.drive.__dict__["shorts_per_lf_count"] = 14
        state = SimpleNamespace(shorts_target=None, shorts_per_day=None, allow_short_count=None)
        assert phase._reconstructed_shortfall_target(state, 2) is None
        self._phase()._enforce_reconstructed_shortfall(state, self._reconstructed(0, n_lfs=1))

    def test_entry_without_new_keys_keeps_strict_behaviour(self):
        # A recorded entry from before these keys existed (or one with no shorts bookkeeping at
        # all) — falls through to the strict per_lf * num_lfs derivation, so
        # a genuinely short day (49 < 14*4=56) still raises.
        state = SimpleNamespace(shorts_target=14, shorts_per_day=None, allow_short_count=None)
        entry = {"master_path": "/fake/day_master.mp4"}
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT.*reconstructed"):
            self._phase()._enforce_reconstructed_shortfall(
                state, self._reconstructed(49, n_lfs=4), entry
            )

    def test_flag_above_the_decision_is_a_cap_not_a_raised_target(self):
        # --shorts-target can only lower Waypost's count (volume_source.apply_flag):
        # a day planned at the decided 7 under a habitual --shorts-target 14 ran
        # under 7, so a resume with the same flag must not demand 14.
        self.decided["value"] = 7
        state = SimpleNamespace(shorts_target=14, shorts_per_day=None, allow_short_count=None)
        phase = self._phase()
        assert phase._reconstructed_shortfall_target(state, 2) == 14
        entry = {"shorts_target_in_force": 7, "shorts_expected": 7}
        phase._enforce_reconstructed_shortfall(state, self._reconstructed(7), entry)

    def test_flag_raised_toward_the_decision_still_gates(self):
        # First run --shorts-target 2, resume --shorts-target 5 under a decided 7:
        # the operator raised the cap, so the completed day is short of 5.
        self.decided["value"] = 7
        state = SimpleNamespace(shorts_target=5, shorts_per_day=None, allow_short_count=None)
        entry = {"shorts_target_in_force": 2, "shorts_expected": 2}
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT.*reconstructed"):
            self._phase()._enforce_reconstructed_shortfall(state, self._reconstructed(2), entry)

    def test_no_flag_never_reads_the_decision_store(self, monkeypatch):
        from gomoveshift.pipeline import volume_source

        def _boom(db=None):
            raise AssertionError("decision store read with no --shorts-target")

        monkeypatch.setattr(volume_source, "shorts_per_lf", _boom)
        state = SimpleNamespace(shorts_target=None, shorts_per_day=None, allow_short_count=None)
        assert self._phase()._reconstructed_shortfall_target(state, 2) is None


class TestRecordDayAndSaveShortsBookkeeping:
    """``_record_day_and_save`` persists the post-cap shorts
    expectation alongside ``master_path`` so a later resume can recognize a
    legitimately-capped day (see ``TestReconstructedShortfallGate``)."""

    def _artifacts(self, tmp_path, planned_shorts_count, shorts_target_in_force):
        master = tmp_path / "day_2026-02-20_master.mp4"
        master.write_bytes(b"x")
        return SimpleNamespace(
            date="2026-02-20",
            master_path=master,
            master_record=None,
            planned_shorts_count=planned_shorts_count,
            shorts_target_in_force=shorts_target_in_force,
        )

    def test_writes_both_keys_when_capped(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _record_day_and_save

        state = SimpleNamespace(per_day_completed={})
        artifacts = self._artifacts(tmp_path, planned_shorts_count=49, shorts_target_in_force=14)
        _record_day_and_save(artifacts, state, state_save_fn=None)
        entry = state.per_day_completed["2026-02-20"]
        assert entry["shorts_target_in_force"] == 14
        assert entry["shorts_expected"] == 49

    def test_omits_both_keys_when_no_target_in_force(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _record_day_and_save

        state = SimpleNamespace(per_day_completed={})
        artifacts = self._artifacts(
            tmp_path, planned_shorts_count=None, shorts_target_in_force=None
        )
        _record_day_and_save(artifacts, state, state_save_fn=None)
        entry = state.per_day_completed["2026-02-20"]
        assert "shorts_target_in_force" not in entry
        assert "shorts_expected" not in entry
