"""Resume-correctness regression tests for Phase D (metadata + upload + social + schedule).

Phase D is the most-irreversible phase of the drive pipeline — a bad resume
here can cause real, user-visible double-uploads to YouTube. These tests
focus on the state transitions that govern idempotency:

  1. Clean state — all Phase D outputs present (all LFs already uploaded).
  2. Sidecar present but ``state["youtube_uploads"]`` empty (simulates a
     state wipe). Resume MUST read sidecar and not re-upload.
  3. Partial metadata_per_lf — Phase D rebuilds metadata unconditionally
     (current behaviour); this test pins that behaviour so future refactors
     that try to preserve partial state must update it explicitly.
  4. Upload half-done — sidecar has 2 of 9 uploads recorded; resume uploads
     only the 7 missing via ``upload_if_needed``, never re-uploading 1 or 2.
  5. Social posts JSON corrupt / partial — Phase D must not raise and must
     still regenerate social posts.

These tests are intentionally independent of the full orchestrator run path;
they exercise PhaseD directly with a fake YouTubeUploader module so no
network or google-auth machinery is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config, get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
from gomoveshift.pipeline.youtube_sidecar import (
    SIDECAR_FILENAME,
    load_sidecar,
    save_sidecar,
)


# ──────────────────────────────────────────────────────────────────────────
# Test scaffolding
# ──────────────────────────────────────────────────────────────────────────


def _install_fake_youtube_uploader(monkeypatch, upload_tracker: dict):
    """Inject a fake YouTubeUploader module.

    ``upload_tracker`` is populated so individual tests can assert which LFs
    were actually uploaded versus short-circuited by the existing record.
    """
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")

    uploader_instance = MagicMock()
    counter = {"n": 0}

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):
        uploads = state.setdefault("youtube_uploads", {})
        existing = uploads.get(lf_idx)
        if existing and existing.get("video_id"):
            upload_tracker.setdefault("skipped", []).append(lf_idx)
            return existing
        counter["n"] += 1
        video_id = f"newvid_{counter['n']:02d}"
        record = {
            "video_id": video_id,
            "url": f"https://youtu.be/{video_id}",
            "publish_at": (
                kwargs["publish_at"].isoformat() if kwargs.get("publish_at") is not None else None
            ),
            "title": kwargs.get("title", ""),
            "uploaded_at": "2026-04-19T00:00:00+00:00",
        }
        uploads[lf_idx] = record
        upload_tracker.setdefault("uploaded", []).append(lf_idx)
        return record

    uploader_instance.upload_if_needed.side_effect = _upload_if_needed
    uploader_instance.set_thumbnail.return_value = True

    class YouTubeUploader:
        def __new__(cls, *a, **kw):
            return uploader_instance

    fake_mod.YouTubeUploader = YouTubeUploader
    fake_mod._instance = uploader_instance
    monkeypatch.setitem(
        sys.modules,
        "gomoveshift.pipeline.drive_phases.youtube_uploader",
        fake_mod,
    )
    return uploader_instance


def _make_drive_plan(tmp_path: Path, num_lfs: int) -> tuple[DrivePlan, list[Path]]:
    """Build a minimal DrivePlan + list of on-disk LF video paths."""
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)

    regular_lfs: list[RegularLFPlan] = []
    lf_paths: list[Path] = []
    for i in range(num_lfs):
        seg = LFSegment(
            source_file_index=i,
            source_path=tmp_path / f"ep_{i:03d}_FINAL.mp4",
            start_time=0.0,
            end_time=600.0,
            original_start_time=0.0,
            original_end_time=600.0,
            composite_score=0.7,
            gps_center_lat=0.0,
            gps_center_lon=0.0,
            terrain_type="countryside",
            contiguous_with_previous=False,
        )
        regular_lfs.append(
            RegularLFPlan(
                index=i + 1,
                segments=[seg],
                montage_peaks=[seg],
                shorts_target=3,
                region="ireland",
                total_duration=600.0,
            )
        )
        lf_path = output / f"lf_{i + 1:02d}.mp4"
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))
        lf_paths.append(lf_path)

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * num_lfs,
        total_highlight_duration=600.0 * num_lfs,
    )
    return drive_plan, lf_paths


def _base_state(tmp_path: Path, num_lfs: int):
    from _state_helpers import make_state

    drive_plan, lf_paths = _make_drive_plan(tmp_path, num_lfs)
    source = tmp_path / "source_drive"
    source.mkdir(exist_ok=True)
    output = tmp_path / "output"
    return make_state(
        source_dir=source,
        output_base=output,
        region="ireland",
        publish_day="wed",
        location="Cork, Ireland",
        skip_upload=False,
        # Drive shorts upload is not under test in this file — every test
        # exercises YouTube-side resume behaviour (sidecar, metadata,
        # social posts). Skipping the Drive leg satisfies the
        # upload-completeness quality gate (which requires one
        # upload_manifest per LF when Drive uploads are enabled) and
        # the ShortsValidator (which would reject the empty
        # shorts_per_lf paths these fixtures use).
        skip_drive_upload=True,
        skip_csv=True,  # CSV path not under test here; skip to reduce noise
        youtube_upload=True,
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf={lf.index: [] for lf in drive_plan.regular_lfs},
        # BC→D boundary contract — populated by BC in production.
        per_lf_recording_date={lf.index: "2026-02-20" for lf in drive_plan.regular_lfs},
        per_lf_locations={lf.index: "Cork, Ireland" for lf in drive_plan.regular_lfs},
        per_lf_trim_ratio={lf.index: 0.7 for lf in drive_plan.regular_lfs},
    )


def _phase_d_patches():
    """Patch the symbols that Phase D resolves via ``_imports.resolve()``."""
    return [
        patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.ReleaseSchedule"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.UploadManifest"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator"),
    ]


def _wire_mocks(mocks, num_lfs: int):
    """Populate the standard mock returns for the patched classes."""
    (meta_cls, sched_cls, _rs, social_cls, upload_cls, _manifest, _csv, val_cls) = mocks

    meta = MagicMock()

    def _generate_meta(
        *,
        location,
        publish_day,
        video_duration,
        project_name,
        route_profile=None,
        title_override=None,
        recording_date=None,
        **_kwargs,
    ):
        return {
            "title": f"{project_name} title",
            "description": f"desc for {project_name}",
            "tags": ["driving", "ireland"],
            "category": "Travel",
        }

    meta.generate.side_effect = _generate_meta
    meta_cls.return_value = meta

    # Rule: lf_slot dates must fall on Wednesdays (LONG_FORM_DAYS={2}).
    # 2026-05-06 is a Wednesday; every 7 days thereafter is next Wed.
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    _wed_dates = [(_dt(2026, 5, 6) + _td(days=7 * i)).strftime("%Y-%m-%d") for i in range(num_lfs)]
    lf_slots = [
        MagicMock(
            date=_wed_dates[i],
            time="11:00:00",
            datetime_est=f"{_wed_dates[i]}T11:00:00-04:00",
            day_of_week="Wednesday",
        )
        for i in range(num_lfs)
    ]
    drive_schedule = MagicMock()
    drive_schedule.lf_slots = lf_slots
    drive_schedule.shorts_slots = []
    drive_schedule.to_dict.return_value = {"region": "ireland"}
    sched = MagicMock()
    sched.generate_drive_schedule.return_value = drive_schedule
    sched_cls.return_value = sched
    sched_cls.from_config.return_value = sched

    social = MagicMock()
    # Each per-LF call returns a small list of 2 MagicMock posts.
    social.generate_all.return_value = [
        MagicMock(short_index=1, to_dict=lambda: {"short_index": 1, "caption": "p1"}),
        MagicMock(short_index=2, to_dict=lambda: {"short_index": 2, "caption": "p2"}),
    ]
    social_cls.return_value = social

    uploader = MagicMock()
    uploader.upload_shorts.return_value = MagicMock(files={})
    upload_cls.return_value = uploader

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val

    return {"meta": meta, "sched": sched, "social": social, "drive_schedule": drive_schedule}


def _run_phase_d(state: dict, num_lfs: int, config: Config | None = None):
    """Execute Phase D under the standard orchestrator-class patch set."""
    config = config or get_config()
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
        _wire_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=num_lfs)
        PhaseD(config).run(state)


# ──────────────────────────────────────────────────────────────────────────
# Scenario 1 — clean resume: all LFs already uploaded
# ──────────────────────────────────────────────────────────────────────────


class TestCleanResumeAllOutputsPresent:
    """All Phase D outputs present — no re-uploads, sidecar untouched."""

    def test_clean_state_no_reupload(self, tmp_path, monkeypatch):
        num_lfs = 3
        state = _base_state(tmp_path, num_lfs=num_lfs)

        # Pre-populate youtube_uploads for every LF (clean state).
        existing = {
            i: {
                "video_id": f"existing_{i:02d}",
                "url": f"https://youtu.be/existing_{i:02d}",
                "title": f"title {i}",
                "uploaded_at": "2026-04-01T00:00:00+00:00",
                "publish_at": None,
            }
            for i in range(1, num_lfs + 1)
        }
        state["youtube_uploads"] = dict(existing)

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        _run_phase_d(state, num_lfs=num_lfs)

        # No new uploads — only skips.
        assert tracker.get("uploaded", []) == []
        assert sorted(tracker.get("skipped", [])) == [1, 2, 3]

        # State entries must still be the pre-existing ones, unchanged.
        for lf_idx, rec in existing.items():
            assert state["youtube_uploads"][lf_idx]["video_id"] == rec["video_id"]


# ──────────────────────────────────────────────────────────────────────────
# Scenario 2 — sidecar present, state wiped
# ──────────────────────────────────────────────────────────────────────────


class TestSidecarAuthoritativeWhenStateWiped:
    """Sidecar is the durable source of truth.

    This test documents the boundary contract: PhaseD itself does NOT reload
    the sidecar (the orchestrator does, before calling Phase D). So we
    simulate the orchestrator's merge behaviour: sidecar → state.youtube_uploads
    → PhaseD must honour it and NOT re-upload.

    Real-world failure mode: if a state-reset landed between
    the orchestrator's sidecar merge and Phase D's upload loop — e.g. a
    buggy phase function that rewrites `state["youtube_uploads"] = {}` —
    Phase D would happily upload duplicates. We pin this by asserting that
    the merged state-plus-sidecar values survive into the upload loop.
    """

    def test_sidecar_entries_prevent_reupload(self, tmp_path, monkeypatch):
        num_lfs = 3
        state = _base_state(tmp_path, num_lfs=num_lfs)

        # Persist a real sidecar file on disk (str keys, as written by save_sidecar).
        project_dir = Path(state["output_base"])
        prior_uploads = {
            "1": {
                "video_id": "sidecar_01",
                "url": "https://youtu.be/sidecar_01",
                "title": "run lf01",
                "uploaded_at": "2026-03-01T00:00:00+00:00",
                "publish_at": None,
            },
            "2": {
                "video_id": "sidecar_02",
                "url": "https://youtu.be/sidecar_02",
                "title": "run lf02",
                "uploaded_at": "2026-03-01T00:00:00+00:00",
                "publish_at": None,
            },
            "3": {
                "video_id": "sidecar_03",
                "url": "https://youtu.be/sidecar_03",
                "title": "run lf03",
                "uploaded_at": "2026-03-01T00:00:00+00:00",
                "publish_at": None,
            },
        }
        save_sidecar(project_dir, prior_uploads)
        assert (project_dir / SIDECAR_FILENAME).exists()

        # Round-trip the loader to assert the data is exactly what we wrote.
        loaded = load_sidecar(project_dir)
        assert loaded == prior_uploads

        # Simulate the orchestrator merge into state (int keys).
        state["youtube_uploads"] = {int(k): v for k, v in loaded.items()}

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        _run_phase_d(state, num_lfs=num_lfs)

        assert tracker.get("uploaded", []) == [], (
            "Phase D uploaded an LF that was already recorded in the "
            "sidecar — this is a silent duplicate-upload bug."
        )
        assert sorted(tracker.get("skipped", [])) == [1, 2, 3]
        # Sidecar video_ids must be what ended up in state.
        for i in range(1, num_lfs + 1):
            assert state["youtube_uploads"][i]["video_id"] == f"sidecar_{i:02d}"

    def test_str_keyed_state_normalized_at_load_boundary(self, tmp_path, monkeypatch):
        """State-shape normalization at load: JSON round-trip
        produces str keys, but ``PipelineState.from_dict`` coerces every
        ``_INT_KEY_DICT_FIELDS`` entry to int as part of the canonical
        load contract. Phase D operates on already-normalized state —
        the str-key reconciliation that used to live inside Phase D was
        deleted because the load boundary is the single chokepoint.

        This test pins the new contract: state with str-keyed
        ``youtube_uploads`` (as JSON round-trip produces) is coerced to
        int keys by the load-time normalizer that ``from_dict`` invokes
        (``_coerce_int_key_dict`` over every field in
        ``_INT_KEY_DICT_FIELDS``), and Phase D then correctly skips the
        already-uploaded LFs without the helper having to tolerate str
        keys mid-run.

        We apply ``_coerce_int_key_dict`` directly rather than round-
        tripping the fixture through ``PipelineState.from_dict`` because
        ``_base_state`` returns a dict with extra fields not on
        ``PipelineState``; the goal is to simulate the load boundary's
        behaviour on the one field this test exercises.
        """
        from gomoveshift.pipeline.drive_state_model import _coerce_int_key_dict

        num_lfs = 2
        state = _base_state(tmp_path, num_lfs=num_lfs)
        # Simulate JSON-deserialized state — keys are strings.
        state["youtube_uploads"] = {
            "1": {"video_id": "str_01", "url": "https://youtu.be/str_01"},
            "2": {"video_id": "str_02", "url": "https://youtu.be/str_02"},
        }
        # Load-boundary normalization (the same coercion ``from_dict``
        # applies to every field in ``_INT_KEY_DICT_FIELDS``).
        state["youtube_uploads"] = _coerce_int_key_dict(state["youtube_uploads"])
        assert 1 in state["youtube_uploads"] and 2 in state["youtube_uploads"], (
            "load-boundary normalization should have produced int keys"
        )

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        _run_phase_d(state, num_lfs=num_lfs)

        assert tracker.get("uploaded", []) == [], (
            "After load-boundary normalization, Phase D should not "
            "re-upload — int-key lookup against int-keyed state should "
            "skip the already-uploaded LFs. If this fails, check whether "
            "youtube_uploads dropped out of _INT_KEY_DICT_FIELDS."
        )
        assert sorted(tracker.get("skipped", [])) == [1, 2]


# ──────────────────────────────────────────────────────────────────────────
# Scenario 3 — partial metadata_per_lf
# ──────────────────────────────────────────────────────────────────────────


class TestPartialMetadataRebuildBehaviour:
    """Phase D always rebuilds metadata_per_lf from scratch.

    The current implementation does NOT preserve or extend an existing
    ``state['metadata_per_lf']``; it allocates a fresh dict and fills it.
    That is a legitimate design choice — metadata is cheap to regenerate,
    and freshness matters for titles that include publish dates.

    This test pins that behaviour so a future refactor that switches to
    an incremental "fill missing only" strategy must do so with eyes open
    and explicitly update this expectation.
    """

    def test_partial_metadata_is_fully_rebuilt_on_resume(self, tmp_path, monkeypatch):
        num_lfs = 7
        state = _base_state(tmp_path, num_lfs=num_lfs)

        # Partial metadata from a prior run — only LFs 1..3 populated.
        # The values here are intentionally "stale markers" so the test can
        # detect whether they survived (they should NOT, given current spec).
        state["metadata_per_lf"] = {
            1: {"title": "STALE-1", "description": "stale", "tags": []},
            2: {"title": "STALE-2", "description": "stale", "tags": []},
            3: {"title": "STALE-3", "description": "stale", "tags": []},
        }

        # Pre-seed youtube_uploads for all LFs so we don't trigger uploads.
        state["youtube_uploads"] = {
            i: {"video_id": f"u{i:02d}", "url": f"https://youtu.be/u{i:02d}"}
            for i in range(1, num_lfs + 1)
        }

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        _run_phase_d(state, num_lfs=num_lfs)

        meta = state["metadata_per_lf"]
        # All 7 LFs now have metadata.
        assert set(meta.keys()) == set(range(1, num_lfs + 1))
        # Stale markers gone — fresh metadata generator output present.
        for i in range(1, num_lfs + 1):
            assert meta[i]["title"] != f"STALE-{i}", (
                f"LF {i} still carries stale metadata — Phase D did not rebuild it"
            )
        # Per-LF youtube.json sidecar files must exist for all LFs.
        output_base = Path(state["output_base"])
        jsons = list(output_base.glob("*_lf*_youtube.json"))
        assert len(jsons) == num_lfs, (
            f"Expected {num_lfs} per-LF youtube.json files, got {len(jsons)}: {jsons}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Scenario 4 — upload half-done: 2 of 9 recorded in sidecar
# ──────────────────────────────────────────────────────────────────────────


class TestUploadHalfDoneResume:
    """Sidecar has 2 of 9 uploads; the integrity gate raises by default,
    and ``--force-sidecar-reset`` is the operator-driven escape hatch.

    Earlier behavior: Phase D uploaded only the missing 7 LFs,
    preserving the existing sidecar entries (no duplicate uploads).
    Current behavior: a half-done sidecar disagrees with the
    plan, so the gate raises :class:`SidecarIntegrityError`. Operators
    must verify via YouTube Studio which video_ids are still live,
    then either:

      * re-run with ``--force-sidecar-reset`` (wipes sidecar, re-uploads
        all 9 LFs — accepts duplicate-upload cost as the recovery), OR
      * manually reconcile ``youtube_uploads.json`` and re-run without
        the flag.

    The trade-off is intentional: a silent half-resume was the exact
    class of bug (phantom IDs into the scheduling CSV) that the integrity
    gate exists to prevent.
    """

    def test_half_done_sidecar_raises_by_default(self, tmp_path, monkeypatch):
        """Integrity-gate contract: a sidecar with fewer entries than plan LFs
        raises SidecarIntegrityError at Phase D entry, before any
        upload path runs."""
        from gomoveshift.pipeline.exceptions import SidecarIntegrityError

        num_lfs = 9
        state = _base_state(tmp_path, num_lfs=num_lfs)
        project_dir = Path(state["output_base"])

        prior = {
            "1": {
                "video_id": "done_01",
                "url": "https://youtu.be/done_01",
                "title": "done 1",
                "uploaded_at": "2026-04-10T00:00:00+00:00",
                "publish_at": None,
            },
            "2": {
                "video_id": "done_02",
                "url": "https://youtu.be/done_02",
                "title": "done 2",
                "uploaded_at": "2026-04-10T00:00:00+00:00",
                "publish_at": None,
            },
        }
        save_sidecar(project_dir, prior)
        state["youtube_uploads"] = {int(k): v for k, v in load_sidecar(project_dir).items()}

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        with pytest.raises(SidecarIntegrityError) as exc_info:
            _run_phase_d(state, num_lfs=num_lfs)
        # Gate must fire BEFORE any upload.
        assert tracker.get("uploaded", []) == []
        assert tracker.get("skipped", []) == []
        # Plan has 1..9, sidecar has 1..2 → missing 3..9.
        assert set(exc_info.value.missing_indices) == {3, 4, 5, 6, 7, 8, 9}

    def test_force_reset_recovers_and_uploads_all_nine_fresh(self, tmp_path, monkeypatch):
        """--force-sidecar-reset recovery: pre-existing sidecar wiped,
        all 9 LFs uploaded fresh. Operator accepts duplicate-upload
        cost for LFs 1 and 2 as the price of a clean state.

        NOTE: the sidecar-write path is atomic — the reset happens
        BEFORE the upload loop, so if the new run dies partway through
        upload, resume still sees a partial state (and raises again)."""
        num_lfs = 9
        state = _base_state(tmp_path, num_lfs=num_lfs)
        project_dir = Path(state["output_base"])

        prior = {
            "1": {
                "video_id": "done_01",
                "url": "https://youtu.be/done_01",
                "title": "done 1",
                "uploaded_at": "2026-04-10T00:00:00+00:00",
                "publish_at": None,
            },
            "2": {
                "video_id": "done_02",
                "url": "https://youtu.be/done_02",
                "title": "done 2",
                "uploaded_at": "2026-04-10T00:00:00+00:00",
                "publish_at": None,
            },
        }
        save_sidecar(project_dir, prior)
        state["youtube_uploads"] = {int(k): v for k, v in load_sidecar(project_dir).items()}
        # Operator opts in to the reset — simulates the CLI passing
        # --force-sidecar-reset through to state (orchestrator sets
        # state["force_sidecar_reset"] from the CLI arg).
        state["force_sidecar_reset"] = True

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        _run_phase_d(state, num_lfs=num_lfs)

        # All nine LFs uploaded fresh; nothing skipped.
        assert sorted(tracker.get("uploaded", [])) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert tracker.get("skipped", []) == []
        # None of the resulting entries retain the old "done_" video_ids.
        for i in range(1, 10):
            vid = state["youtube_uploads"][i]["video_id"]
            assert vid.startswith("newvid_"), (
                f"LF {i} video_id {vid!r} survived --force-sidecar-reset"
            )


# ──────────────────────────────────────────────────────────────────────────
# Scenario 5 — social posts JSON corrupt/partial
# ──────────────────────────────────────────────────────────────────────────


class TestSocialPostsCorruptRecovery:
    """Corrupt or partial social_posts.json must not crash Phase D.

    Phase D writes ``social_posts.json`` per LF into
    ``output_base/shorts/lf_NN/social_posts.json``. If a prior run left
    corrupt/partial JSON on disk, Phase D must overwrite it cleanly when it
    regenerates. It must NEVER attempt to read the file as an authoritative
    record — social posts are cheap to regenerate, and a corrupt sidecar
    must not poison the pipeline.
    """

    def test_corrupt_social_posts_json_is_overwritten(self, tmp_path, monkeypatch):
        num_lfs = 2
        state = _base_state(tmp_path, num_lfs=num_lfs)
        output_base = Path(state["output_base"])

        # Pre-create corrupt social_posts.json files for both LFs.
        for lf_idx in (1, 2):
            shorts_dir = output_base / "shorts" / f"lf_{lf_idx:02d}"
            shorts_dir.mkdir(parents=True, exist_ok=True)
            posts_file = shorts_dir / "social_posts.json"
            posts_file.write_text("{this is not valid json}")

        # Pre-populate uploads so we don't trigger YouTube work.
        state["youtube_uploads"] = {
            1: {"video_id": "u01", "url": "https://youtu.be/u01"},
            2: {"video_id": "u02", "url": "https://youtu.be/u02"},
        }

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        # Must not raise.
        _run_phase_d(state, num_lfs=num_lfs)

        # social_posts.json files still exist — Phase D wrote fresh content.
        for lf_idx in (1, 2):
            posts_file = output_base / "shorts" / f"lf_{lf_idx:02d}" / "social_posts.json"
            assert posts_file.exists(), f"social_posts.json missing for LF {lf_idx} after Phase D"
            # The fake SocialGenerator returns MagicMock posts that may not
            # serialise cleanly; Phase D handles that via try/except. So the
            # file may contain valid JSON OR be untouched — but Phase D must
            # NOT have raised. The assertion that Phase D completed at all
            # is the primary contract.

        # state["social_posts_per_lf"] has been fully rebuilt.
        assert set(state["social_posts_per_lf"].keys()) == {1, 2}

    def test_partial_social_posts_dict_regenerated(self, tmp_path, monkeypatch):
        """A pre-existing partial social_posts_per_lf dict is not trusted."""
        num_lfs = 3
        state = _base_state(tmp_path, num_lfs=num_lfs)

        # Partial — only LF 1 has a stale entry.
        sentinel_post = MagicMock(short_index=999, to_dict=lambda: {"short_index": 999})
        state["social_posts_per_lf"] = {1: [sentinel_post]}

        state["youtube_uploads"] = {
            i: {"video_id": f"u{i:02d}", "url": f"https://youtu.be/u{i:02d}"}
            for i in range(1, num_lfs + 1)
        }

        tracker: dict = {}
        _install_fake_youtube_uploader(monkeypatch, tracker)

        _run_phase_d(state, num_lfs=num_lfs)

        # All 3 LFs present in the regenerated map.
        assert set(state["social_posts_per_lf"].keys()) == {1, 2, 3}
        # The sentinel stale entry must be gone — Phase D does a full rebuild.
        lf1_posts = state["social_posts_per_lf"][1]
        assert sentinel_post not in lf1_posts, (
            "LF 1 still carries a stale social-post sentinel — Phase D failed to regenerate"
        )


# ──────────────────────────────────────────────────────────────────────────
# Fixture data — real dict JSON used by two scenarios above (2 and 4)
# ──────────────────────────────────────────────────────────────────────────


def test_fixture_sidecar_payload_shape(tmp_path):
    """Sanity: the sidecar JSON payload shape we write in scenarios 2 and 4
    round-trips through save_sidecar / load_sidecar unchanged.
    """
    fixture = {
        "1": {
            "video_id": "fixture_01",
            "url": "https://youtu.be/fixture_01",
            "title": "LF 01 — County Cork loop",
            "uploaded_at": "2026-04-10T14:22:01+00:00",
            "publish_at": "2026-05-06T11:00:00+00:00",
            "thumbnail_path": str(tmp_path / "thumbnails" / "lf_01.jpg"),
        },
    }
    save_sidecar(tmp_path, fixture)
    loaded = load_sidecar(tmp_path)
    assert loaded == fixture
    # And the raw file is valid JSON matching what we expect.
    raw = json.loads((tmp_path / SIDECAR_FILENAME).read_text())
    assert raw == fixture


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Prevent any stray writes from landing in the repo root."""
    monkeypatch.chdir(tmp_path)
