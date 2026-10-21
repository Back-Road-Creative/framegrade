"""Unit tests for Phase D residual helpers extracted from the run body.

Helpers under test:

* ``_run_sidecar_integrity_gate``
* ``_build_and_persist_drive_schedule``
* ``_build_social_posts_per_lf``
* ``_run_final_validation_gate``
* ``_upload_all_lfs_shorts_to_drive``

These were the final inline blocks in ``PhaseD.run``. With them
extracted the run body stays under 300 lines.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    _build_and_persist_drive_schedule,
    _build_social_posts_per_lf,
    _run_final_validation_gate,
    _run_sidecar_integrity_gate,
    _upload_all_lfs_shorts_to_drive,
)

from _state_helpers import make_state


def _lf(idx, *, total_duration=1800.0):
    return SimpleNamespace(index=idx, total_duration=total_duration)


# ─── _run_sidecar_integrity_gate ────────────────────────────────────────


class TestRunSidecarIntegrityGate:
    def test_empty_drive_plan_is_noop(self, tmp_path):
        _run_sidecar_integrity_gate(
            state=make_state(),
            drive_plan=None,
            output_base=tmp_path,
            force_sidecar_reset=False,
        )

    def test_force_reset_wipes_sidecar_and_state(self, tmp_path):
        plan = SimpleNamespace(regular_lfs=[_lf(1)])
        state = make_state(youtube_uploads={1: {"video_id": "old"}})
        with (
            patch("gomoveshift.pipeline.youtube_sidecar.save_sidecar") as save_mock,
            patch("gomoveshift.pipeline.youtube_sidecar.assert_sidecar_integrity") as assert_mock,
        ):
            _run_sidecar_integrity_gate(
                state=state,
                drive_plan=plan,
                output_base=tmp_path,
                force_sidecar_reset=True,
            )
        save_mock.assert_called_once_with(tmp_path, {})
        assert state["youtube_uploads"] == {}
        assert_mock.assert_called_once()

    def test_assert_sidecar_integrity_called_when_not_reset(self, tmp_path):
        plan = SimpleNamespace(regular_lfs=[_lf(1)])
        with (
            patch("gomoveshift.pipeline.youtube_sidecar.save_sidecar") as save_mock,
            patch("gomoveshift.pipeline.youtube_sidecar.assert_sidecar_integrity") as assert_mock,
        ):
            _run_sidecar_integrity_gate(
                state=make_state(),
                drive_plan=plan,
                output_base=tmp_path,
                force_sidecar_reset=False,
            )
        save_mock.assert_not_called()
        assert_mock.assert_called_once_with(tmp_path, plan)


# ─── _build_and_persist_drive_schedule ──────────────────────────────────


class TestBuildAndPersistDriveSchedule:
    def _mk_scheduler_cls(self):
        scheduler = MagicMock()
        scheduler.generate_drive_schedule.return_value = SimpleNamespace(
            to_dict=lambda: {"slots": []}
        )
        scheduler_cls = MagicMock()
        scheduler_cls.from_config.return_value = scheduler
        return scheduler_cls, scheduler

    def test_persists_schedule_json(self, tmp_path):
        scheduler_cls, scheduler = self._mk_scheduler_cls()
        result = _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=self._cfg(["15:00", "18:00"], tmp_path),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=make_state(),
            region="usa",
            stages_dir=tmp_path,
        )
        assert (tmp_path / "drive_schedule.json").exists()
        assert json.loads((tmp_path / "drive_schedule.json").read_text()) == {"slots": []}
        scheduler.generate_drive_schedule.assert_called_once()
        assert result is scheduler.generate_drive_schedule.return_value

    @staticmethod
    def _cfg(times, tmp_path):
        """Minimal config stub exposing drive.shorts_publish_times plus a
        real seo.storage_path — the ledger read in
        ``_build_and_persist_drive_schedule`` now raises rather than
        degrading on a missing ``config.seo``, so the stub must carry one."""
        return SimpleNamespace(
            drive=SimpleNamespace(
                shorts_publish_times=times,
                long_form_publish_days=[1],
                long_form_publish_time="16:00",
            ),
            seo=SimpleNamespace(storage_path=str(tmp_path)),
        )

    def test_shorts_per_day_one_takes_first_configured_window(self, tmp_path):
        scheduler_cls, scheduler = self._mk_scheduler_cls()
        _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=self._cfg(["15:00", "18:00"], tmp_path),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=make_state(shorts_per_day=1),
            region="usa",
            stages_dir=tmp_path,
        )
        from datetime import time as dt_time

        assert scheduler.shorts_times == [dt_time(15, 0, 0)]

    def test_shorts_per_day_two_takes_both_configured_windows(self, tmp_path):
        scheduler_cls, scheduler = self._mk_scheduler_cls()
        _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=self._cfg(["15:00", "18:00"], tmp_path),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=make_state(shorts_per_day=2),
            region="usa",
            stages_dir=tmp_path,
        )
        from datetime import time as dt_time

        assert scheduler.shorts_times == [dt_time(15, 0, 0), dt_time(18, 0, 0)]

    def test_shorts_per_day_honors_custom_shorts_times(self, tmp_path):
        # An explicit --shorts-times (which lands in
        # config.drive.shorts_publish_times) must win — the old hardcoded
        # presets silently ignored it.
        scheduler_cls, scheduler = self._mk_scheduler_cls()
        _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=self._cfg(["16:00", "19:00"], tmp_path),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=make_state(shorts_per_day=2),
            region="usa",
            stages_dir=tmp_path,
        )
        from datetime import time as dt_time

        assert scheduler.shorts_times == [dt_time(16, 0, 0), dt_time(19, 0, 0)]

    def test_publish_day_and_start_date_threaded(self, tmp_path):
        scheduler_cls, scheduler = self._mk_scheduler_cls()
        _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=MagicMock(),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=make_state(start_date="2026-03-01", publish_day="wed"),
            region="ireland",
            stages_dir=tmp_path,
        )
        kwargs = scheduler.generate_drive_schedule.call_args.kwargs
        assert kwargs["start_date"] == "2026-03-01"
        assert kwargs["publish_day"] == "wed"
        assert kwargs["region"] == "ireland"


# ─── _build_social_posts_per_lf ─────────────────────────────────────────


class TestBuildSocialPostsPerLf:
    def _mk_social_gen_cls(self, posts_by_call=None):
        calls = []

        def gen_all(
            num_shorts,
            index_offset,
            global_indexes=None,
            per_short_locations=None,
            per_short_regions=None,
            per_short_roads=None,
            per_short_title_facts=None,
        ):
            call = SimpleNamespace(
                num=num_shorts, offset=index_offset, title_facts=per_short_title_facts
            )
            calls.append(call)
            n = num_shorts if posts_by_call is None else posts_by_call.pop(0)
            return [
                SimpleNamespace(short_index=index_offset + i, to_dict=lambda i=i: {"i": i})
                for i in range(n)
            ]

        gen_instance = MagicMock()
        gen_instance.generate_all.side_effect = gen_all
        cls = MagicMock(return_value=gen_instance)
        return cls, calls

    def test_zero_shorts_produces_zero_posts(self, tmp_path):
        cls, calls = self._mk_social_gen_cls()
        out = _build_social_posts_per_lf(
            social_gen_cls=cls,
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            shorts_per_lf={1: []},
            per_lf_locations_override={1: "x"},
            project_base="p",
            output_base=tmp_path,
        )
        assert out == {1: []}
        assert calls[0].num == 0

    def test_global_index_offsets_advance_across_lfs(self, tmp_path):
        cls, calls = self._mk_social_gen_cls()
        _build_social_posts_per_lf(
            social_gen_cls=cls,
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1), _lf(2)]),
            shorts_per_lf={1: ["a", "b"], 2: ["c", "d", "e"]},
            per_lf_locations_override={1: "x", 2: "x"},
            project_base="p",
            output_base=tmp_path,
        )
        assert calls[0].offset == 0
        assert calls[0].num == 2
        # LF 1 produced 2 posts → LF 2 starts at offset 2
        assert calls[1].offset == 2
        assert calls[1].num == 3

    def test_writes_social_posts_json_per_lf(self, tmp_path):
        cls, _ = self._mk_social_gen_cls()
        _build_social_posts_per_lf(
            social_gen_cls=cls,
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            shorts_per_lf={1: ["a"]},
            per_lf_locations_override={1: "x"},
            project_base="p",
            output_base=tmp_path,
        )
        posts_file = tmp_path / "shorts" / "lf_01" / "social_posts.json"
        assert posts_file.exists()

    def test_per_lf_location_override(self, tmp_path):
        cls, _ = self._mk_social_gen_cls()
        _build_social_posts_per_lf(
            social_gen_cls=cls,
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            shorts_per_lf={1: []},
            per_lf_locations_override={1: "Dingle, Kerry"},
            project_base="p",
            output_base=tmp_path,
        )
        cls.assert_called_with(
            location="Dingle, Kerry",
            project_name="p_lf01",
            series_part=1,
            titles_file=None,
            resolved_location=None,
        )

    def test_upstream_gate_names_verified_short_without_title_facts(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
            assert_all_places_verified,
            unverified_place_shorts,
        )

        day = tmp_path / "day_2026-09-14" / "shorts"
        day.mkdir(parents=True)
        (tmp_path / "shorts_ost_audit_2026-09-14.json").write_text(
            json.dumps(
                {
                    "shorts": [
                        {
                            "rank": 1,
                            "start_sec": 10.0,  # no end_sec: duration unreadable
                            "ost_text": "Wythe County, VA",
                            "regions": ["Wythe County"],
                            "place_source": "clip_gps",
                        }
                    ]
                }
            )
        )
        shorts = {1: [day / "short_01.mp4"]}
        problems = unverified_place_shorts(shorts)
        assert len(problems) == 1 and "short_01.mp4" in problems[0]
        assert "no title facts" in problems[0]
        with pytest.raises(RuntimeError, match=r"(?s)short_01\.mp4.*no title facts"):
            assert_all_places_verified(shorts, enforce=True)

    def test_build_social_posts_passes_title_facts_from_sidecar(self, tmp_path):
        stages = tmp_path / "stages"
        day = stages / "day_2026-09-14" / "shorts"
        day.mkdir(parents=True)
        rows = [
            {
                "rank": 5,
                "start_sec": 300.0,
                "end_sec": 341.4,
                "regions": ["Wythe County"],
                "place_source": "clip_gps",
                "ost_text": "Wythe County, VA",
            },
            {
                "rank": 2,
                "start_sec": 100.0,
                "end_sec": 130.0,
                "regions": [],
                "place_source": "lf_label",
                "ost_text": "X, VA",
            },
            {
                "rank": 9,
                "start_sec": 900.0,
                "end_sec": 930.0,
                "regions": ["Y"],
                "place_source": "clip_gps",
                "ost_text": "Y, VA",
            },
        ]
        (stages / "shorts_ost_audit_2026-09-14.json").write_text(json.dumps({"shorts": rows}))
        shorts = [day / "short_05.mp4", day / "short_02.mp4", tmp_path / "odd" / "short_01.mp4"]
        cls, calls = self._mk_social_gen_cls()
        _build_social_posts_per_lf(
            social_gen_cls=cls,
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            shorts_per_lf={1: shorts},
            per_lf_locations_override={1: "x"},
            project_base="p",
            output_base=tmp_path,
        )
        five, two, odd = calls[0].title_facts
        assert (five.date, five.ordinal, five.secs, five.place_ok) == ("2026-09-14", 2, 41, True)
        assert (two.date, two.ordinal, two.secs, two.place_ok) == ("2026-09-14", 1, 30, False)
        assert odd is None


# ─── _run_final_validation_gate ─────────────────────────────────────────


class TestRunFinalValidationGate:
    def _mk_config(self, *, skip_audio=False):
        return SimpleNamespace(
            drive=SimpleNamespace(skip_audio=skip_audio),
            output=SimpleNamespace(audio_max_lufs=-18.0, audio_target_tp=-1.5),
        )

    def test_missing_file_hard_fails(self, tmp_path):
        # A missing FINAL.mp4 is a file_integrity HARD FAIL upstream
        # of any upload (was silently skipped). validate is never called on an
        # absent file — the failure is recorded directly.
        validator = MagicMock()
        validator_cls = MagicMock(return_value=validator)
        with pytest.raises(RuntimeError, match="File integrity HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=validator_cls,
                config=self._mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[tmp_path / "missing.mp4"],
                metadata_per_lf={},
            )
        validator.validate.assert_not_called()

    def test_small_file_hard_fails(self, tmp_path):
        # An under-1-MiB (truncated) FINAL.mp4 is a file_integrity
        # HARD FAIL upstream of any upload (was silently skipped).
        validator = MagicMock()
        validator_cls = MagicMock(return_value=validator)
        small = tmp_path / "lf.mp4"
        small.write_bytes(b"x" * 100)  # under 1 MiB threshold
        with pytest.raises(RuntimeError, match="File integrity HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=validator_cls,
                config=self._mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[small],
                metadata_per_lf={},
            )
        validator.validate.assert_not_called()

    def test_audio_failure_raises_hard_fail(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        audio_gate = SimpleNamespace(gate="audio_loudness", passed=False, message="too quiet")
        result = SimpleNamespace(passed=False, gates=[audio_gate], summary=lambda: "audio fail")
        validator = MagicMock()
        validator.validate.return_value = result
        with pytest.raises(RuntimeError, match="HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=MagicMock(return_value=validator),
                config=self._mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {}},
            )

    def test_skip_audio_bypasses_hard_fail(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        audio_gate = SimpleNamespace(gate="audio_loudness", passed=False, message="too quiet")
        result = SimpleNamespace(passed=False, gates=[audio_gate], summary=lambda: "x")
        validator = MagicMock()
        validator.validate.return_value = result
        # No raise — skip_audio=True means audio sub-gate is bypassed.
        _run_final_validation_gate(
            validator_cls=MagicMock(return_value=validator),
            config=self._mk_config(skip_audio=True),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            regular_lf_paths=[big],
            metadata_per_lf={1: {}},
        )


# ─── _upload_all_lfs_shorts_to_drive ────────────────────────────────────


class TestUploadAllLfsShortsToDrive:
    def test_skip_flag_returns_empty_uploader(self, tmp_path):
        manifests, uploader = _upload_all_lfs_shorts_to_drive(
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=make_state(),
            project_base="p",
            output_base=tmp_path,
            uploader_factory=MagicMock(),
            skip_drive_upload=True,
        )
        assert manifests == {}
        assert uploader is None

    def test_complete_prior_manifest_skips_reupload(self, tmp_path):
        prior = SimpleNamespace(files={"short_01.mp4": "url"}, completed=True)
        state = make_state(
            upload_manifests={1: prior},
            shorts_per_lf={1: ["/x/short_01.mp4"]},
        )
        factory = MagicMock()
        manifests, uploader = _upload_all_lfs_shorts_to_drive(
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            state=state,
            project_base="p",
            output_base=tmp_path,
            uploader_factory=factory,
            skip_drive_upload=False,
        )
        # Factory still called to mint the uploader (the loop is entered);
        # but no upload happens because manifest matches expected count.
        assert manifests[1] is prior
        factory.return_value.upload_shorts.assert_not_called()

    def test_partial_manifest_triggers_reupload(self, tmp_path):
        prior = SimpleNamespace(files={"short_01.mp4": "url1"}, completed=False)
        state = make_state(
            upload_manifests={1: prior},
            shorts_per_lf={1: ["/x/short_01.mp4", "/x/short_02.mp4"]},
        )
        new_uploader = MagicMock()
        new_uploader.upload_shorts.return_value = [
            "https://drive/short_01",
            "https://drive/short_02",
        ]
        with patch(
            "gomoveshift.pipeline.drive_phases.external_systems_coordinator._resolve_drive_subfolder_id",
            return_value="folder",
        ):
            manifests, _ = _upload_all_lfs_shorts_to_drive(
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                state=state,
                project_base="p",
                output_base=tmp_path,
                uploader_factory=lambda: new_uploader,
                skip_drive_upload=False,
            )
        # Re-upload happened → new manifest with both files
        assert len(manifests[1].files) == 2

    def test_per_lf_failure_propagates_immediately(self, tmp_path, caplog):
        """Per-LF failure aborts the loop rather than being swallowed.

        A broad ``except Exception`` that logged WARN and continued would
        ship a Metricool CSV with empty ``Picture Url 1`` cells. Instead the loop
        propagates the exception with an LF-index ERROR log.
        """
        import pytest

        state = make_state(shorts_per_lf={1: ["/x/a.mp4"], 2: ["/x/b.mp4"]})
        uploader = MagicMock()
        uploader.upload_shorts.side_effect = [
            RuntimeError("first lf boom"),
            ["https://drive/b"],
        ]
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.external_systems_coordinator._resolve_drive_subfolder_id",
                return_value="folder",
            ),
            caplog.at_level("ERROR"),
        ):
            with pytest.raises(RuntimeError, match="first lf boom"):
                _upload_all_lfs_shorts_to_drive(
                    drive_plan=SimpleNamespace(regular_lfs=[_lf(1), _lf(2)]),
                    state=state,
                    project_base="p",
                    output_base=tmp_path,
                    uploader_factory=lambda: uploader,
                    skip_drive_upload=False,
                )
        # Loop aborted on first failure; second LF never attempted.
        assert uploader.upload_shorts.call_count == 1
        assert any("Drive upload failed for LF 1" in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]
