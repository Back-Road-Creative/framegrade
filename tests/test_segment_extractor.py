"""Behaviour-parity pins for the keycut rewire.

The lossless-cutting logic that used to live in
``gomoveshift/pipeline/segment_extractor.py`` now comes from the published
``keycut`` package. keycut is a PORT, not a rename: five of its defaults
differ from what this pipeline shipped. Each one is pinned here, because an
unpinned delta is a silent production change in a pipeline that runs
unattended.

  1. Re-encode VIDEO args — keycut derives an encoder from the master's own
     codec at CRF 18; this pipeline pins the shared long-form encoder args.
  2. Re-encode AUDIO — keycut stream-copies; this pipeline pins AAC 256k.
  3. ``remap_segment_indices`` returns a NEW list instead of mutating in
     place. No production caller ever used the mutating helper — the
     authoritative remap is ``gps_enrichment._partition_lf_plans_by_date``,
     which does its own in-place rewrite. Pinned so that stays true.
  4. Merge threshold — keycut MEASURES the master's real GOP; this pipeline
     pins the fixed 2.1 s constant.
  5. Renames — ``stop_mask`` → ``exclude_mask``, and
     ``assign_shorts_to_lfs_by_source_window`` → ``assign_clips_by_source_window``
     over a plain ``Mapping[key, Sequence[Segment]]``.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import day_pipeline, montage_builder


def _seg(ep_idx, start, end):
    return SimpleNamespace(
        source_file_index=ep_idx,
        start_time=float(start),
        end_time=float(end),
    )


def _short(start, end):
    return SimpleNamespace(start_sec=float(start), end_sec=float(end))


def _batch(durations):
    return SimpleNamespace(
        date="2026-02-20",
        episodes=[SimpleNamespace(total_duration=float(d)) for d in durations],
    )


# ── boundaries + segment adaptation ─────────────────────────────────────


class TestBoundaries:
    def test_boundaries_start_at_zero_and_accumulate(self):
        bounds = day_pipeline.compute_day_file_boundaries(_batch([1200.0, 900.0, 600.0]))
        assert bounds == [(0.0, 1200.0), (1200.0, 2100.0), (2100.0, 2700.0)]

    def test_empty_batch_gives_no_boundaries(self):
        assert day_pipeline.compute_day_file_boundaries(SimpleNamespace(episodes=[])) == []

    def test_adapter_maps_gms_field_names_onto_keycut_segment(self):
        cut = montage_builder.to_cut_segment(_seg(2, 10.0, 20.0))
        assert (cut.source_index, cut.start, cut.end) == (2, 10.0, 20.0)


# ── delta 4: merge threshold pinned, not measured ───────────────────────


class TestMergeThresholdPinned:
    def test_constant_is_one_gop_plus_margin(self):
        """A 60-frame GOP at 29.97 fps is ~2.002 s; the constant adds margin."""
        assert montage_builder.GOP_DURATION_SEC == 2.1

    def test_extract_passes_the_fixed_gap_instead_of_measuring(self, tmp_path):
        """keycut defaults ``max_gap_sec=None``, which measures the master's
        real GOP. On an irregular master that changes which ranges merge."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")

        with patch.object(day_pipeline, "extract_and_concat_ranges") as ranges_fn:
            ranges_fn.return_value = tmp_path / "out.mp4"
            day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[_seg(0, 10.0, 20.0)],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
            )

        assert ranges_fn.call_args.kwargs["max_gap_sec"] == 2.1


# ── deltas 1 + 2: re-encode video + audio args pinned ───────────────────


class TestReencodeArgsPinned:
    def test_video_args_are_the_shared_longform_encoder(self):
        """keycut's default is ``-c:v <master codec> -crf 18``. This pipeline
        pins ``get_encoding_args(3840, 2160, 29.97, has_nvenc=False)`` — libx265
        slow CRF 18 with the 4K bitrate floor and BT.709 colour metadata."""
        from gomoveshift.pipeline.color import get_encoding_args

        expected = get_encoding_args(3840, 2160, 29.97, has_nvenc=False)
        args = montage_builder.lf_reencode_args()
        assert args[: len(expected)] == expected
        assert "libx265" in args
        # Never NVENC: this join-repair path must stay GPU-independent.
        assert not any("nvenc" in a for a in args)

    def test_deliberate_cpu_path_does_not_warn_nvenc_unavailable(self, caplog, monkeypatch):
        """The CPU encode here is a choice, not a missing GPU. The shared
        "NVENC unavailable" WARNING told operators their GPU was down on every
        drive run while NVENC was working fine."""
        import logging

        monkeypatch.setattr(montage_builder, "_LF_REENCODE_ARGS", None)
        with caplog.at_level(logging.WARNING):
            montage_builder.lf_reencode_args()
        assert not [r for r in caplog.records if "NVENC" in r.getMessage()]

    def test_real_fallback_still_warns(self, caplog):
        """A caller that wanted NVENC and lost it must still hear about it."""
        import logging

        from gomoveshift.pipeline.color import get_encoding_args

        with caplog.at_level(logging.WARNING):
            get_encoding_args(1920, 1080, 29.97, has_nvenc=False)
        assert any("NVENC" in r.getMessage() for r in caplog.records)

    def test_audio_is_reencoded_aac_256k_not_stream_copied(self):
        """keycut's default is ``-c:a copy``. This pipeline has always emitted
        AAC 256k here; copying would change the shipped artefact."""
        args = montage_builder.lf_reencode_args()
        assert args[-4:] == ["-c:a", "aac", "-b:a", "256k"]

    def test_extract_passes_reencode_args_explicitly(self, tmp_path):
        """Left to default, keycut would probe the master and re-encode at
        CRF 18 — different settings than this pipeline has been shipping."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")

        with patch.object(day_pipeline, "extract_and_concat_ranges") as ranges_fn:
            ranges_fn.return_value = tmp_path / "out.mp4"
            day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[_seg(0, 10.0, 20.0)],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
            )

        passed = ranges_fn.call_args.kwargs["reencode_args"]
        assert passed is not None, "must not fall through to keycut's own default"
        assert passed == montage_builder.lf_reencode_args()

    def test_montage_reencode_uses_the_same_args(self, tmp_path):
        """The montage's probe-unavailable fallback must not drift from the LF
        body's — the two pieces are joined by a stream copy."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")

        with (
            patch.object(montage_builder, "probe_keyframes", return_value=None),
            patch.object(montage_builder, "reencode_range", return_value=True) as reenc,
        ):
            ok = montage_builder._trim_peak_to(
                master, _seg(0, 10.0, 20.0), [(0.0, 60.0)], tmp_path / "clip.mp4", 2.0
            )

        assert ok
        assert reenc.call_args.kwargs["encode_args"] == montage_builder.lf_reencode_args()


# ── delta 5: stop_mask → exclude_mask ───────────────────────────────────


class TestStopMaskRenamedToExcludeMask:
    def test_stop_mask_reaches_keycut_as_exclude_mask(self, tmp_path):
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        mask = [False] * 30 + [True] * 5

        with patch.object(day_pipeline, "extract_and_concat_ranges") as ranges_fn:
            ranges_fn.return_value = tmp_path / "out.mp4"
            day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[_seg(0, 10.0, 20.0)],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
                stop_mask=mask,
            )

        assert ranges_fn.call_args.kwargs["exclude_mask"] is mask


# ── delta 3: remap returns a new list ───────────────────────────────────


class TestRemapNoLongerMutates:
    def test_returns_new_list_and_leaves_input_untouched(self):
        from keycut import Segment, remap_segment_indices

        original = [Segment(5, 10.0, 20.0)]
        out = remap_segment_indices(original, [5])
        assert out == [Segment(0, 10.0, 20.0)]
        assert original == [Segment(5, 10.0, 20.0)], "input must not be mutated"

    def test_index_outside_this_master_raises(self):
        from keycut import Segment, remap_segment_indices

        with pytest.raises(IndexError, match="not among this master's parts"):
            remap_segment_indices([Segment(9, 0.0, 1.0)], [5])

    def test_no_production_caller_relies_on_the_in_place_remap(self):
        """The mutating ``remap_segments_drive_to_day_local`` had NO production
        caller: the authoritative remap is
        ``gps_enrichment._partition_lf_plans_by_date``, which rewrites
        ``source_file_index`` itself. Pinned so nobody adds a caller that would
        silently receive un-remapped segments from keycut's replacement."""
        pkg = Path(day_pipeline.__file__).resolve().parent.parent
        offenders = [
            str(p.relative_to(pkg))
            for p in sorted(pkg.rglob("*.py"))
            if "remap_segment_indices(" in p.read_text()
            or "remap_segments_drive_to_day_local" in p.read_text()
        ]
        assert offenders == [], (
            "keycut's remap returns a NEW list; a caller that ignores the return "
            f"value silently keeps drive-global indices. Offenders: {offenders}"
        )


# ── shorts attribution across the assign_clips rename ───────────────────


class TestShortsAssignment:
    def test_containment_wins_over_proximity(self):
        boundaries = day_pipeline.compute_day_file_boundaries(_batch([600.0, 900.0]))
        lf1 = SimpleNamespace(index=1, segments=[_seg(0, 0.0, 300.0)])
        lf2 = SimpleNamespace(index=2, segments=[_seg(1, 0.0, 300.0)])

        out = day_pipeline.assign_shorts_to_lfs_by_source_window(
            short_ranges=[_short(10, 40), _short(610, 640)],
            lf_plans=[lf1, lf2],
            file_boundaries=boundaries,
        )
        assert out == {1: [0], 2: [1]}

    def test_orphan_short_lands_on_nearest_lf_never_dropped(self):
        boundaries = day_pipeline.compute_day_file_boundaries(_batch([1800.0]))
        lf1 = SimpleNamespace(index=1, segments=[_seg(0, 0.0, 300.0)])
        lf2 = SimpleNamespace(index=2, segments=[_seg(0, 1200.0, 1500.0)])

        out = day_pipeline.assign_shorts_to_lfs_by_source_window(
            short_ranges=[_short(320, 350), _short(1150, 1180)],
            lf_plans=[lf1, lf2],
            file_boundaries=boundaries,
        )
        assert out == {1: [0], 2: [1]}

    def test_no_lf_plans_returns_empty(self):
        out = day_pipeline.assign_shorts_to_lfs_by_source_window(
            short_ranges=[_short(0, 30)], lf_plans=[], file_boundaries=[(0.0, 1000.0)]
        )
        assert out == {}

    def test_every_lf_keeps_a_bucket_with_no_shorts(self):
        boundaries = day_pipeline.compute_day_file_boundaries(_batch([500.0]))
        lf1 = SimpleNamespace(index=1, segments=[_seg(0, 0.0, 100.0)])
        lf2 = SimpleNamespace(index=2, segments=[_seg(0, 200.0, 300.0)])

        out = day_pipeline.assign_shorts_to_lfs_by_source_window(
            short_ranges=[], lf_plans=[lf1, lf2], file_boundaries=boundaries
        )
        assert out == {1: [], 2: []}


# ── degradation contract preserved across the move ──────────────────────


class TestDegradationStillReturnsNone:
    def test_missing_master(self, tmp_path):
        assert (
            day_pipeline.extract_and_concat_segments(
                master_path=tmp_path / "nope.mp4",
                segments=[MagicMock()],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
            )
            is None
        )

    def test_no_segments(self, tmp_path):
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        assert (
            day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
            )
            is None
        )

    def test_source_index_outside_boundaries(self, tmp_path):
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        assert (
            day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[_seg(5, 0.0, 10.0)],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
            )
            is None
        )


# ── faststart on an intermediate is wasted work that can time out ───────────


class TestFaststartDisabledOnIntermediates:
    def test_extract_disables_faststart(self, tmp_path):
        """``*_segments.mp4`` is an INTERMEDIATE — apply_ost re-encodes it into
        the final artifact, which writes its own faststart, and cleanup treats
        the concat as disposable. keycut defaults ``faststart=True``, which
        forces a full moov rewrite over the whole concat: on a superlong part
        that is the entire ~118 GB day, and it blew keycut's 1800s stream-copy
        timeout (observed on a superlong part — the concat is not given a timeout
        argument by this caller, so the only lever is not doing the work).
        """
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")

        with patch.object(day_pipeline, "extract_and_concat_ranges") as ranges_fn:
            ranges_fn.return_value = tmp_path / "out.mp4"
            day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[_seg(0, 10.0, 20.0)],
                file_boundaries=[(0.0, 60.0)],
                output_path=tmp_path / "out.mp4",
                work_dir=tmp_path / "work",
            )

        assert ranges_fn.call_args.kwargs["faststart"] is False
