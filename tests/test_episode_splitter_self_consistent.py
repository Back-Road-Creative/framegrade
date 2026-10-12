"""Tests for self-consistent sub-Episode construction after GPS splitting.

Sub-episodes produced by a GPS split (for example a 220-minute episode split
into 4) must carry their own recordings. With recordings=[] total_duration is 0
and the following ffmpeg concat step fails.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.ingest import (
    Episode,
    GoProChapter,
    GoProRecording,
    IngestPlan,
    MAX_EPISODE_DURATION,
)
from gomoveshift.pipeline.ingest.concatenator import (
    _split_episode_at_points,
    concat_episode,
    extract_and_merge_gpx,
)


# The adoption gate now parses the pre-set GPX and requires ≥1
# trackpoint (bare `<gpx/>` no longer passes — that's what a crash-truncated
# write looks like), so fixtures standing in for a legitimate pre-set GPX
# need a real trackpoint.
_VALID_GPX = (
    '<?xml version="1.0"?><gpx version="1.1" creator="test">'
    '<trk><trkseg><trkpt lat="45.0" lon="-63.0">'
    "<time>2026-03-04T08:00:00Z</time></trkpt></trkseg></trk></gpx>"
)


def _make_recording(rec_num: int, duration_sec: float, ct: datetime) -> GoProRecording:
    rec = GoProRecording(recording_num=rec_num)
    rec.chapters = [
        GoProChapter(
            path=Path(f"/fake/CAM01{rec_num:04d}.MP4"),
            chapter_num=1,
            recording_num=rec_num,
            duration=duration_sec,
            creation_time=ct,
        )
    ]
    return rec


class TestEpisodeExplicitFields:
    """Episode._explicit_* overrides take priority over recording-derived values."""

    def test_explicit_total_duration_returned(self):
        ep = Episode(
            index=401,
            recordings=[],
            _explicit_total_duration=2880.0,
        )
        assert ep.total_duration == 2880.0

    def test_explicit_total_duration_overrides_recordings(self):
        ct = datetime(2026, 3, 4, 8, 0)
        rec = _make_recording(61, 13200.0, ct)  # 220min
        ep = Episode(
            index=401,
            recordings=[rec],
            _explicit_total_duration=2880.0,  # 48min override
        )
        assert ep.total_duration == 2880.0

    def test_explicit_creation_time_returned(self):
        ct = datetime(2026, 3, 4, 10, 30)
        ep = Episode(index=401, recordings=[], _explicit_creation_time=ct)
        assert ep.creation_time == ct

    def test_explicit_creation_time_overrides_recordings(self):
        original_ct = datetime(2026, 3, 4, 8, 0)
        override_ct = datetime(2026, 3, 4, 10, 30)
        rec = _make_recording(61, 1800.0, original_ct)
        ep = Episode(index=401, recordings=[rec], _explicit_creation_time=override_ct)
        assert ep.creation_time == override_ct

    def test_chapter_derived_when_no_explicit(self):
        ct = datetime(2026, 3, 4, 8, 0)
        rec = _make_recording(61, 3600.0, ct)
        ep = Episode(index=1, recordings=[rec])
        assert ep.total_duration == 3600.0
        assert ep.creation_time == ct

    def test_none_explicit_does_not_override(self):
        ct = datetime(2026, 3, 4, 8, 0)
        rec = _make_recording(61, 1800.0, ct)
        ep = Episode(index=1, recordings=[rec], _explicit_total_duration=None)
        assert ep.total_duration == 1800.0

    def test_total_duration_min_uses_explicit(self):
        ep = Episode(index=401, recordings=[], _explicit_total_duration=2880.0)
        assert abs(ep.total_duration_min - 48.0) < 0.01


class TestConcatEpisodePresetPath:
    """concat_episode returns early when concat_path is already set and on disk."""

    def test_returns_early_when_preset(self, tmp_path):
        slice_file = tmp_path / "episode_04_a.mp4"
        slice_file.write_bytes(b"\x00" * 512)

        ep = Episode(
            index=401,
            recordings=[],
            date="2026-03-04",
            concat_path=slice_file,
            _explicit_total_duration=2880.0,
        )

        with patch("subprocess.run") as mock_run:
            result = concat_episode(ep, tmp_path)

        assert result == slice_file
        mock_run.assert_not_called()

    def test_proceeds_when_concat_path_none(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        rec = _make_recording(61, 1800.0, ct)
        ep = Episode(index=1, recordings=[rec], date="2026-03-04", concat_path=None)

        chapter_file = tmp_path / "CAM010061.MP4"
        chapter_file.write_bytes(b"\x00" * 100)
        rec.chapters[0] = GoProChapter(chapter_file, 1, 61, 1800.0, ct)

        output_file = tmp_path / "episode_01_2026-03-04.mp4"
        output_file.write_bytes(b"\x00" * 256)

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.get_clip_duration", return_value=1800.0
        ):
            result = concat_episode(ep, tmp_path)
        assert result == output_file

    def test_returns_early_when_file_is_zero_bytes(self, tmp_path):
        """Zero-byte preset must NOT trigger early-return — ffmpeg wrote nothing."""
        slice_file = tmp_path / "episode_04_a.mp4"
        slice_file.write_bytes(b"")  # 0 bytes

        ep = Episode(
            index=401,
            recordings=[],
            date="2026-03-04",
            concat_path=slice_file,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="no input")
            with pytest.raises(Exception):
                concat_episode(ep, tmp_path)

        mock_run.assert_called()


class TestExtractGpxPresetPath:
    """extract_and_merge_gpx returns early when gpx_path is already set and on disk."""

    def test_returns_early_when_preset(self, tmp_path):
        gpx_file = tmp_path / "episode_04_a.gpx"
        gpx_file.write_text(_VALID_GPX, encoding="utf-8")

        ep = Episode(
            index=401,
            recordings=[],
            date="2026-03-04",
            gpx_path=gpx_file,
            _explicit_total_duration=2880.0,
        )

        # Early-return fires before the lazy `from ..gopro_gps import detect_gpmf_stream`
        # inside the function body — patch not required, just verify the result.
        result = extract_and_merge_gpx(ep, tmp_path)

        assert result == gpx_file

    def test_proceeds_when_gpx_path_none(self, tmp_path):
        # Episode has no recordings → for loop is a no-op → returns None.
        # detect_gpmf_stream is never reached, so no patching needed.
        ep = Episode(index=1, recordings=[], date="2026-03-04", gpx_path=None)

        result = extract_and_merge_gpx(ep, tmp_path)

        assert result is None


class TestSplitEpisodeAtPoints:
    """_split_episode_at_points produces self-consistent sub-Episodes."""

    def _make_episode_with_concat(
        self, tmp_path: Path, duration_sec: float, ct: datetime
    ) -> Episode:
        """Build a chapter-derived episode with a real concat file on disk."""
        rec = _make_recording(61, duration_sec, ct)
        concat_file = tmp_path / "episode_04_2026-03-04.mp4"
        concat_file.write_bytes(b"\x00" * 1024)

        ep = Episode(
            index=4,
            recordings=[rec],
            date="2026-03-04",
            concat_path=concat_file,
        )
        return ep

    def _mock_ffmpeg_success(self, tmp_path: Path):
        """Return a mock subprocess.run that creates each output file."""

        def side_effect(cmd, **kwargs):
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        return side_effect

    def test_all_sub_eps_have_empty_recordings(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)  # 220min

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=self._mock_ffmpeg_success(tmp_path),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0, 5760.0, 8640.0],  # 48/96/144 min
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert len(sub_eps) == 4
        for sub_ep in sub_eps:
            assert sub_ep.recordings == [], f"sub-ep {sub_ep.index} has non-empty recordings"

    def test_sub_ep_total_duration_matches_slice_extent(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=self._mock_ffmpeg_success(tmp_path),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0, 5760.0, 8640.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert len(sub_eps) == 4
        # Slices: 0→48min, 48→96min, 96→144min, 144→220min
        expected = [2880.0, 2880.0, 2880.0, 4560.0]
        for sub_ep, exp in zip(sub_eps, expected):
            assert abs(sub_ep.total_duration - exp) < 0.01, (
                f"sub-ep {sub_ep.index}: expected {exp}s, got {sub_ep.total_duration}s"
            )

    def test_sub_ep_creation_time_offset_from_parent(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=self._mock_ffmpeg_success(tmp_path),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0, 5760.0, 8640.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        # Sub-ep[0] starts at t=0 → same as parent
        assert sub_eps[0].creation_time == ct
        # Sub-ep[1] starts at 48min offset
        assert sub_eps[1].creation_time == ct + timedelta(seconds=2880.0)
        # Sub-ep[2] starts at 96min offset
        assert sub_eps[2].creation_time == ct + timedelta(seconds=5760.0)
        # Sub-ep[3] starts at 144min offset
        assert sub_eps[3].creation_time == ct + timedelta(seconds=8640.0)

    def test_sub_ep_concat_path_is_slice_file(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=self._mock_ffmpeg_success(tmp_path),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0, 5760.0, 8640.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert sub_eps[0].concat_path == tmp_path / "episode_04_a.mp4"
        assert sub_eps[1].concat_path == tmp_path / "episode_04_b.mp4"
        assert sub_eps[2].concat_path == tmp_path / "episode_04_c.mp4"
        assert sub_eps[3].concat_path == tmp_path / "episode_04_d.mp4"

    def test_sub_ep_gpx_path_set_when_parent_has_gpx(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)
        parent_gpx = tmp_path / "episode_04.gpx"
        parent_gpx.write_bytes(b"<gpx/>")
        ep.gpx_path = parent_gpx

        sliced_gpx = tmp_path / "_stages" / "episode_04_a.gpx"
        sliced_gpx.parent.mkdir(parents=True, exist_ok=True)
        sliced_gpx.write_bytes(b"<gpx/>")

        with (
            patch(
                "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
                side_effect=self._mock_ffmpeg_success(tmp_path),
            ),
            patch(
                "gomoveshift.pipeline.ingest.concatenator._extract_gpx_time_range",
                return_value=sliced_gpx,
            ),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert len(sub_eps) == 2
        assert sub_eps[0].gpx_path == sliced_gpx

    def test_sub_ep_gpx_path_none_when_parent_lacks_gpx(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)
        ep.gpx_path = None

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=self._mock_ffmpeg_success(tmp_path),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0, 5760.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        for sub_ep in sub_eps:
            assert sub_ep.gpx_path is None

    def test_creation_time_none_when_parent_has_none(self, tmp_path):
        ep = Episode(
            index=4,
            recordings=[],
            date="2026-03-04",
            _explicit_total_duration=13200.0,
        )
        concat_file = tmp_path / "episode_04_2026-03-04.mp4"
        concat_file.write_bytes(b"\x00" * 1024)
        ep.concat_path = concat_file

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=self._mock_ffmpeg_success(tmp_path),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert len(sub_eps) == 2
        for sub_ep in sub_eps:
            assert sub_ep.creation_time is None

    def test_ffmpeg_failure_returns_empty(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        ep = self._make_episode_with_concat(tmp_path, 13200.0, ct)

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            return_value=MagicMock(returncode=1, stderr="error"),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=[2880.0],
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert sub_eps == []


class TestSplitPointsLandOnKeyframes:
    """``-ss <t> -i src … -c copy`` cannot start on a
    non-keyframe, so ffmpeg rewinds to the previous one. Each slice is cut by its
    own independent ffmpeg call, so slice N stopped at the requested point while
    slice N+1 began up to one GOP EARLIER — after the day concat that footage
    plays twice, at every splice. Split boundaries must sit on real keyframes.
    """

    # ~2.002 s GOP, the shape a 29.97 fps GoPro/HEVC source actually has.
    KEYFRAMES = [round(i * 2.002, 3) for i in range(7000)]

    def _run_split(self, tmp_path, split_points, keyframes):
        ct = datetime(2026, 3, 4, 8, 0)
        rec = _make_recording(61, 13200.0, ct)
        concat_file = tmp_path / "episode_04_2026-03-04.mp4"
        concat_file.write_bytes(b"\x00" * 1024)
        ep = Episode(index=4, recordings=[rec], date="2026-03-04", concat_path=concat_file)

        cmds: list[list[str]] = []

        def side_effect(cmd, **kwargs):
            cmds.append(list(cmd))
            Path(cmd[-1]).write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
                side_effect=side_effect,
            ),
            patch(
                "gomoveshift.pipeline.ingest.concatenator.probe_keyframes",
                return_value=keyframes,
            ),
        ):
            sub_eps = _split_episode_at_points(
                ep,
                split_points=split_points,
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )
        starts = [float(c[c.index("-ss") + 1]) for c in cmds]
        durations = [float(c[c.index("-t") + 1]) for c in cmds]
        return sub_eps, starts, durations

    def test_every_slice_starts_on_a_keyframe(self, tmp_path):
        # 2880.0 / 5760.0 / 8640.0 all fall strictly between 2.002 s keyframes.
        _, starts, _ = self._run_split(tmp_path, [2880.0, 5760.0, 8640.0], self.KEYFRAMES)
        assert len(starts) == 4
        on_grid = set(self.KEYFRAMES)
        for s in starts:
            assert s in on_grid, f"-ss {s} is not a source keyframe — the copy rewinds"

    def test_consecutive_slices_tile_without_overlap(self, tmp_path):
        sub_eps, starts, durations = self._run_split(
            tmp_path, [2880.0, 5760.0, 8640.0], self.KEYFRAMES
        )
        for i in range(len(starts) - 1):
            assert starts[i] + durations[i] == pytest.approx(starts[i + 1], abs=1e-6), (
                f"slice {i} ends at {starts[i] + durations[i]} but slice {i + 1} "
                f"starts at {starts[i + 1]} — the day concat replays that footage"
            )
        assert starts[-1] + durations[-1] == pytest.approx(13200.0, abs=1e-6)
        assert sum(e.total_duration for e in sub_eps) == pytest.approx(13200.0, abs=1e-6)

    def test_splits_snapping_onto_one_keyframe_do_not_emit_an_empty_slice(self, tmp_path):
        """Two split points inside the same GOP land on the same keyframe."""
        sub_eps, starts, durations = self._run_split(tmp_path, [2880.0, 2880.5], self.KEYFRAMES)
        assert len(starts) == len(set(starts)) == len(sub_eps)
        assert all(d > 0 for d in durations)

    def test_unprobeable_source_still_splits(self, tmp_path):
        """``probe_keyframes`` returning None means 'unknown', not 'no keyframes'
        — a loose cut beats refusing to split a multi-hour episode."""
        sub_eps, starts, _ = self._run_split(tmp_path, [2880.0, 5760.0], None)
        assert len(sub_eps) == 3
        assert starts == [0.0, 2880.0, 5760.0]


class TestLongEpisodeEndToEndShape:
    """End-to-end check of a 220-minute episode shape.

    220min episode splits at 48/96/144min → 4 sub-episodes. Each sub-ep
    must survive a follow-up concat_episode call (the idempotency gate)
    without error, since that's the path execute_ingest takes.
    """

    def test_long_episode_four_sub_eps_survive_concat(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        parent_duration = 13200.0  # 220 min

        rec = _make_recording(61, parent_duration, ct)
        concat_file = tmp_path / "episode_04_2026-03-04.mp4"
        concat_file.write_bytes(b"\x00" * 1024)

        parent_ep = Episode(
            index=4,
            recordings=[rec],
            date="2026-03-04",
            concat_path=concat_file,
        )

        def ffmpeg_side_effect(cmd, **kwargs):
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=ffmpeg_side_effect,
        ):
            sub_eps = _split_episode_at_points(
                parent_ep,
                split_points=[2880.0, 5760.0, 8640.0],  # 48/96/144 min
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        assert len(sub_eps) == 4, f"expected 4 sub-eps, got {len(sub_eps)}"

        # Each sub-ep must survive concat_episode without error
        for sub_ep in sub_eps:
            assert sub_ep.recordings == []
            assert sub_ep.concat_path is not None
            assert sub_ep.concat_path.exists()
            assert sub_ep.total_duration > 0

            # concat_episode must return early (idempotency) — no ffmpeg invocation
            with patch("gomoveshift.pipeline.ingest.concatenator.subprocess.run") as mock_run:
                result = concat_episode(sub_ep, tmp_path)
                (
                    mock_run.assert_not_called(),
                    (
                        f"concat_episode called ffmpeg on sub-ep {sub_ep.index} — "
                        "idempotency guard failed"
                    ),
                )
            assert result == sub_ep.concat_path

    def test_long_episode_sub_ep_durations_sum_to_parent(self, tmp_path):
        ct = datetime(2026, 3, 4, 8, 0)
        parent_duration = 13200.0

        rec = _make_recording(61, parent_duration, ct)
        concat_file = tmp_path / "episode_04_2026-03-04.mp4"
        concat_file.write_bytes(b"\x00" * 1024)

        parent_ep = Episode(
            index=4,
            recordings=[rec],
            date="2026-03-04",
            concat_path=concat_file,
        )

        split_points = [2880.0, 5760.0, 8640.0]

        def ffmpeg_side_effect(cmd, **kwargs):
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=ffmpeg_side_effect,
        ):
            sub_eps = _split_episode_at_points(
                parent_ep,
                split_points=split_points,
                output_dir=tmp_path,
                stages_dir=tmp_path / "_stages",
            )

        total = sum(ep.total_duration for ep in sub_eps)
        assert abs(total - parent_duration) < 0.01, (
            f"sub-ep durations sum to {total}s, expected {parent_duration}s"
        )


class TestExecuteIngestDeferredMutation:
    """execute_ingest must not re-process sub-episodes via mid-loop mutation."""

    def test_split_episode_not_reprocessed(self, tmp_path):
        """Sub-episodes inserted during split must not be re-visited by the loop."""
        from gomoveshift.pipeline.ingest import execute_ingest

        ct = datetime(2026, 3, 4, 8, 0)
        rec = _make_recording(61, 13200.0, ct)
        concat_file = tmp_path / "episode_04_2026-03-04.mp4"
        concat_file.write_bytes(b"\x00" * 1024)

        ep = Episode(
            index=4,
            recordings=[rec],
            date="2026-03-04",
            concat_path=concat_file,
        )
        plan = IngestPlan(source_dir=tmp_path, episodes=[ep])

        slice_a = tmp_path / "episode_04_a.mp4"
        slice_b = tmp_path / "episode_04_b.mp4"
        slice_a.write_bytes(b"\x00" * 512)
        slice_b.write_bytes(b"\x00" * 512)

        sub_a = Episode(
            index=401,
            recordings=[],
            date="2026-03-04",
            concat_path=slice_a,
            _explicit_total_duration=2880.0,
            _explicit_creation_time=ct,
        )
        sub_b = Episode(
            index=402,
            recordings=[],
            date="2026-03-04",
            concat_path=slice_b,
            _explicit_total_duration=10320.0,
            _explicit_creation_time=ct + timedelta(seconds=2880.0),
        )

        concat_call_count = []
        fake_gpx = Path("/fake/episode_04.gpx")

        def mock_concat(episode, output_dir):
            concat_call_count.append(episode.index)
            if episode.concat_path and episode.concat_path.exists():
                return episode.concat_path
            raise RuntimeError(f"unexpected concat call for ep {episode.index}")

        def mock_gpx(episode, stages_dir):
            # Return a truthy GPX path only for the original parent so the
            # split branch (`if gpx and episode.total_duration > MAX`) fires.
            return fake_gpx

        def mock_split(episode, splits, output_dir, stages_dir):
            return [sub_a, sub_b]

        def mock_find_splits(gpx, **kwargs):
            # Return split points for the parent; sub-eps don't get this far
            # because after deferred mutation they are inserted after the loop.
            return [2880.0]

        with (
            patch("gomoveshift.pipeline.ingest.concat_episode", side_effect=mock_concat),
            patch("gomoveshift.pipeline.ingest.extract_and_merge_gpx", side_effect=mock_gpx),
            patch(
                "gomoveshift.pipeline.ingest._split_episode_at_points",
                side_effect=mock_split,
            ),
            patch(
                "gomoveshift.pipeline.ingest.find_gps_split_points", side_effect=mock_find_splits
            ),
        ):
            # Set duration > MAX so the split branch fires
            ep._explicit_total_duration = MAX_EPISODE_DURATION + 1

            result = execute_ingest(plan, tmp_path, derive_locations=False)

        # Parent ep (index 4) concat was called once; sub-eps should NOT
        # have triggered a second concat call through the loop.
        assert 401 not in concat_call_count, "sub-ep 401 was re-processed by the for-loop"
        assert 402 not in concat_call_count, "sub-ep 402 was re-processed by the for-loop"
        assert len(result.episodes) == 2
        assert [ep.index for ep in result.episodes] == [1, 2]  # renumbered
