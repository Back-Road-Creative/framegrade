"""Tests for day_master_encoder — the day-master encode step of the day-first pipeline.

Given a DayBatch from day_grouper, produces a single day-level master
video file by:

  1. Concatenating all the day's raw per-episode files via ffmpeg stream
     copy (no re-encode) into `day_{date}_raw.mp4`.
  2. Running ONE color-correction + audio-loudness-normalization pass
     across the full raw concat via SinglePassProcessor (with OST
     drawtext disabled — OST is applied post-assembly by ost_applier).
  3. Cleaning up the intermediate raw concat on success.

The result is `day_{date}_master.mp4` — the day's authoritative
source of truth. Regular LFs and shorts for that day will be derived
from this master.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    """These tests cover concat / encode / resume MECHANICS, orthogonal to the
    privacy blur that is now default-ON in ``encode_day_master`` (its own coverage
    lives in ``test_day_master_blur.py``). Force blur off so the mocked encode isn't
    gated by the day-master mask build / ``.blurred`` marker."""
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


def _make_episode(
    index: int, concat_path: Path, start: datetime = None, duration_min: float = 30.0
):
    ep = MagicMock()
    ep.index = index
    ep.creation_time = start or datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
    ep.total_duration = duration_min * 60
    ep.total_duration_min = duration_min
    ep.concat_path = concat_path
    ep.gpx_path = None
    return ep


def _make_day_batch(date: str, episode_paths: list):
    from gomoveshift.pipeline.day_grouper import DayBatch

    episodes = []
    for i, p in enumerate(episode_paths, start=1):
        start = datetime.fromisoformat(f"{date}T{10 + i:02d}:00:00+00:00")
        episodes.append(_make_episode(i, p, start=start))
    return DayBatch(date=date, episodes=episodes)


class TestEncodeDayMaster:
    def test_single_episode_day_produces_master(self, tmp_path, monkeypatch):
        """One-episode day: concat step is a simple copy, then SinglePass runs."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep_file = tmp_path / "ep_001.mp4"
        ep_file.touch()
        batch = _make_day_batch("2026-02-20", [ep_file])

        captured_ffmpeg = []

        def fake_run(cmd, *args, **kwargs):
            captured_ffmpeg.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
            return result

        # Patch subprocess.run so no real ffmpeg invocation happens
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            fake_run,
        )

        # Mock SinglePassProcessor so we capture its call without running ffmpeg
        fake_result = MagicMock()
        fake_result.success = True
        fake_result.output_path = tmp_path / "day_master.mp4"

        sp_calls = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_calls.append(
                {
                    "input": video_path,
                    "output": output_path,
                    "location": location,
                    "skip_ost": kwargs.get("skip_ost"),
                }
            )
            # Create the expected output file so the encoder can verify
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return fake_result

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        output_dir = tmp_path / "days_out"
        result_path = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        # Master file was created at the expected path
        assert result_path is not None
        assert result_path.name == "day_2026-02-20_master.mp4"
        assert result_path.parent == output_dir

        # SinglePassProcessor was called once, with OST explicitly disabled.
        # Per the atomic-rename pattern, it writes to a .tmp sidecar
        # which is `os.replace`d onto the final master path on success.
        assert len(sp_calls) == 1
        assert sp_calls[0]["skip_ost"] is True
        assert sp_calls[0]["location"] == ""
        assert (
            sp_calls[0]["output"]
            == result_path.parent / f"{result_path.stem}.tmp{result_path.suffix}"
        )

    @pytest.mark.parametrize("skip_audio", [True, False])
    def test_config_skip_audio_threaded_to_single_pass(self, tmp_path, monkeypatch, skip_audio):
        """--no-audio (config.drive.skip_audio) must reach SinglePassProcessor so
        the day master is encoded -an — otherwise a --no-audio drive's LFs
        (stream-cut from the master) still ship audio."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep_file = tmp_path / "ep_001.mp4"
        ep_file.touch()
        batch = _make_day_batch("2026-02-20", [ep_file])

        def fake_run(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
            return result

        monkeypatch.setattr("gomoveshift.pipeline.day_master_encoder.subprocess.run", fake_run)

        sp_calls = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_calls.append({"skip_audio": kwargs.get("skip_audio")})
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            fake_result = MagicMock()
            fake_result.success = True
            fake_result.output_path = output_path
            return fake_result

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process", fake_process
        )

        cfg = Config()
        cfg.drive.skip_audio = skip_audio
        encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "days_out",
            config=cfg,
        )

        assert len(sp_calls) == 1
        assert sp_calls[0]["skip_audio"] is skip_audio

    def test_multi_episode_day_concats_before_encoding(self, tmp_path, monkeypatch):
        """Multi-episode day: ffmpeg concat demuxer is invoked first, then
        SinglePass runs on the concatenated raw."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        ep3 = tmp_path / "ep_003.mp4"
        for p in (ep1, ep2, ep3):
            p.touch()

        batch = _make_day_batch("2026-02-20", [ep1, ep2, ep3])

        captured_ffmpeg = []

        def fake_run(cmd, *args, **kwargs):
            captured_ffmpeg.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
            return result

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            fake_run,
        )

        sp_calls = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_calls.append({"input": video_path, "output": output_path})
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            fake_result = MagicMock()
            fake_result.success = True
            fake_result.output_path = output_path
            return fake_result

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        output_dir = tmp_path / "days_out"
        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        # At least one ffmpeg call must be a concat demuxer invocation
        concat_cmds = [
            c for c in captured_ffmpeg if isinstance(c, list) and any("concat" in str(a) for a in c)
        ]
        assert concat_cmds, f"No concat ffmpeg call captured: {captured_ffmpeg}"
        concat_cmd = concat_cmds[0]
        # Stream copy (no re-encode)
        assert "-c" in concat_cmd and "copy" in concat_cmd

        # SinglePassProcessor called once with the concat result as input
        assert len(sp_calls) == 1

    def test_day_concat_subprocess_timeout_is_3600s(self, tmp_path, monkeypatch):
        """The day-master raw concat reads episode files from cross-mount
        (a WSL drive mount) and writes to local-fs cache. A multi-hundred-GB day at WSL2 cross-mount
        throughput (~100 MB/s) runs ~33-37 min — the legacy 1800s ceiling would trip exactly
        like the splitter's 300s ceiling did before it was raised.

        Lock the timeout at 3600s so a future regression to 1800s
        (e.g. someone "tightening" the ceiling without doing the
        cross-mount throughput math) breaks this test. Mirrors the
        contract pattern used by
        ``test_subprocess_timeout_matches_concat_path`` in
        ``test_gps_split_null_island_filter.py``.
        """
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        for p in (ep1, ep2):
            p.touch()
        batch = _make_day_batch("2026-03-05", [ep1, ep2])

        captured_concat_timeouts = []

        def fake_run(cmd, *args, **kwargs):
            # Only capture the concat demuxer invocations — other
            # subprocess.run calls in this module (ffprobe duration
            # checks) use shorter timeouts by design.
            if isinstance(cmd, list) and "-f" in cmd and "concat" in cmd:
                captured_concat_timeouts.append(kwargs.get("timeout"))
            result = MagicMock()
            result.returncode = 0
            result.stdout = b""
            result.stderr = b""
            return result

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            fake_run,
        )

        def fake_process(self, video_path, output_path, location, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            r = MagicMock()
            r.success = True
            r.output_path = output_path
            return r

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "out",
            config=Config(),
        )

        assert captured_concat_timeouts, (
            "expected at least one ffmpeg concat invocation to be captured"
        )
        assert all(t == 3600 for t in captured_concat_timeouts), (
            f"day-master concat timeout regressed: {captured_concat_timeouts}"
        )

    def test_concat_list_file_contains_all_episodes_in_order(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The concat demuxer's list file must reference each episode in the
        batch's order (which is chronological)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        for p in (ep1, ep2):
            p.touch()

        batch = _make_day_batch("2026-02-21", [ep1, ep2])

        captured_list_contents = []

        real_write_text = Path.write_text

        def sniffing_write_text(self, data, *args, **kwargs):
            if self.name.endswith(".txt") and "concat" in self.name:
                captured_list_contents.append(data)
            return real_write_text(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", sniffing_write_text)
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        def fake_process(self, video_path, output_path, location, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "days_out",
            config=Config(),
        )

        # Exactly one concat list was written with both episode paths in order
        assert len(captured_list_contents) == 1
        contents = captured_list_contents[0]
        # Order: ep_001 must appear before ep_002
        idx1 = contents.find("ep_001.mp4")
        idx2 = contents.find("ep_002.mp4")
        assert idx1 != -1 and idx2 != -1, f"Concat list missing episodes: {contents}"
        assert idx1 < idx2, f"Episodes out of order in concat list: {contents}"

    def test_empty_day_batch_returns_none(self, tmp_path):
        """A day batch with no episodes produces no master — return None
        without attempting any ffmpeg work."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_grouper import DayBatch
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        batch = DayBatch(date="2026-02-20", episodes=[])
        result = encode_day_master(
            day_batch=batch,
            output_dir=tmp_path,
            config=Config(),
        )
        assert result is None

    def test_raw_concat_cleaned_up_after_success(self, tmp_path, monkeypatch):
        """The intermediate raw concat file is deleted once the master is
        produced — saves disk space, especially for long multi-hour drives."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        for p in (ep1, ep2):
            p.touch()

        batch = _make_day_batch("2026-02-20", [ep1, ep2])

        # The encoder creates a raw concat file, then an _master.mp4. After
        # success, only the master should remain; the raw concat should be
        # cleaned up.
        raw_paths_seen = []

        def fake_run(cmd, *args, **kwargs):
            # Simulate ffmpeg creating the output file
            for i, tok in enumerate(cmd):
                if isinstance(tok, (str, Path)) and str(tok).endswith("_raw.mp4"):
                    out = Path(tok)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.touch()
                    raw_paths_seen.append(out)
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            fake_run,
        )

        def fake_process(self, video_path, output_path, location, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "days_out",
            config=Config(),
        )

        # Every raw concat file the encoder created must NOT exist anymore
        assert raw_paths_seen, "No raw concat file was created during the test"
        for raw in raw_paths_seen:
            assert not raw.exists(), f"Raw concat {raw} was not cleaned up"

    def test_failed_single_pass_returns_none_and_leaves_no_partial_master(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When SinglePassProcessor fails (e.g. encode error), the function
        returns None and does not leave a partially-written master file."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        def fake_process(self, video_path, output_path, location, **kwargs):
            # Simulate encode failure — output file not created, success=False
            return MagicMock(success=False, output_path=None)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        output_dir = tmp_path / "days_out"
        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )
        assert result is None

    def test_single_pass_called_with_skip_ost_true(self, tmp_path, monkeypatch):
        """OST drawtext must NOT be applied to the day master — that's
        ost_applier's job, post-assembly."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        captured_kwargs = {}

        def fake_process(self, video_path, output_path, location, **kwargs):
            captured_kwargs.update(kwargs)
            captured_kwargs["location"] = location
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "days_out",
            config=Config(),
        )

        assert captured_kwargs.get("skip_ost") is True
        assert captured_kwargs.get("location") == ""

    def test_skip_ost_true_even_with_truthy_location(self, tmp_path, monkeypatch):
        """A day-wide location must NOT cause the master to burn OST: OST is
        post-assembly (apply_ost / build_ost_filter), and the master is
        stream-cut into LFs AND shorts, so a master-level overlay would bleed.
        Regression: the master once used skip_ost=not bool(location)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        captured_kwargs = {}

        def fake_process(self, video_path, output_path, location, **kwargs):
            captured_kwargs.update(kwargs)
            captured_kwargs["location"] = location
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "days_out",
            config=Config(),
            location="Dingle Peninsula, County Kerry",
        )

        # Location still threads through (process accepts it) but OST is skipped.
        assert captured_kwargs.get("skip_ost") is True

    def test_output_filename_includes_date(self, tmp_path, monkeypatch):
        """Master filename format: day_{YYYY-MM-DD}_master.mp4"""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-24", [ep])

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        def fake_process(self, video_path, output_path, location, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )

        result = encode_day_master(
            day_batch=batch,
            output_dir=tmp_path / "days_out",
            config=Config(),
        )
        assert result.name == "day_2026-02-24_master.mp4"


class TestEncodeDayMasterResumeSkip:
    """Resume behavior: if a valid day master already exists, skip re-encoding.

    Motivation: the day-first pipeline has no per-day granularity in its
    phase-completion state. On resume, phase_bc_day_first iterates every day
    batch unconditionally. Without this check, a day whose master is already
    present gets concatenated + re-encoded from scratch (≈8 hours of wasted
    GPU time per day), overwriting the valid existing master in the process.

    The check: probe the existing master's duration; if it's within 1% of
    the sum of the batch's episode durations, consider it complete and
    return its path without invoking SinglePassProcessor.
    """

    def test_existing_valid_master_is_returned_without_reencoding(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Master already exists with a matching duration → skip concat + encode."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        ep1.touch()
        ep2.touch()
        batch = _make_day_batch("2026-02-20", [ep1, ep2])
        expected_total = batch.total_duration  # 2 * 30 * 60 = 3600s

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-20_master.mp4"
        master.write_bytes(b"pretend this is a 30GB encode")

        # Mock the ffprobe helper: master probes to full expected_total;
        # source episodes probe to their per-episode duration so the
        # integrity guard's per-episode sum agrees with batch.total_duration.
        per_ep = expected_total / len(batch.episodes)

        def fake_probe(self, path):
            if Path(path).name == master.name:
                return expected_total
            return per_ep

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )

        # Fail loudly if SinglePassProcessor is called — it must not be
        def forbidden_process(self, video_path, output_path, location, **kwargs):
            raise AssertionError(
                "SinglePassProcessor.process() should not be called "
                "when a valid master already exists"
            )

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            forbidden_process,
        )

        # Fail loudly if any concat ffmpeg runs either
        def forbidden_run(cmd, *a, **kw):
            raise AssertionError(f"subprocess.run should not be called on resume-skip path: {cmd}")

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            forbidden_run,
        )

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert result == master
        assert master.exists()

    def test_disk_space_check_before_encode(self, tmp_path, monkeypatch):
        """Disk space check in encode_day_master must fire before any
        ffmpeg work — ensures no partial file on disk-full."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 10_000)
        batch = _make_day_batch("2026-02-25", [ep])

        def fake_check(path, required):
            raise RuntimeError(
                f"Insufficient disk space at {path}: 0.1 GB free, "
                f"need ~{required / 1_073_741_824:.1f} GB"
            )

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.check_disk_space",
            fake_check,
        )

        def forbidden_run(cmd, *a, **kw):
            raise AssertionError("subprocess.run called despite insufficient space")

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            forbidden_run,
        )

        import pytest

        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            encode_day_master(
                day_batch=batch,
                output_dir=tmp_path / "days_out",
                config=Config(),
            )

    def test_disk_space_check_sizes_what_is_still_to_write(self, tmp_path, monkeypatch):
        """The check requires the raw concat still to be written plus a master of up
        to the source's own size. Two failures it replaces:
        - A resume after a crash found the day's raw already on disk and demanded
          its bytes again, so a day on a tight volume could never resume.
        - The old 1.5x rule assumed a master of half the source, but recorded
          masters measure 0.63x-0.89x, so a tight volume passed the check and
          could fill mid-encode."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        import pytest

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 10_000)
        batch = _make_day_batch("2026-02-25", [ep])
        output_dir = tmp_path / "days_out"
        output_dir.mkdir()
        raw = output_dir / "day_2026-02-25_raw.mp4"

        seen = []

        def fake_check(path, required):
            seen.append(required)
            raise RuntimeError("stop after the disk check")

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.check_disk_space",
            fake_check,
        )

        for raw_bytes in (None, 6_000, 10_000):
            if raw_bytes is not None:
                raw.write_bytes(b"r" * raw_bytes)
            with pytest.raises(RuntimeError, match="stop after the disk check"):
                encode_day_master(day_batch=batch, output_dir=output_dir, config=Config())

        # fresh: raw + master; partial raw: the rest of it + master; full raw: master
        assert seen == [20_000, 14_000, 10_000]

    def test_existing_partial_master_is_reencoded(self, tmp_path, monkeypatch):
        """Master exists but its duration is far below expected → re-encode."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        ep1.touch()
        ep2.touch()
        batch = _make_day_batch("2026-02-21", [ep1, ep2])
        expected_total = batch.total_duration  # 3600s

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"partial encode, bailed mid-stream")

        # Master probes to 18% of expected (mid-encode stall); source episodes
        # probe to their full per-episode duration so the integrity guard
        # passes and the partial-master gate is the one that fires.
        per_ep = expected_total / len(batch.episodes)
        sp_calls = []

        def fake_probe(self, path):
            if Path(path).name == master.name:
                # Before the re-encode this is the stalled partial; after it, the
                # freshly promoted master, which the promotion-time duration gate
                # probes — and a healthy re-encode is full length.
                return expected_total if sp_calls else expected_total * 0.18
            return per_ep

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_calls.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert len(sp_calls) == 1, "Partial master should trigger a full re-encode, not skip"

    def test_existing_master_unprobeable_is_reencoded(self, tmp_path, monkeypatch):
        """Master exists but ffprobe returns 0 (corrupt/unreadable) → re-encode."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-22", [ep])

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-22_master.mp4"
        master.write_bytes(b"corrupt")

        # Master is unprobeable; sources probe to their per-episode duration
        # so the integrity guard passes and the master probe gate fires.
        per_ep = batch.total_duration / len(batch.episodes)

        def fake_probe(self, path):
            if Path(path).name == master.name:
                return 0.0
            return per_ep

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )

        sp_calls = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_calls.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert len(sp_calls) == 1, "Unprobeable master should fall through to re-encode, not skip"

    def test_resume_skip_rejects_zero_duration(self, tmp_path, monkeypatch):
        """When expected_duration is 0 (missing total_duration on batch),
        the resume check must NOT skip — ANY existing file would pass
        the 0 * 0.99 == 0.0 comparison, even a 1-byte truncation."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-02-23", [ep])
        # Force total_duration to 0 by zeroing every episode's duration
        for episode in batch.episodes:
            episode.total_duration = 0.0

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-23_master.mp4"
        master.write_bytes(b"truncated junk - should NOT pass resume")

        sp_calls = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_calls.append(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )
        # Patch disk space check to not interfere
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.check_disk_space",
            lambda path, required: None,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert len(sp_calls) == 1, "Zero expected_duration must force re-encode, not skip"

    def test_resume_skip_accepts_valid_duration(self, tmp_path, monkeypatch):
        """When expected_duration=100 and actual=99.5 (within 1%), the
        resume check should skip re-encoding."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-02-24", [ep])
        # Set episode duration so batch.total_duration == 100.0
        for episode in batch.episodes:
            episode.total_duration = 100.0

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-24_master.mp4"
        master.write_bytes(b"complete master")

        # Report actual duration at 99.5s — within 1% of 100s
        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            lambda self, path: 99.5,
        )

        # Fail loudly if SinglePassProcessor is called
        def forbidden_process(self, video_path, output_path, location, **kwargs):
            raise AssertionError(
                "SinglePassProcessor.process() should not be called "
                "when a valid master already exists"
            )

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            forbidden_process,
        )

        def forbidden_run(cmd, *a, **kw):
            raise AssertionError("subprocess.run should not be called on resume")

        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            forbidden_run,
        )

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert result == master
