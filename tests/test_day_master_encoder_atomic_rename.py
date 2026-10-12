"""Regression tests for atomic master rename.

The day master must be encoded to a temporary path and only renamed to its
final filename after `SinglePassProcessor.process()` returns success. A
SIGKILL or power loss mid-encode would otherwise leave a corrupt mp4 (no
moov atom) sitting at the final filename, which the resume fast-path in
phase_bc_day_first would silently trust on the next run.

Mirrors the atomic-write pattern already used in `gomoveshift/pipeline/content_ledger.py` (write to tmp, then
`os.replace`).
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    """Atomic-rename MECHANICS, orthogonal to the privacy blur now default-ON in
    ``encode_day_master`` (its coverage lives in ``test_day_master_blur.py``). Force
    blur off so the mocked encode isn't gated by the day-master mask build / marker."""
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


def _make_episode(index: int, concat_path: Path):
    ep = MagicMock()
    ep.index = index
    ep.creation_time = datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc)
    ep.total_duration = 30 * 60
    ep.total_duration_min = 30.0
    ep.concat_path = concat_path
    ep.gpx_path = None
    return ep


def _make_day_batch(date: str, episode_paths: list):
    from gomoveshift.pipeline.day_grouper import DayBatch

    episodes = [_make_episode(i + 1, p) for i, p in enumerate(episode_paths)]
    return DayBatch(date=date, episodes=episodes)


def _tmp_path_for(master: Path) -> Path:
    """Convention: master.mp4 → master.tmp.mp4

    The ``.tmp`` token sits between the stem and the real extension so ffmpeg
    can still infer the output container from the trailing ``.mp4``. The older
    ``master.mp4.tmp`` shape broke ffmpeg's muxer auto-detection in production.
    """
    return master.parent / f"{master.stem}.tmp{master.suffix}"


class TestAtomicMasterRename:
    def test_success_leaves_final_path_only_no_tmp_residue(self, tmp_path, monkeypatch):
        """On success, master_path exists and the .tmp sibling does not."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        # Capture which path SinglePass writes to — it must be a .tmp path,
        # not the final master path.
        sp_target_paths = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            sp_target_paths.append(Path(output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        output_dir = tmp_path / "days_out"
        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        master = output_dir / "day_2026-02-20_master.mp4"
        tmp_master = _tmp_path_for(master)

        assert result == master
        assert master.exists(), "Final master must exist after successful encode"
        assert not tmp_master.exists(), (
            "Tmp sidecar must be renamed away on success, not linger as residue"
        )

        # SinglePass must have been told to write to the .tmp path, not directly
        # to the final master — that's the whole point of the atomic-rename fix.
        assert len(sp_target_paths) == 1
        assert sp_target_paths[0] == tmp_master, (
            f"SinglePass should encode to {tmp_master}, got {sp_target_paths[0]}"
        )

    def test_encoder_failure_leaves_neither_final_nor_tmp(self, tmp_path, monkeypatch):
        """When SinglePassProcessor.process raises, no master or .tmp file
        must remain — the partial encode must be unlinked so the next resume
        does not mistake it for a finished master."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        def exploding_process(self, video_path, output_path, location, **kwargs):
            # Simulate an encoder that wrote a partial .tmp before dying
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"corrupt partial encode, no moov atom")
            raise RuntimeError("encoder crashed mid-stream")

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            exploding_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        output_dir = tmp_path / "days_out"
        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        master = output_dir / "day_2026-02-20_master.mp4"
        tmp_master = _tmp_path_for(master)

        assert result is None
        assert not master.exists(), (
            "No final master must exist on encoder failure — corrupt files at "
            "the final path are silently trusted by the resume fast-path"
        )
        assert not tmp_master.exists(), (
            "Tmp sidecar must be unlinked on encoder failure, not left as residue"
        )

    def test_failed_result_success_false_unlinks_tmp(self, tmp_path, monkeypatch):
        """When SinglePassProcessor returns success=False (clean failure),
        no master or .tmp file must remain."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        def failing_process(self, video_path, output_path, location, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"partial encode, success=False")
            return MagicMock(success=False, output_path=None)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            failing_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        output_dir = tmp_path / "days_out"
        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        master = output_dir / "day_2026-02-20_master.mp4"
        tmp_master = _tmp_path_for(master)

        assert result is None
        assert not master.exists()
        assert not tmp_master.exists(), (
            "Tmp sidecar must be cleaned up when SinglePassProcessor reports failure"
        )

    def test_preexisting_tmp_from_killed_run_is_overwritten_cleanly(self, tmp_path, monkeypatch):
        """A leftover .tmp from a prior killed run must be overwritten without
        raising FileExistsError. Uses os.replace semantics — destination is
        atomically replaced regardless of pre-existing content."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-20_master.mp4"
        stale_tmp = _tmp_path_for(master)
        stale_tmp.write_bytes(b"stale .tmp from a previous SIGKILL")

        def fake_process(self, video_path, output_path, location, **kwargs):
            # The encoder is expected to overwrite the stale .tmp, not error
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fresh encode bytes")
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert result == master
        assert master.exists()
        assert master.read_bytes() == b"fresh encode bytes", (
            "Final master must contain the fresh encode bytes, not the stale .tmp"
        )
        assert not stale_tmp.exists(), "Stale .tmp must be cleared after a successful run"


class TestTmpMasterNamingFFmpegCompatible:
    """The atomic-rename ``.tmp`` token must sit *inside* the filename, not at
    the tail, so ffmpeg can still infer the output muxer from the trailing
    ``.mp4`` extension.

    Production failure: the original
    convention ``<name>.mp4.tmp`` made ffmpeg abort with ``Unable to choose
    an output format for '....mp4.tmp'``, killing the day master encode for
    every day in the batch.
    """

    def test_tmp_master_path_uses_dot_tmp_dot_mp4_shape(self):
        """Direct unit test of the naming convention. The path returned for
        ``day_2026-02-20_master.mp4`` must end in ``.tmp.mp4`` — never ``.mp4.tmp``."""
        master = Path("/x/day_2026-02-20_master.mp4")
        # Mirror the production helper inline so the assertion is independent
        # of the encoder module's internal symbol naming.
        tmp = master.parent / f"{master.stem}.tmp{master.suffix}"

        assert tmp == Path("/x/day_2026-02-20_master.tmp.mp4")
        assert tmp.suffix == ".mp4", (
            "Trailing extension must remain .mp4 so ffmpeg can pick the muxer"
        )
        assert not tmp.name.endswith(".mp4.tmp"), (
            "Old .mp4.tmp convention broke ffmpeg muxer auto-detection"
        )

    def test_singlepass_receives_path_with_mp4_tail_not_tmp_tail(self, tmp_path, monkeypatch):
        """End-to-end: the path handed to SinglePassProcessor (which becomes
        the ffmpeg ``-i ... <output>`` arg) must end in ``.mp4`` so ffmpeg
        can infer the container format. Regression test for the
        production failure."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        captured_outputs: list[Path] = []

        def fake_process(self, video_path, output_path, location, **kwargs):
            captured_outputs.append(Path(output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).touch()
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        output_dir = tmp_path / "days_out"
        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert len(captured_outputs) == 1
        out = captured_outputs[0]
        assert out.suffix == ".mp4", (
            f"SinglePass output path must end in .mp4 (got {out!r}); "
            "ffmpeg cannot infer container from .tmp"
        )
        assert out.name.endswith(".tmp.mp4"), (
            f"Expected <stem>.tmp.mp4 sidecar shape, got {out.name!r}"
        )
        assert not out.name.endswith(".mp4.tmp"), (
            "Regression: the .mp4.tmp shape broke ffmpeg in production"
        )

    def test_stale_cleanup_targets_dot_tmp_dot_mp4_files(self, tmp_path, monkeypatch):
        """Stale-tmp cleanup must unlink the new ``<stem>.tmp.mp4`` shape, not
        any leftover legacy ``.mp4.tmp`` files (those are dead artifacts from
        a never-successful run and would never have been written
        successfully anyway)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.touch()
        batch = _make_day_batch("2026-02-20", [ep])

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-02-20_master.mp4"
        new_shape_stale = output_dir / "day_2026-02-20_master.tmp.mp4"
        new_shape_stale.write_bytes(b"stale .tmp.mp4 from a previous SIGKILL")

        def fake_process(self, video_path, output_path, location, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fresh encode bytes")
            return MagicMock(success=True, output_path=output_path)

        monkeypatch.setattr(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
            fake_process,
        )
        monkeypatch.setattr(
            "gomoveshift.pipeline.day_master_encoder.subprocess.run",
            lambda cmd, *a, **kw: MagicMock(returncode=0, stdout=b"", stderr=b""),
        )

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert result == master
        assert master.exists()
        assert master.read_bytes() == b"fresh encode bytes"
        assert not new_shape_stale.exists(), (
            "Stale .tmp.mp4 from previous run must be cleared via os.replace promotion"
        )
