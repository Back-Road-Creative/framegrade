"""Tests for preflight source-filesystem and concat-symlink checks.

Both checks are robust-by-construction
gates: refuse the bad input before any irreversible artifact runs.
"""

import pytest

from gomoveshift.pipeline.preflight import (
    check_concat_target_alignment,
    check_source_filesystem,
)


import errno
import os

from gomoveshift.pipeline import preflight

# Two synthetic volumes stand in for the WSL layout: "ext4" (the Linux
# root, where ~/.cache/gomoveshift lives) and "drvfs" (the D: mount the
# footage really sits on). ``_device_of`` is the one seam; every path the
# check touches goes through it, so a test that forgets to resolve a
# symlink sees the symlink's own volume and fails.
EXT4, DRVFS = 1, 2


def _volumes(monkeypatch, drvfs_root):
    drvfs_root = os.path.realpath(drvfs_root)

    def fake_device(path):
        real = os.path.realpath(path)
        inside = real == drvfs_root or real.startswith(drvfs_root + os.sep)
        return DRVFS if inside else EXT4

    monkeypatch.setattr(preflight, "_device_of", fake_device)


def _production_layout(tmp_path, n=2):
    """ext4 symlink dir -> footage on the D: volume (the documented layout)."""
    dvol = tmp_path / "dvol" / "footage" / "trip"
    dvol.mkdir(parents=True)
    links = tmp_path / "cache" / "sources" / "trip"
    links.mkdir(parents=True)
    for i in range(1, n + 1):
        real = dvol / f"GX0{i:05}.MP4"
        real.write_bytes(b"\0" * 64)
        (links / real.name).symlink_to(real)
    return links, tmp_path / "dvol"


def _perf(caplog):
    return [r for r in caplog.records if "PERFORMANCE" in r.getMessage()]


class TestCheckSourceFilesystem:
    def test_same_filesystem_real_files_pass_quietly(self, tmp_path, caplog):
        src = tmp_path / "src"
        src.mkdir()
        (src / "GX010001.MP4").write_bytes(b"x")
        stg = tmp_path / "stg"
        stg.mkdir()
        check_source_filesystem(src, stg)
        assert _perf(caplog) == []

    def test_production_layout_warns_not_refuses(self, tmp_path, monkeypatch, caplog):
        # Case 1: ext4 staging, symlinks into D:. The old check
        # compared the symlink DIR (ext4) and passed silently; the files are
        # read from drvfs, so this is a speed cost and must be said aloud.
        src, dvol = _production_layout(tmp_path)
        stg = tmp_path / "cache" / "stages"
        stg.mkdir()
        _volumes(monkeypatch, dvol)
        check_source_filesystem(src, stg)
        (rec,) = _perf(caplog)
        assert rec.levelname == "WARNING"
        msg = rec.getMessage()
        assert "2 of 2" in msg
        assert str(dvol.resolve()) in msg  # where the files really are
        assert str(stg) in msg  # where staging is

    def test_staging_on_footage_volume_passes_without_warning(self, tmp_path, monkeypatch, caplog):
        # Case 2: staging on D:, symlink dir on ext4.
        # The old check refused this; the files and staging share a volume.
        src, dvol = _production_layout(tmp_path)
        stg = dvol / "stages"
        stg.mkdir()
        _volumes(monkeypatch, dvol)
        check_source_filesystem(src, stg)
        assert _perf(caplog) == []

    def test_source_dir_itself_on_other_volume_warns(self, tmp_path, monkeypatch, caplog):
        dvol = tmp_path / "dvol"
        dvol.mkdir()
        (dvol / "GX010001.MP4").write_bytes(b"x")
        stg = tmp_path / "stg"
        stg.mkdir()
        _volumes(monkeypatch, dvol)
        check_source_filesystem(dvol, stg)
        assert len(_perf(caplog)) == 1

    def test_mixed_volumes_counts_only_the_remote_files(self, tmp_path, monkeypatch, caplog):
        src, dvol = _production_layout(tmp_path, n=2)
        (src / "GX099999.MP4").write_bytes(b"local copy")
        stg = tmp_path / "stg"
        stg.mkdir()
        _volumes(monkeypatch, dvol)
        check_source_filesystem(src, stg)
        (rec,) = _perf(caplog)
        assert "2 of 3" in rec.getMessage()

    def test_symlink_chain_and_relative_link_resolve_to_final_target(
        self, tmp_path, monkeypatch, caplog
    ):
        dvol = tmp_path / "dvol"
        dvol.mkdir()
        real = dvol / "GX010001.MP4"
        real.write_bytes(b"x")
        hop = tmp_path / "hop"
        hop.mkdir()
        (hop / "mid.MP4").symlink_to(real)
        src = tmp_path / "src"
        src.mkdir()
        (src / "GX010001.MP4").symlink_to(os.path.relpath(hop / "mid.MP4", src))
        stg = dvol / "stg"
        stg.mkdir()
        _volumes(monkeypatch, dvol)
        check_source_filesystem(src, stg)
        assert _perf(caplog) == []

    def test_opt_in_flag_keeps_the_report_at_info(self, tmp_path, monkeypatch, caplog):
        caplog.set_level("INFO")
        src, dvol = _production_layout(tmp_path)
        stg = tmp_path / "stg"
        stg.mkdir()
        _volumes(monkeypatch, dvol)
        check_source_filesystem(src, stg, allow_cross_fs=True)
        (rec,) = _perf(caplog)
        assert rec.levelname == "INFO"

    def test_non_video_entries_ignored(self, tmp_path, monkeypatch, caplog):
        src = tmp_path / "src"
        src.mkdir()
        (src / "notes.txt").symlink_to(tmp_path / "gone.txt")  # dangling, not video
        (src / "GX010001.MP4").write_bytes(b"x")
        stg = tmp_path / "stg"
        stg.mkdir()
        check_source_filesystem(src, stg, video_extensions=[".mp4"])
        assert _perf(caplog) == []

    def test_missing_staging_defers_to_other_checks(self, tmp_path):
        src, _ = _production_layout(tmp_path)
        check_source_filesystem(src, tmp_path / "missing")


class TestSourceReadabilityIsACorrectnessFailure:
    """Refusals that no flag bypasses: the pipeline cannot read the footage."""

    def test_dangling_symlink_refuses_even_with_opt_in(self, tmp_path):
        # D: dropped off the bus; symlinks into it dangle. Step 2's
        # ``is_file()`` used to drop such entries silently.
        src, dvol = _production_layout(tmp_path)
        (dvol / "footage" / "trip" / "GX000001.MP4").unlink()
        stg = tmp_path / "stg"
        stg.mkdir()
        with pytest.raises(RuntimeError, match="GX000001.MP4") as exc:
            check_source_filesystem(src, stg, allow_cross_fs=True)
        assert "missing target" in str(exc.value)

    def test_read_error_refuses(self, tmp_path, monkeypatch):
        # Same failure mode: stat served from cache while every read hit EIO.
        src, _ = _production_layout(tmp_path, n=1)
        stg = tmp_path / "stg"
        stg.mkdir()

        def eio(path):
            raise OSError(errno.EIO, "Input/output error", str(path))

        monkeypatch.setattr(preflight, "_read_probe", eio)
        with pytest.raises(RuntimeError, match="cannot be read") as exc:
            check_source_filesystem(src, stg, allow_cross_fs=True)
        assert "EIO" in str(exc.value)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads mode-000 files")
    def test_unreadable_permissions_refuse(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        f = src / "GX010001.MP4"
        f.write_bytes(b"x")
        f.chmod(0)
        stg = tmp_path / "stg"
        stg.mkdir()
        try:
            with pytest.raises(RuntimeError, match="cannot be read"):
                check_source_filesystem(src, stg)
        finally:
            f.chmod(0o644)

    def test_run_preflight_refuses_dangling_source(self, tmp_path):
        src, dvol = _production_layout(tmp_path)
        (dvol / "footage" / "trip" / "GX000002.MP4").unlink()
        with pytest.raises(RuntimeError, match="missing target"):
            preflight.run_preflight(
                source_dir=src,
                output_dir=tmp_path / "out",
                staging_dir=tmp_path / "stg",
                require_tools=None,
            )


class TestMountFor:
    TABLE = [("/", "ext4"), ("/drv", "tmpfs"), ("/drv/d", "9p")]

    def test_longest_prefix_wins(self):
        assert preflight._mount_for("/drv/d/footage/x.MP4", self.TABLE) == ("/drv/d", "9p")

    def test_prefix_is_component_bounded(self):
        assert preflight._mount_for("/drv/dx/y", self.TABLE) == ("/drv", "tmpfs")

    def test_root_fallback(self):
        assert preflight._mount_for("/srv/cache/gomoveshift", self.TABLE) == ("/", "ext4")


class TestCheckConcatTargetAlignment:
    def test_no_output_dir_passes(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        check_concat_target_alignment(src, tmp_path / "missing")

    def test_symlink_inside_source_passes(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        target = src / "VID_001.mp4"
        target.write_bytes(b"")
        out = tmp_path / "out"
        out.mkdir()
        (out / "episode_01_2025-09-14.mp4").symlink_to(target)
        check_concat_target_alignment(src, out)

    def test_symlink_outside_source_raises(self, tmp_path):
        src = tmp_path / "src_new"
        src.mkdir()
        old = tmp_path / "src_old"
        old.mkdir()
        target = old / "VID_001.mp4"
        target.write_bytes(b"")
        out = tmp_path / "out"
        out.mkdir()
        (out / "episode_01_2025-09-14.mp4").symlink_to(target)
        with pytest.raises(RuntimeError, match="Stale concat symlink"):
            check_concat_target_alignment(src, out)

    def test_non_episode_symlinks_ignored(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        old = tmp_path / "old"
        old.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        unrelated = old / "log.txt"
        unrelated.write_bytes(b"")
        (out / "drive.log").symlink_to(unrelated)
        # Non-episode_* symlinks should not trigger
        check_concat_target_alignment(src, out)

    def test_error_lists_all_stale_links(self, tmp_path):
        src = tmp_path / "src_new"
        src.mkdir()
        old = tmp_path / "src_old"
        old.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        for i in (1, 2):
            t = old / f"VID_{i:03}.mp4"
            t.write_bytes(b"")
            (out / f"episode_0{i}_2025-09-14.mp4").symlink_to(t)
        with pytest.raises(RuntimeError) as exc:
            check_concat_target_alignment(src, out)
        msg = str(exc.value)
        assert "episode_01" in msg
        assert "episode_02" in msg


class TestConcatAlignmentSymlinkedSource:
    """A source dir of symlinks to another volume."""

    def _layout(self, tmp_path):
        real = tmp_path / "dwork"
        real.mkdir()
        vid = real / "VID_001.mp4"
        vid.write_bytes(b"")
        src = tmp_path / "sources"
        src.mkdir()
        (src / "VID_001.mp4").symlink_to(vid)
        out = tmp_path / "out"
        out.mkdir()
        return src, out, vid

    def test_link_to_resolved_target_of_source_file_passes(self, tmp_path):
        src, out, vid = self._layout(tmp_path)
        (out / "episode_01_2025-09-17.mp4").symlink_to(vid.resolve())
        check_concat_target_alignment(src, out)

    def test_link_to_file_outside_source_set_still_raises(self, tmp_path):
        src, out, _ = self._layout(tmp_path)
        other = tmp_path / "dwork" / "VID_999.mp4"
        other.write_bytes(b"")
        (out / "episode_01_2025-09-17.mp4").symlink_to(other)
        with pytest.raises(RuntimeError, match="Stale concat symlink"):
            check_concat_target_alignment(src, out)
