"""Tests for preflight source-filesystem and concat-symlink checks.

Both checks are robust-by-construction
gates: refuse the bad input before any irreversible artifact runs.
"""

from unittest.mock import patch

import pytest

from gomoveshift.pipeline.preflight import (
    check_concat_target_alignment,
    check_source_filesystem,
)


class _FakeStat:
    def __init__(self, st_dev):
        self.st_dev = st_dev


class TestCheckSourceFilesystem:
    def test_same_filesystem_passes(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        stg = tmp_path / "stg"
        stg.mkdir()
        # Both under tmp_path → same st_dev
        check_source_filesystem(src, stg)

    def test_cross_filesystem_raises_by_default(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        stg = tmp_path / "stg"
        stg.mkdir()
        with patch.object(type(src), "stat", autospec=True) as m_stat:

            def fake_stat(self, *args, **kwargs):
                if "src" in str(self):
                    return _FakeStat(st_dev=1)
                return _FakeStat(st_dev=2)

            m_stat.side_effect = fake_stat
            with pytest.raises(RuntimeError, match="different filesystem"):
                check_source_filesystem(src, stg)

    def test_cross_filesystem_passes_with_opt_in(self, tmp_path, caplog):
        src = tmp_path / "src"
        src.mkdir()
        stg = tmp_path / "stg"
        stg.mkdir()
        with patch.object(type(src), "stat", autospec=True) as m_stat:

            def fake_stat(self, *args, **kwargs):
                if "src" in str(self):
                    return _FakeStat(st_dev=1)
                return _FakeStat(st_dev=2)

            m_stat.side_effect = fake_stat
            check_source_filesystem(src, stg, allow_cross_fs=True)
        assert any("different filesystem" in r.message for r in caplog.records)

    def test_error_message_includes_copy_command(self, tmp_path):
        src = tmp_path / "weird_project"
        src.mkdir()
        stg = tmp_path / "stg"
        stg.mkdir()
        with patch.object(type(src), "stat", autospec=True) as m_stat:

            def fake_stat(self, *args, **kwargs):
                return _FakeStat(st_dev=1 if "weird" in str(self) else 2)

            m_stat.side_effect = fake_stat
            with pytest.raises(RuntimeError) as exc:
                check_source_filesystem(src, stg)
        msg = str(exc.value)
        assert "mkdir" in msg
        assert "cp " in msg
        assert "weird_project" in msg
        assert "--allow-cross-fs-source" in msg


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
