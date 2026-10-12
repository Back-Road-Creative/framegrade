"""``gomoveshift cleanup`` reclaims a published drive's stage cache.

Drive runs keep their heavy intermediates under the ext4 stage cache
(``~/.cache/gomoveshift/<project>/_stages``, or ``GMS_INTERMEDIATE_DIR``) and
never write a ``<project>_FINAL.mp4``, so the single-video gate skipped every
finished drive and ``cleanup`` printed "Nothing to clean up." while the
uploaded renders stayed on disk until a later preflight refused for lack of space.

Rule pinned here: once ``phase_d`` is done, an ``lf_NN.mp4`` render whose LF
has a YouTube ``video_id`` is reclaimable, and the episode concat is
reclaimable once every LF is uploaded and ``source_dir`` still holds the
footage to rebuild it. Symlinked episodes (single-file days point at the
source) are never touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli.handlers import cleanup as cleanup_handler
from gomoveshift.pipeline.cleanup import PipelineCleanup


def _drive(
    base: Path,
    name: str = "example-trip-2026-03-05-battlefield",
    *,
    uploads=None,
    phases=("phase_a", "phase_bc_day_first", "phase_d"),
    source_files=("GX010001.MP4",),
    symlink_episode: bool = False,
) -> Path:
    project = base / name
    stages = project / "_stages"
    lfs = stages / "day_2026-03-05" / "regular_lfs"
    lfs.mkdir(parents=True)
    source = base / "sources" / name
    source.mkdir(parents=True)
    for f in source_files:
        (source / f).write_bytes(b"s" * 10)
    episode = stages / "episode_01_2026-03-05.mp4"
    if symlink_episode:
        episode.symlink_to(source / source_files[0])
    else:
        episode.write_bytes(b"e" * 4000)
    (lfs / "lf_01.mp4").write_bytes(b"l" * 3000)
    (stages / "day_2026-03-05" / "day_2026-03-05_blur_mask.mkv").write_bytes(b"m" * 10)
    state = {
        "phases_completed": list(phases),
        "source_dir": str(source),
        "episode_files": [str(episode)],
        "youtube_uploads": {"1": {"video_id": "vid00000001"}} if uploads is None else uploads,
    }
    (stages / "pipeline_state.json").write_text(json.dumps(state))
    old = 1_700_000_000
    os.utime(stages, (old, old))
    return project


def _plan(tmp_path: Path, **kw):
    cache = tmp_path / "cache"
    project = _drive(cache, **kw)
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out", staging_dir=tmp_path / "stg", stage_cache_base=cache
    )
    plan = cleanup.plan_cleanup(cleanup.discover_projects(), min_age_days=0)
    return project, {Path(f["path"]).name: f for f in plan.files}, plan


def test_discovery_includes_the_stage_cache(tmp_path):
    cache = tmp_path / "cache"
    project = _drive(cache)
    (tmp_path / "out").mkdir()
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out", staging_dir=tmp_path / "stg", stage_cache_base=cache
    )
    assert cleanup.discover_projects() == [project]


def test_uploaded_drive_reclaims_lf_render_and_episode_concat(tmp_path):
    _, files, _ = _plan(tmp_path)
    assert set(files) == {"lf_01.mp4", "episode_01_2026-03-05.mp4", "GX010001.MP4"}
    assert files["lf_01.mp4"]["size"] == 3000
    assert "vid00000001" in files["lf_01.mp4"]["reason"]
    assert "source" in files["episode_01_2026-03-05.mp4"]["reason"]


def test_state_mask_and_blur_files_are_kept(tmp_path):
    _, files, _ = _plan(tmp_path)
    assert "pipeline_state.json" not in files
    assert not any(n.endswith(".mkv") for n in files)


def test_lf_without_video_id_is_kept_and_episode_too(tmp_path):
    _, files, plan = _plan(tmp_path, uploads={})
    assert files == {}
    assert any("not uploaded" in s["reason"] for s in plan.skipped)


def test_partial_upload_keeps_the_episode_concat(tmp_path):
    project = _drive(tmp_path / "cache", uploads={"1": {"video_id": "abc"}, "2": {}})
    (project / "_stages" / "day_2026-03-05" / "regular_lfs" / "lf_02.mp4").write_bytes(b"x")
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out",
        staging_dir=tmp_path / "stg",
        stage_cache_base=tmp_path / "cache",
    )
    plan = cleanup.plan_cleanup([project], min_age_days=0)
    names = {Path(f["path"]).name for f in plan.files}
    assert names == {"lf_01.mp4"}  # sources kept: LF 2 is still unpublished


def test_unfinished_drive_is_untouched(tmp_path):
    _, files, _ = _plan(tmp_path, phases=("phase_a",))
    assert files == {}


def test_missing_source_keeps_the_episode_concat(tmp_path):
    project = _drive(tmp_path / "cache")
    for f in Path(
        json.loads((project / "_stages" / "pipeline_state.json").read_text())["source_dir"]
    ).iterdir():
        f.unlink()
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out",
        staging_dir=tmp_path / "stg",
        stage_cache_base=tmp_path / "cache",
    )
    plan = cleanup.plan_cleanup([project], min_age_days=0)
    names = {Path(f["path"]).name for f in plan.files}
    assert names == {"lf_01.mp4"}
    assert any("source" in s["reason"] for s in plan.skipped)


def test_symlinked_episode_is_never_listed(tmp_path):
    _, files, _ = _plan(tmp_path, symlink_episode=True)
    assert set(files) == {"lf_01.mp4", "GX010001.MP4"}


def _sources(project: Path) -> Path:
    return project.parent / "sources" / project.name


def test_published_drive_lists_and_reclaims_copied_sources(tmp_path):
    project = _drive(tmp_path / "cache")
    src = _sources(project)
    (src / "GX020002.MP4").write_bytes(b"t" * 20)
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out",
        staging_dir=tmp_path / "stg",
        stage_cache_base=tmp_path / "cache",
    )
    plan = cleanup.plan_cleanup([project], min_age_days=0)
    listed = {Path(f["path"]).name: f for f in plan.files}
    assert listed["GX020002.MP4"]["size"] == 20
    assert plan.source_dirs == [{"path": str(src), "size": 30}]
    assert (src / "GX020002.MP4").exists()  # dry run deletes nothing
    cleanup.execute_plan(plan)
    assert not (src / "GX020002.MP4").exists()
    assert not src.exists()


def test_sources_symlinks_are_unlinked_never_followed(tmp_path):
    project = _drive(tmp_path / "cache")
    src = _sources(project)
    original = tmp_path / "d_drive_original.MP4"
    original.write_bytes(b"o" * 99)
    (src / "link.MP4").symlink_to(original)
    d_dir = tmp_path / "d_dir"
    d_dir.mkdir()
    (d_dir / "keep.MP4").write_bytes(b"k")
    (src / "linked_dir").symlink_to(d_dir)
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out",
        staging_dir=tmp_path / "stg",
        stage_cache_base=tmp_path / "cache",
    )
    plan = cleanup.plan_cleanup([project], min_age_days=0)
    paths = {f["path"] for f in plan.files}
    assert str(src / "link.MP4") in paths and str(src / "linked_dir") in paths
    assert not any(str(d_dir) in p and p != str(src / "linked_dir") for p in paths)
    cleanup.execute_plan(plan)
    assert not (src / "link.MP4").is_symlink()
    assert original.read_bytes() == b"o" * 99
    assert (d_dir / "keep.MP4").exists()


def test_symlinked_sources_dir_is_skipped_not_followed(tmp_path):
    project = _drive(tmp_path / "cache")
    src = _sources(project)
    real = tmp_path / "d_sources"
    src.rename(real)
    src.symlink_to(real)
    cleanup = PipelineCleanup(
        output_base=tmp_path / "out",
        staging_dir=tmp_path / "stg",
        stage_cache_base=tmp_path / "cache",
    )
    plan = cleanup.plan_cleanup([project], min_age_days=0)
    assert plan.source_dirs == []
    assert not any("GX010001" in f["path"] for f in plan.files)
    assert any("symlink" in s["reason"] for s in plan.skipped)


def test_unpublished_or_unfinished_drive_keeps_its_sources(tmp_path):
    for i, kw in enumerate(({"uploads": {}}, {"phases": ("phase_a",)})):
        _, files, plan = _plan(tmp_path / str(i), **kw)
        assert "GX010001.MP4" not in files
        assert plan.source_dirs == []


def test_cli_dry_run_shows_sources_dir_and_size(tmp_path, capsys):
    cache = tmp_path / "cache"
    project = _drive(cache)
    out, stg = tmp_path / "out", tmp_path / "stg"
    out.mkdir()
    stg.mkdir()
    args = SimpleNamespace(project=None, confirm=False, min_age=0, archive_base=None)
    with (
        patch("gomoveshift.paths.gms_output_dir", return_value=out),
        patch("gomoveshift.paths.gms_staging_dir", return_value=stg),
        patch("gomoveshift.paths.gms_intermediate_base", return_value=cache),
    ):
        rc = cleanup_handler.cmd_cleanup(args, None)
    text = capsys.readouterr().out
    assert rc == 0
    assert f"SOURCES DIR: {_sources(project)}" in text
    assert (_sources(project) / "GX010001.MP4").exists()


def test_cli_finds_drive_projects_in_the_stage_cache(tmp_path, capsys):
    cache = tmp_path / "cache"
    _drive(cache)
    out, stg = tmp_path / "out", tmp_path / "stg"
    out.mkdir()
    stg.mkdir()
    args = SimpleNamespace(project=None, confirm=False, min_age=0, archive_base=None)
    with (
        patch("gomoveshift.paths.gms_output_dir", return_value=out),
        patch("gomoveshift.paths.gms_staging_dir", return_value=stg),
        patch("gomoveshift.paths.gms_intermediate_base", return_value=cache),
    ):
        rc = cleanup_handler.cmd_cleanup(args, None)
    text = capsys.readouterr().out
    assert rc == 0
    assert "WOULD DELETE" in text and "lf_01.mp4" in text


def test_cli_named_project_resolves_in_the_stage_cache(tmp_path, capsys):
    cache = tmp_path / "cache"
    project = _drive(cache)
    out, stg = tmp_path / "out", tmp_path / "stg"
    out.mkdir()
    stg.mkdir()
    args = SimpleNamespace(project=project.name, confirm=False, min_age=0, archive_base=None)
    with (
        patch("gomoveshift.paths.gms_output_dir", return_value=out),
        patch("gomoveshift.paths.gms_staging_dir", return_value=stg),
        patch("gomoveshift.paths.gms_intermediate_base", return_value=cache),
    ):
        rc = cleanup_handler.cmd_cleanup(args, None)
    assert rc == 0
    assert "episode_01_2026-03-05.mp4" in capsys.readouterr().out


def test_intermediate_base_honours_the_env_override(monkeypatch, tmp_path):
    from gomoveshift.paths import gms_intermediate_base

    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path))
    assert gms_intermediate_base() == tmp_path
    monkeypatch.delenv("GMS_INTERMEDIATE_DIR")
    assert gms_intermediate_base() == Path(os.path.expanduser("~/.cache/gomoveshift"))
