"""photos sort: parser, config, dispatch and admission fence."""

import fcntl
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from gomoveshift import config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import sort as ps


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _photo(path, seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 24, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_parser_accepts_flags_and_has_no_out():
    args = _build_parser().parse_args(
        "photos sort /tmp/x --dry-run --index-only --reprocess --workers 3 --no-stacks".split()
    )
    assert (args.photos_command, args.folder) == ("sort", "/tmp/x")
    assert args.dry_run and args.index_only and args.reprocess and args.no_stacks
    assert args.workers == 3
    defaults = _build_parser().parse_args("photos sort /tmp/x".split())
    assert defaults.workers == 2 and not defaults.dry_run
    with pytest.raises(SystemExit):
        _build_parser().parse_args("photos sort /tmp/x --out /tmp/y".split())
    with pytest.raises(SystemExit):
        _build_parser().parse_args("photos sort /tmp/x --gpu".split())


def test_config_default_and_override(tmp_path):
    assert PhotosConfig.model_fields["sort_root"].default == "/mnt/d/camPics_output"
    assert PhotosConfig().resolve_sort_root() == Path("/mnt/d/camPics_output")
    assert PhotosConfig(sort_root=str(tmp_path / "o")).resolve_sort_root() == tmp_path / "o"
    assert PhotosConfig(sort_root="~/x").resolve_sort_root() == Path("~/x").expanduser()


def test_busy_admission_returns_2_before_handler(tmp_path, monkeypatch, capsys):
    lock_path = tmp_path / "admission.lock"
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(lock_path))
    handler = Mock(return_value=0)
    monkeypatch.setattr(photos_handler, "_cmd_photos_sort", handler)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert photos_handler.cmd_photos(Namespace(photos_command="sort"), None) == 2
    handler.assert_not_called()
    assert "admission" in capsys.readouterr().err


def test_missing_folder_exits_2_without_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    _use_photos_config(monkeypatch, sort_root=str(tmp_path / "out"))
    spy = Mock()
    monkeypatch.setattr(ps, "develop_one", spy)
    assert _run(f"photos sort {tmp_path / 'nope'}") == 2
    spy.assert_not_called()
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_dry_run_through_cli_prints_counts_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    root = tmp_path / "out"
    _use_photos_config(monkeypatch, sort_root=str(root))
    nice = Mock()
    monkeypatch.setattr(ps.os, "nice", nice)
    folder = tmp_path / "card"
    _photo(folder / "a.jpg")
    assert _run(f"photos sort {folder} --dry-run --workers 1") == 0
    nice.assert_called_once_with(19)
    out = capsys.readouterr().out
    assert "estimate" in out.lower() and "other" in out
    assert not root.exists()


def test_resume_flag_reaches_run_sort(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    _use_photos_config(monkeypatch, sort_root=str(tmp_path / "out"))
    monkeypatch.setattr(ps.os, "nice", Mock())
    run_sort = Mock()
    monkeypatch.setattr(ps, "run_sort", run_sort)
    run_dir = tmp_path / "out" / "2026-11-09-card"
    assert _run(f"photos sort {tmp_path / 'card'} --resume {run_dir}") == 0
    assert run_sort.call_args.kwargs["resume"] == run_dir


def test_without_resume_run_sort_gets_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    _use_photos_config(monkeypatch, sort_root=str(tmp_path / "out"))
    monkeypatch.setattr(ps.os, "nice", Mock())
    run_sort = Mock()
    monkeypatch.setattr(ps, "run_sort", run_sort)
    assert _run(f"photos sort {tmp_path / 'card'}") == 0
    assert run_sort.call_args.kwargs["resume"] is None


def test_resort_flag_parses_reaches_run_sort_and_conflicts_with_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    _use_photos_config(monkeypatch, sort_root=str(tmp_path / "out"))
    monkeypatch.setattr(ps.os, "nice", Mock())
    run_sort = Mock()
    monkeypatch.setattr(ps, "run_sort", run_sort)
    run_dir = tmp_path / "out" / "2026-11-09-card"
    assert _run(f"photos sort {tmp_path / 'card'} --resort {run_dir}") == 0
    assert run_sort.call_args.kwargs["resort"] == run_dir
    assert run_sort.call_args.kwargs["resume"] is None
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            f"photos sort /tmp/x --resort {run_dir} --resume {run_dir}".split()
        )


def test_reprocess_help_says_re_develop():
    sort_p = next(
        a
        for action in _build_parser()._subparsers._group_actions
        for a in [action.choices.get("photos")]
        if a
    )
    text = sort_p._subparsers._group_actions[0].choices["sort"].format_help()
    assert "Re-develop" in text and "--resort" in text
