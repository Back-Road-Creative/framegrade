"""``photos.site_root`` has no default: the photo commands that need it say so.

An unset ``photos.site_root`` must surface as one readable line on stderr
naming the key, with a non-zero exit, never as a traceback and never as a
silent fallback to some other directory.
"""

from unittest.mock import patch

import pytest

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, ConfigError
from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.pipeline.photos.selector import PhotoSelectionState


@pytest.mark.parametrize(
    "argv",
    [
        "photos status",
        "photos generate --week 2000-01-03 --reset-selection-state",
    ],
)
def test_unset_site_root_is_reported_not_configured(argv, monkeypatch, tmp_path, capsys):
    config = Config()
    assert not config.photos.site_root
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: config)
    monkeypatch.delenv("GMS_PHOTO_OUTPUT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    rc = photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), config)

    assert rc != 0
    assert "photos.site_root is not configured" in capsys.readouterr().err


@pytest.mark.parametrize("argv", ["photos status"])
def test_unset_site_root_exits_with_the_config_error_code(argv, monkeypatch, tmp_path, capsys):
    """``ConfigError`` maps to exit 2 on the CLI, as for the other handlers."""
    config = Config()
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: config)
    monkeypatch.delenv("GMS_PHOTO_OUTPUT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    rc = photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), config)

    assert rc == 2
    assert "photos.site_root is not configured" in capsys.readouterr().err


def test_release_needs_no_site_root(monkeypatch, tmp_path, capsys):
    """``photos release <week>`` only rewrites the selection state, so it runs
    with the site unset and says nothing about ``photos.site_root``."""
    config = Config()
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: config)
    monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    keys = ["journey-a/photo-001.jpg", "journey-a/photo-002.jpg"]
    state_path = tmp_path / ".photo_selection_state.json"
    PhotoSelectionState(posted_photos=list(keys), pending_photos=list(keys)).save(state_path)
    CaptionBatch(
        week_start="2026-01-05",
        voice_style="first_person",
        captions=[
            PhotoCaption(
                photo_index=i,
                main_caption="x",
                tiktok_caption="x",
                hashtags=[],
                journey_slug="journey-a",
                filename=f"photo-00{i}.jpg",
            )
            for i in (1, 2)
        ],
    ).save(tmp_path / "photo_captions" / "2026-01-05_captions.yaml")

    rc = photos_handler.cmd_photos(
        _build_parser().parse_args("photos release 2026-01-05".split()), config
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "photos.site_root" not in captured.err
    state = PhotoSelectionState.load(state_path)
    assert state.pending_photos == []
    assert state.posted_photos == []


def test_release_config_error_exits_2_naming_the_key(monkeypatch, tmp_path, capsys):
    """A ``ConfigError`` out of the release path maps to exit 2 with the key on stderr."""
    config = Config()
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: config)
    monkeypatch.setenv("GMS_PHOTO_OUTPUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    with patch.object(
        PhotoOrchestrator,
        "release_batch",
        side_effect=ConfigError("photos.site_root is not configured"),
    ):
        rc = photos_handler.cmd_photos(
            _build_parser().parse_args("photos release 2026-01-05".split()), config
        )

    assert rc == 2
    assert "photos.site_root" in capsys.readouterr().err
