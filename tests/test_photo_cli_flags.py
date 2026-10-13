"""Operator escape flags and CLI gaps for the photo pipeline.

Covers the plumbing: argparse surface -> handler -> orchestrator -> callee.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos.caption_generator import CaptionBatch
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.pipeline.photos.voice_styles import VoiceStyle
from gomoveshift.publishing.photo_run_hooks import photo_run_hooks

JOURNEY_MD = """\
---
title: Journey A
days:
- day_number: 1
  date: '2025-01-01'
  photos:
  - location: Place A
    photos:
    - photo-001.jpg
    - photo-002.jpg
---
Content
"""


def _parse(argv):
    return _build_parser().parse_args(argv.split())


def _run_handler(args, orchestrator):
    with (
        patch(
            "gomoveshift.pipeline.photos.orchestrator.PhotoOrchestrator", return_value=orchestrator
        ),
        patch.object(photos_handler, "configure_logging"),
    ):
        return photos_handler.cmd_photos(args, None)


class TestPhotoParserFlags:
    def test_generate_escape_flags(self):
        default = _parse("photos generate")
        assert default.allow_photo_count is False
        assert default.overwrite is False

        args = _parse("photos generate --allow-photo-count --overwrite")
        assert args.allow_photo_count is True
        assert args.overwrite is True

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_generate_count_below_one_rejected(self, bad, capsys):
        """A non-positive --count must die at argparse — it used to reach the
        selector, whose empty selection wiped the posted-photos ledger."""
        with pytest.raises(SystemExit):
            _parse(f"photos generate --count {bad}")
        assert "--count must be >= 1" in capsys.readouterr().err

    def test_generate_count_valid(self):
        assert _parse("photos generate").count == 7
        assert _parse("photos generate --count 3").count == 3

    def test_publish_start_and_end_date(self):
        default = _parse("photos publish --captions c.yaml")
        assert default.start is None
        assert default.end_date is None

        args = _parse("photos publish --captions c.yaml --start 2026-08-10 --end-date 2026-08-17")
        assert args.start == "2026-08-10"
        assert args.end_date == "2026-08-17"

    @pytest.mark.parametrize(
        "argv",
        [
            "photos generate --voice interactive",
            "photos batch --start 2026-08-01 --end 2026-08-07 --voice interactive",
        ],
    )
    def test_interactive_voice_is_selectable(self, argv):
        assert _parse(argv).voice == "interactive"


class TestPhotoWeekValidation:
    """--week is used verbatim as a path component (captions YAML, resized
    dir, Metricool CSV name) — anything but a strict zero-padded YYYY-MM-DD
    must die at argparse, or ``--week ../../x`` writes outside the output tree.
    """

    def test_week_default_is_empty_meaning_today(self):
        assert _parse("photos generate").week == ""

    def test_week_valid_date_accepted(self):
        assert _parse("photos generate --week 2026-07-22").week == "2026-07-22"

    @pytest.mark.parametrize("bad", ["../../x", "2026-7-2", "2026-02-30", "notadate"])
    def test_week_invalid_rejected(self, bad, capsys):
        with pytest.raises(SystemExit):
            _parse(f"photos generate --week {bad}")
        assert "--week must be YYYY-MM-DD" in capsys.readouterr().err


def _action_help(subcommand: str, dest: str) -> str:
    """Help string argparse would print for one ``photos <subcommand>`` flag."""

    def _sub(parser):
        return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))

    cmd = _sub(_sub(_build_parser()).choices["photos"]).choices[subcommand]
    return next(a for a in cmd._actions if a.dest == dest).help


class TestNoFacesHelpIsAccurate:
    """--no-faces no longer means "a face skips the photo".

    Blur moved to the resize stage, so with blur on (the default) a face is
    blurred rather than withheld and only an unreadable photo is skipped; the
    detect-and-skip behaviour survives solely under --skip-blur. The help text
    has to say so, or the flag reads as doing something it stopped doing.
    """

    @pytest.mark.parametrize("subcommand", ["generate", "batch"])
    def test_help_names_the_skip_blur_condition(self, subcommand):
        assert "--skip-blur" in _action_help(subcommand, "no_faces")

    @pytest.mark.parametrize("subcommand", ["generate", "batch"])
    def test_help_does_not_advertise_the_retired_haar_detector(self, subcommand):
        assert "haar" not in _action_help(subcommand, "no_faces").lower()


def _gen_args(**overrides):
    args = SimpleNamespace(
        photos_command="generate",
        week="2026-08-01",
        count=7,
        voice=None,
        no_faces=False,
        allow_photo_count=False,
        overwrite=False,
    )
    args.__dict__.update(overrides)
    return args


def _gen_orchestrator(caption_dir):
    orchestrator = MagicMock()
    orchestrator.caption_gen.caption_dir = caption_dir
    orchestrator.generate_batch.return_value = SimpleNamespace(
        captions=[], voice_style="voice_interactive", week_start="2026-08-01"
    )
    return orchestrator


class TestPhotoHandlerPlumbing:
    def test_generate_forwards_escapes_and_interactive_voice(self, tmp_path):
        orchestrator = _gen_orchestrator(tmp_path)
        args = _gen_args(voice="interactive", allow_photo_count=True, overwrite=True)

        assert _run_handler(args, orchestrator) == 0
        kwargs = orchestrator.generate_batch.call_args.kwargs
        assert kwargs["allow_shortfall"] is True
        assert kwargs["overwrite"] is True
        assert kwargs["voice_style"] is VoiceStyle.INTERACTIVE

    def test_no_faces_still_reaches_the_selection_gate(self, tmp_path):
        """The flag is kept, not retired: it is the only switch that arms the
        withhold gate, and under --skip-blur the only detect-and-skip left."""
        orchestrator = _gen_orchestrator(tmp_path)

        assert _run_handler(_gen_args(no_faces=True), orchestrator) == 0
        assert orchestrator.generate_batch.call_args.kwargs["filter_faces"] is True

    def test_generate_hint_prints_resolved_caption_path(self, tmp_path, capsys):
        caption_dir = tmp_path / "photo_captions"

        assert _run_handler(_gen_args(), _gen_orchestrator(caption_dir)) == 0
        assert str(caption_dir / "2026-08-01_captions.yaml") in capsys.readouterr().out

    def test_publish_forwards_start_and_end_date(self):
        orchestrator = MagicMock()
        orchestrator.publish_batch.return_value = Path("out.csv")
        args = SimpleNamespace(
            photos_command="publish", captions="c.yaml", force=False, start="2026-08-10"
        )
        args.end_date = "2026-08-17"

        assert _run_handler(args, orchestrator) == 0
        kwargs = orchestrator.publish_batch.call_args.kwargs
        assert kwargs["start_date"] == "2026-08-10"
        assert kwargs["end_date"] == "2026-08-17"

    def test_publish_does_not_claim_the_post_went_live(self, capsys):
        """`photos publish` writes a Metricool CSV. Metricool is an external
        scheduler: nothing in ``_publish_batch`` calls a platform and nothing
        returns a post ID, so the batch is exported for scheduling, not
        published. The handler said "Published!" — the one word an operator
        would read as "it is on Instagram now".

        This matters beyond wording: ``publishing/dispatch.py`` counts
        ``result.published`` for genuine platform publishes through registered
        handlers. Two different states cannot share the word, or the photo
        route borrows the credibility of the route that really posts.
        """
        orchestrator = MagicMock()
        orchestrator.publish_batch.return_value = Path("out.csv")
        args = SimpleNamespace(photos_command="publish", captions="c.yaml", force=False, start=None)
        args.end_date = None

        assert _run_handler(args, orchestrator) == 0
        out = capsys.readouterr().out

        assert "Published!" not in out
        assert "out.csv" in out, "the operator still needs the CSV path"
        assert "scheduling" in out.lower(), "say what actually happened"
        # The absence of a receipt is the point, so it is stated, not implied.
        assert "receipt" in out.lower()


@pytest.fixture
def orchestrator(tmp_path):
    """Orchestrator over a site root holding exactly two photos.

    ``caption_gen.generate_batch`` is stubbed with a stand-in that keeps the
    real generator's refuse-to-clobber guard but never calls an LLM.
    """
    journeys = tmp_path / "site" / "content" / "journeys"
    journeys.mkdir(parents=True)
    (journeys / "journey-a.md").write_text(JOURNEY_MD)
    img_dir = tmp_path / "site" / "static" / "images" / "journeys" / "journey-a"
    img_dir.mkdir(parents=True)
    for name in ["photo-001.jpg", "photo-002.jpg"]:
        (img_dir / name).write_bytes(b"fake jpg")

    orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path / "out")

    def _generate(selections, voice_style, week_start=""):
        path = orch.caption_gen.caption_dir / f"{week_start}_captions.yaml"
        if path.exists():
            raise FileExistsError(f"Caption review file already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new: true\n")
        return SimpleNamespace(captions=list(selections), week_start=week_start)

    orch.caption_gen.generate_batch = _generate
    return orch


def _seed_existing(orchestrator):
    caption_dir = orchestrator.caption_gen.caption_dir
    caption_dir.mkdir(parents=True, exist_ok=True)
    path = caption_dir / "2026-08-01_captions.yaml"
    path.write_text("old: true\n")
    return path


def _generate(orchestrator, **kwargs):
    return orchestrator.generate_batch(
        week_start="2026-08-01", voice_style=VoiceStyle.FIRST_PERSON, **kwargs
    )


class TestOrchestratorEscapeFlags:
    def test_shortfall_raises_without_flag(self, orchestrator):
        with pytest.raises(RuntimeError, match="PHOTO-CONTRACT"):
            _generate(orchestrator, count=5)

    def test_shortfall_accepted_with_flag(self, orchestrator):
        batch = _generate(orchestrator, count=5, allow_shortfall=True)
        assert len(batch.captions) == 2

    def test_existing_review_file_blocks_regenerate(self, orchestrator):
        _seed_existing(orchestrator)
        with pytest.raises(FileExistsError):
            _generate(orchestrator, count=2)

    def test_overwrite_moves_existing_file_aside(self, orchestrator):
        path = _seed_existing(orchestrator)
        _generate(orchestrator, count=2, overwrite=True)

        assert path.read_text() == "new: true\n"
        backups = list(path.parent.glob("2026-08-01_captions.yaml.bak-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "old: true\n"


def test_publish_batch_start_date_overrides_week_start(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_EXPORT_DIR", str(tmp_path / "nonexistent"))
    from gomoveshift.pipeline.photos.caption_generator import PhotoCaption
    from gomoveshift.pipeline.photos.scheduler import PhotoSlot

    batch = CaptionBatch(
        week_start="2026-08-01",
        voice_style="voice_first_person",
        captions=[PhotoCaption(0, "Caption", "T", ["t"], "journey-a", "photo-001.jpg", "Place")],
        reviewed=True,
    )
    yaml_path = tmp_path / "c.yaml"
    batch.save(yaml_path)
    # The count-match gate requires the caption's source file on disk.
    site_root = tmp_path / "site"
    photo_dir = site_root / "static" / "images" / "journeys" / "journey-a"
    photo_dir.mkdir(parents=True)
    (photo_dir / "photo-001.jpg").write_bytes(b"fake photo")

    with (
        patch("gomoveshift.pipeline.photos.orchestrator.PipelineUploader") as uploader_cls,
        patch("gomoveshift.publishing.photo_run_hooks.record_confirmed_upload") as _book,
    ):
        uploader_cls.return_value.verify_uploads.return_value = SimpleNamespace(failed=[])

        orch = PhotoOrchestrator(
            site_root=site_root, output_dir=tmp_path / "out", **photo_run_hooks()
        )
        orch.resizer.resize_batch = MagicMock(return_value=[])
        orch.scheduler.schedule_batch = MagicMock(
            return_value=[PhotoSlot(date="2026-08-10", time="10:00:00", day_of_week="Monday")]
        )
        orch.csv_gen.generate = MagicMock(return_value=tmp_path / "out.csv")
        orch._record_theme_usage = MagicMock()

        orch.publish_batch(str(yaml_path), start_date="2026-08-10", end_date="2026-08-17")

    orch.scheduler.schedule_batch.assert_called_once_with(
        count=1, start_date="2026-08-10", end_date="2026-08-17"
    )


@pytest.mark.parametrize(
    "argv,expected",
    [
        ("photos generate --skip-blur", False),
        ("photos generate", True),
        ("photos batch --start 2026-08-01 --end 2026-08-07 --skip-blur", False),
        ("photos publish --captions c.yaml --skip-blur", False),
    ],
)
def test_skip_blur_flag_reaches_the_blur_setting(argv, expected, monkeypatch):
    """--skip-blur is the operator opt-out; blur stays ENABLED without it."""
    from gomoveshift.pipeline.photos import face_filter

    monkeypatch.setattr(face_filter, "_blur_enabled", not expected)
    _run_handler(_parse(argv), MagicMock())


@pytest.mark.parametrize("argv, expected", [("", False), (" --lens-correct", True)])
def test_lens_correct_flag_reaches_develop_options(argv, expected, tmp_path, monkeypatch):
    """--lens-correct is run-owned, like --straighten/--crop: off unless passed."""
    import numpy as np
    from PIL import Image

    import gomoveshift.config as config_module
    from gomoveshift.config import Config, PhotosConfig
    from gomoveshift.pipeline.photos import develop as dev

    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda *a, **k: Config(photos=PhotosConfig(developed_output_dir=str(tmp_path / "out"))),
    )
    photo = tmp_path / "IMG_9001.jpg"
    rng = np.random.default_rng(4)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(photo, "JPEG")

    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev, "develop_variants", lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1]
    )
    args = _parse(f"photos develop {photo} --styles natural{argv}")
    assert photos_handler.cmd_photos(args, None) == 0
    assert [o.lens_correct for o in seen] == [expected]
