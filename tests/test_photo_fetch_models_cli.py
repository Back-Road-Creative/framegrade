"""photos fetch-models: the one explicit network step for the aesthetic weights."""

from pathlib import Path
from unittest.mock import Mock

from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos import aesthetic, sort


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_parser_has_fetch_models_subcommand():
    assert (
        _build_parser().parse_args("photos fetch-models".split()).photos_command == "fetch-models"
    )


def test_success_prints_both_paths_and_exits_0(monkeypatch, capsys):
    monkeypatch.setattr(aesthetic, "fetch_models", lambda: (Path("/m/v.onnx"), Path("/m/h.pth")))
    assert _run("photos fetch-models") == 0
    out = capsys.readouterr().out
    assert "/m/v.onnx" in out and "/m/h.pth" in out


def test_unavailable_prints_message_and_exits_1(monkeypatch, capsys):
    def boom():
        raise aesthetic.AestheticModelUnavailable("sha mismatch")

    monkeypatch.setattr(aesthetic, "fetch_models", boom)
    assert _run("photos fetch-models") == 1
    assert "sha mismatch" in capsys.readouterr().err


def test_fetch_models_needs_no_media_admission(monkeypatch):
    monkeypatch.setattr(aesthetic, "fetch_models", lambda: (Path("v"), Path("h")))
    admission = Mock(side_effect=AssertionError("admission taken"))
    monkeypatch.setattr("gomoveshift.pipeline.media_admission.media_admission", admission)
    assert _run("photos fetch-models") == 0
    admission.assert_not_called()


def test_sort_cli_runs_with_the_real_scoring_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_NIGHT_ADMISSION_LOCK", str(tmp_path / "admission.lock"))
    seen = {}

    def fake_run_sort(folder, root, options, hooks=None, **kw):
        seen["hooks"] = hooks

    monkeypatch.setattr(sort, "run_sort", fake_run_sort)
    assert _run(f"photos sort {tmp_path}") == 0
    assert seen["hooks"].score is sort.score_shot
    assert seen["hooks"].aesthetic_factory is sort.make_aesthetic_scorer
