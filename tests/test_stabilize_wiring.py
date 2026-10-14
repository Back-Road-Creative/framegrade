"""--stabilize / --deflicker wired into the day master, opt-in and measured.

Every subprocess / ffmpeg call is mocked.
"""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import stabilize
from gomoveshift.pipeline.stabilize import FfmpegFilterCapabilities


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


def _caps(*names):
    return FfmpegFilterCapabilities(frozenset(names))


def test_stabilize_refused_without_libvidstab():
    from gomoveshift.pipeline.preflight import check_picture_filters

    with pytest.raises(RuntimeError) as exc:
        check_picture_filters(stabilize=True, deflicker=False, capabilities=_caps("deflicker"))
    assert stabilize.VIDSTAB_MISSING_REASON in str(exc.value)
    # deflicker set with deflicker present passes; vidstab-only build passes stabilize
    check_picture_filters(stabilize=False, deflicker=True, capabilities=_caps("deflicker"))
    check_picture_filters(
        stabilize=True,
        deflicker=False,
        capabilities=_caps("vidstabdetect", "vidstabtransform"),
    )
    with pytest.raises(RuntimeError, match="deflicker"):
        check_picture_filters(stabilize=False, deflicker=True, capabilities=_caps("scale"))


def test_no_flag_probes_nothing(monkeypatch):
    from gomoveshift.pipeline.preflight import check_picture_filters

    def boom(*a, **k):
        raise AssertionError("probed")

    monkeypatch.setattr(stabilize, "probe_ffmpeg_filters", boom)
    check_picture_filters(stabilize=False, deflicker=False)


def _batch(tmp_path, date="2026-02-20"):
    from gomoveshift.pipeline.day_grouper import DayBatch

    f = tmp_path / "ep.mp4"
    f.touch()
    ep = MagicMock()
    ep.index = 1
    ep.creation_time = datetime(2026, 2, 20, 10, tzinfo=timezone.utc)
    ep.total_duration = 1800.0
    ep.concat_path = f
    ep.gpx_path = None
    return DayBatch(date=date, episodes=[ep])


def _run_encode(tmp_path, monkeypatch, cfg, *, jitters=None):
    """encode_day_master with ffmpeg mocked; returns (argvs, process_inputs, out_dir)."""
    from gomoveshift.pipeline.day_master_encoder import encode_day_master

    argvs: list = []

    def fake_run(cmd, *a, **k):
        cmd = list(cmd)
        argvs.append(cmd)
        if cmd and str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_bytes(b"x")
        r = MagicMock()
        r.returncode = 0
        r.stdout = b""
        r.stderr = ""
        return r

    for mod in ("day_master_encoder", "stabilize"):
        monkeypatch.setattr(f"gomoveshift.pipeline.{mod}.subprocess.run", fake_run)
    if jitters is not None:
        it = iter(jitters)
        monkeypatch.setattr(stabilize, "measure_luma_jitter", lambda *a, **k: next(it))
    inputs: list = []

    def fake_process(self, video_path, output_path, location, **kw):
        inputs.append(Path(video_path))
        Path(output_path).write_bytes(b"x")
        res = MagicMock()
        res.success = True
        return res

    monkeypatch.setattr(
        "gomoveshift.pipeline.single_pass.SinglePassProcessor.process", fake_process
    )
    out_dir = tmp_path / "out"
    encode_day_master(_batch(tmp_path), out_dir, config=cfg)
    return argvs, inputs, out_dir


def test_stabilize_applied_only_when_flag_set(tmp_path, monkeypatch):
    built: list = []
    real = stabilize.build_deflicker_argv

    def spy(*a, **k):
        built.append(a)
        return real(*a, **k)

    monkeypatch.setattr(stabilize, "build_deflicker_argv", spy)
    _, inputs, _ = _run_encode(tmp_path, monkeypatch, Config())
    assert built == []
    assert inputs[0].name == "day_2026-02-20_raw.mp4"

    cfg = Config()
    cfg.drive.deflicker = True
    (tmp_path / "b").mkdir()
    argvs, inputs, out_dir = _run_encode(tmp_path / "b", monkeypatch, cfg, jitters=[2.0, 1.0])
    assert len(built) == 1
    assert any("deflicker" in " ".join(map(str, a)) for a in argvs)
    assert inputs[0].name != "day_2026-02-20_raw.mp4"  # graded the deflickered file
    assert not inputs[0].exists()  # intermediate cleaned
    rec = json.loads((out_dir / "day_2026-02-20_picture_steps.json").read_text())
    assert rec["deflicker_applied"] is True and rec["stabilize_applied"] is False


def test_deflicker_discarded_when_worse(tmp_path, monkeypatch, caplog):
    cfg = Config()
    cfg.drive.deflicker = True
    with caplog.at_level("WARNING"):
        _, inputs, out_dir = _run_encode(tmp_path, monkeypatch, cfg, jitters=[1.0, 3.0])
    assert inputs[0].name == "day_2026-02-20_raw.mp4"  # input used
    assert not list(out_dir.glob("*deflicker*"))  # pass output removed
    rec = json.loads((out_dir / "day_2026-02-20_picture_steps.json").read_text())
    assert rec["deflicker_applied"] is False
    assert rec["steps"]["deflicker"]["kept"] is False
    assert rec["steps"]["deflicker"]["input_jitter"] == 1.0
    assert rec["steps"]["deflicker"]["output_jitter"] == 3.0
    assert "1.0000" in caplog.text and "3.0000" in caplog.text


def test_flags_resume_sentinel():
    from gomoveshift.cli.handlers.drive import apply_picture_flags
    from gomoveshift.cli.parser import _build_parser

    cfg = Config()
    cfg.drive.stabilize = True
    cfg.drive.deflicker = True
    apply_picture_flags(cfg, Namespace(stabilize=None, deflicker=None))
    assert cfg.drive.stabilize is True and cfg.drive.deflicker is True
    fresh = Config()
    apply_picture_flags(fresh, Namespace(stabilize=True, deflicker=None))
    assert fresh.drive.stabilize is True and fresh.drive.deflicker is False
    # parser default is the unset sentinel
    ns = _build_parser().parse_args(["drive", "/tmp/x"])
    assert ns.stabilize is None and ns.deflicker is None


def test_flicker_score():
    import numpy as np

    from gomoveshift.pipeline.quality import flicker_score

    flat = [np.full((4, 4), 100, np.uint8)] * 3
    assert flicker_score(flat) == 0.0
    alt = [np.full((4, 4), v, np.uint8) for v in (100, 110, 100)]
    assert flicker_score(alt) == pytest.approx(10.0)
    assert flicker_score(flat[:1]) == 0.0


def _steps_record(d: Path, date: str, **flags) -> Path:
    cache = d / f"day_{date}_color_cache.json"  # deliberately never written
    rec = {"deflicker_applied": False, "stabilize_applied": False, "steps": {}, **flags}
    (d / f"day_{date}_picture_steps.json").write_text(json.dumps(rec))
    return cache


def test_stabilised_day_counts_as_picture_changed(tmp_path):
    from types import SimpleNamespace

    from gomoveshift.pipeline.quality import picture_changed_for_day, picture_inputs_from_state

    cache = _steps_record(tmp_path, "2026-02-20", stabilize_applied=True)
    assert picture_changed_for_day(cache) is True
    art = SimpleNamespace(
        date="2026-02-20",
        master_path=tmp_path / "day_2026-02-20_master.mp4",
        master_record={},
        regular_lf_indices=[3],
    )
    _, changed = picture_inputs_from_state(SimpleNamespace(day_artifacts=[art]))
    assert changed == {3: True}


def test_discarded_steps_do_not_count(tmp_path):
    from gomoveshift.pipeline.quality import picture_changed_for_day

    cache = _steps_record(tmp_path, "2026-02-21")
    assert picture_changed_for_day(cache) is False


def test_only_a_literal_true_flag_turns_a_step_on():
    """A stand-in args object (MagicMock) answers every attribute with a truthy
    mock; that must not read as ``--stabilize``/``--deflicker`` supplied."""
    from gomoveshift.cli.handlers.drive import apply_picture_flags

    cfg = Config()
    apply_picture_flags(cfg, MagicMock())
    assert cfg.drive.stabilize is False and cfg.drive.deflicker is False
