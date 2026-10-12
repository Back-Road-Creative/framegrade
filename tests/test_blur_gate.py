"""Privacy-blur hard gate.

The gate guarantees a blur-enabled run can never ship an unblurred master:
the mask is repopulated on resume, a missing mask hard-fails before the encode,
and a durable marker proves a resume-accepted FINAL was actually blurred.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.pipeline import blur_gate, ost_audit_sidecar
from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage


def _ctx(tmp_path, *, blur_enabled=True, skip_blur=False, mask_path=None, make_mask=False):
    stages = tmp_path / "_stages"
    stages.mkdir(exist_ok=True)
    if make_mask:
        mask_path = stages / "proj_blur_mask.mkv"
        mask_path.write_bytes(b"\x00" * 32)
    return SimpleNamespace(
        config=SimpleNamespace(
            drive=SimpleNamespace(blur_enabled=blur_enabled, skip_blur=skip_blur)
        ),
        blur_mask_path=mask_path,
        paths=SimpleNamespace(stages_dir=stages),
        project_name="proj",
    )


class TestBlurRequired:
    @pytest.mark.parametrize(
        "enabled,skip,expected",
        [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
    )
    def test_matrix(self, enabled, skip, expected):
        cfg = SimpleNamespace(drive=SimpleNamespace(blur_enabled=enabled, skip_blur=skip))
        assert blur_gate.blur_required(cfg) is expected


class TestMasterBlurGate:
    """Drive-path day-master gate (reused by ``day_master_encoder``)."""

    def _cfg(self, *, enabled=True, skip=False):
        return SimpleNamespace(drive=SimpleNamespace(blur_enabled=enabled, skip_blur=skip))

    def test_proven_only_with_marker(self, tmp_path):
        master = tmp_path / "day_2026-02-24_master.mp4"
        master.write_bytes(b"m")
        assert blur_gate.master_blur_proven(self._cfg(), master) is False
        blur_gate.mark_applied(master)
        assert blur_gate.master_blur_proven(self._cfg(), master) is True

    def test_require_raises_without_marker(self, tmp_path):
        master = tmp_path / "day_2026-02-24_master.mp4"
        master.write_bytes(b"m")
        with pytest.raises(blur_gate.BlurGateError, match="no .blurred marker"):
            blur_gate.require_master_blurred(self._cfg(), master)

    def test_require_passes_with_marker(self, tmp_path):
        master = tmp_path / "day_2026-02-24_master.mp4"
        master.write_bytes(b"m")
        blur_gate.mark_applied(master)
        blur_gate.require_master_blurred(self._cfg(), master)  # no raise

    def test_no_op_when_blur_not_required(self, tmp_path):
        master = tmp_path / "day_2026-02-24_master.mp4"
        master.write_bytes(b"m")  # no marker
        # skip_blur / blur disabled → gate is a no-op even without a marker
        blur_gate.require_master_blurred(self._cfg(skip=True), master)
        blur_gate.require_master_blurred(self._cfg(enabled=False), master)
        assert blur_gate.master_blur_proven(self._cfg(skip=True), master) is True


class TestResolveMask:
    def test_keeps_existing(self, tmp_path):
        ctx = _ctx(tmp_path, mask_path=Path("/already/set.mkv"))
        assert blur_gate.resolve_mask(ctx) == Path("/already/set.mkv")

    def test_repopulates_from_disk_on_resume(self, tmp_path):
        ctx = _ctx(tmp_path, mask_path=None, make_mask=False)
        # simulate resume: stage didn't run, but the mask file is on disk
        (ctx.paths.stages_dir / "proj_blur_mask.mkv").write_bytes(b"\x00" * 16)
        resolved = blur_gate.resolve_mask(ctx)
        assert resolved == ctx.paths.stages_dir / "proj_blur_mask.mkv"
        assert ctx.blur_mask_path == resolved  # mutated for the encode to consume

    def test_stays_none_when_absent(self, tmp_path):
        ctx = _ctx(tmp_path, mask_path=None)
        assert blur_gate.resolve_mask(ctx) is None


class TestCheckMaskReady:
    def test_passes_when_not_required(self, tmp_path):
        ctx = _ctx(tmp_path, skip_blur=True, mask_path=None)
        assert blur_gate.check_mask_ready(ctx) is None

    def test_passes_with_mask(self, tmp_path):
        ctx = _ctx(tmp_path, make_mask=True)
        assert blur_gate.check_mask_ready(ctx) is None

    def test_fails_when_required_and_mask_missing(self, tmp_path):
        ctx = _ctx(tmp_path, mask_path=None)
        err = blur_gate.check_mask_ready(ctx)
        assert err and "unblurred" in err
        # Must say this command does NOT offer --skip-blur (a bare substring
        # check passes equally on a message claiming the opposite), and name
        # the real escape hatch instead.
        assert "no --skip-blur flag" in err
        assert "drive.skip_blur: true in config.yaml" in err

    def test_fails_on_empty_mask(self, tmp_path):
        empty = tmp_path / "_stages" / "proj_blur_mask.mkv"
        empty.parent.mkdir(exist_ok=True)
        empty.write_bytes(b"")
        ctx = _ctx(tmp_path, mask_path=empty)
        assert blur_gate.check_mask_ready(ctx) is not None


class TestMarker:
    def test_mark_then_check_passes(self, tmp_path):
        final = tmp_path / "proj_FINAL.mp4"
        final.write_bytes(b"\x00")
        blur_gate.mark_applied(final)
        assert (tmp_path / "proj_FINAL.mp4.blurred").exists()
        ctx = _ctx(tmp_path)
        assert blur_gate.check_applied(ctx, final) is None

    def test_check_fails_without_marker(self, tmp_path):
        final = tmp_path / "proj_FINAL.mp4"
        final.write_bytes(b"\x00")
        ctx = _ctx(tmp_path)  # blur required, no marker
        err = blur_gate.check_applied(ctx, final)
        assert err and "no blur marker" in err
        # Same polarity requirement as check_mask_ready's message: this
        # command does NOT offer --skip-blur, and the real escape hatch
        # (drive.skip_blur: true in config.yaml) is named.
        assert "no --skip-blur flag" in err
        assert "drive.skip_blur: true in config.yaml" in err

    def test_check_passes_when_not_required(self, tmp_path):
        final = tmp_path / "proj_FINAL.mp4"
        final.write_bytes(b"\x00")
        ctx = _ctx(tmp_path, blur_enabled=False)  # no marker, but blur off
        assert blur_gate.check_applied(ctx, final) is None


def _cabin_ctx(tmp_path, *, cabin_frac=0.6, skip_blur=False):
    stages = tmp_path / "_stages"
    stages.mkdir(exist_ok=True)
    mask = stages / "proj_blur_mask.mkv"
    mask.write_bytes(b"\x00" * 32)
    drive = SimpleNamespace(
        blur_enabled=True,
        skip_blur=skip_blur,
        blur_vehicle_cabin_frac=cabin_frac,
        blur_vehicle_conf=0.6,
        blur_vehicle_min_height_frac=0.07,
        blur_vehicle_ignore_bottom_frac=0.12,
        blur_plate_max_area_frac=0.006,
        blur_cabin_gate_sample_fps=4.0,
        blur_cabin_gate_min_coverage=0.5,
    )
    return SimpleNamespace(
        config=SimpleNamespace(drive=drive),
        blur_mask_path=mask,
        paths=SimpleNamespace(stages_dir=stages),
        project_name="proj",
        current_video=tmp_path / "in.mp4",
    )


class TestCheckCabinCoverage:
    def test_noop_when_cabin_off(self, tmp_path):
        assert blur_gate.check_cabin_coverage(_cabin_ctx(tmp_path, cabin_frac=0.0)) is None

    def test_noop_when_skip_blur(self, tmp_path):
        assert blur_gate.check_cabin_coverage(_cabin_ctx(tmp_path, skip_blur=True)) is None

    def test_noop_when_model_unresolvable(self, tmp_path, monkeypatch):
        from gomoveshift import analysis

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: None)
        # detection stage owns the missing-model hard-fail; the gate is a no-op
        assert blur_gate.check_cabin_coverage(_cabin_ctx(tmp_path)) is None

    def test_passes_when_no_leaks(self, tmp_path, monkeypatch):
        import redactcam

        from gomoveshift import analysis

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        monkeypatch.setattr(
            analysis,
            "verify_cabin_coverage",
            lambda *a, **k: redactcam.CoverageReport(vehicle_frames=5, leaks=[]),
        )
        assert blur_gate.check_cabin_coverage(_cabin_ctx(tmp_path)) is None

    def test_fails_on_exposed_cabin(self, tmp_path, monkeypatch):
        import redactcam

        from gomoveshift import analysis

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        rep = redactcam.CoverageReport(
            vehicle_frames=5, leaks=[redactcam.Leak(100, 0.0, (1, 2, 3, 4))]
        )
        monkeypatch.setattr(analysis, "verify_cabin_coverage", lambda *a, **k: rep)
        err = blur_gate.check_cabin_coverage(_cabin_ctx(tmp_path))
        assert err and "EXPOSED" in err and "--skip-blur" in err


class TestCabinLeakReport:
    """The error string names 5 example leaks. Recovering from a real failure needs
    ALL of them: re-deriving the list costs a multi-hour re-scan of the master
    (13 leaks reported 5 at a time). The report the gate already
    computed is persisted instead of discarded."""

    @staticmethod
    def _stub(monkeypatch, leaks):
        import redactcam

        from gomoveshift import analysis

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        rep = redactcam.CoverageReport(vehicle_frames=552, leaks=leaks)
        monkeypatch.setattr(analysis, "verify_cabin_coverage", lambda *a, **k: rep)

    def test_writes_every_leak_not_just_the_five_examples(self, tmp_path, monkeypatch):
        import redactcam

        leaks = [redactcam.Leak(i * 1000, i / 100, (i, i, 40, 30)) for i in range(1, 14)]
        self._stub(monkeypatch, leaks)
        config, video, mask = _cabin_cfg(tmp_path)
        err = blur_gate.check_day_master_cabin_coverage(config, video, mask)
        assert err
        report = blur_gate.cabin_leak_report_path(mask)
        assert report.exists(), "the full leak list must survive the run that found it"
        data = json.loads(report.read_text())
        assert data["vehicle_frames"] == 552
        assert [leak["frame"] for leak in data["leaks"]] == [i * 1000 for i in range(1, 14)]
        assert data["leaks"][0]["box"] == [1, 1, 40, 30]
        assert report.name in err, "the operator must be told where the full list is"

    def test_clears_a_stale_report_when_the_gate_passes(self, tmp_path, monkeypatch):
        import redactcam

        config, video, mask = _cabin_cfg(tmp_path)
        stale = blur_gate.cabin_leak_report_path(mask)
        stale.write_text('{"leaks": [{"frame": 999}]}')
        self._stub(monkeypatch, [])
        assert blur_gate.check_day_master_cabin_coverage(config, video, mask) is None
        assert not stale.exists(), "a passing gate must not leave a failing run's leak list"


def _cabin_cfg(tmp_path, *, cabin_frac=0.6, skip_blur=False):
    """Drive-path equivalent of ``_cabin_ctx`` — a bare ``config`` plus explicit
    video/mask paths, since the day-master path has no stage ctx to read from."""
    drive = SimpleNamespace(
        blur_enabled=True,
        skip_blur=skip_blur,
        blur_vehicle_cabin_frac=cabin_frac,
        blur_vehicle_conf=0.6,
        blur_vehicle_min_height_frac=0.07,
        blur_vehicle_ignore_bottom_frac=0.12,
        blur_plate_max_area_frac=0.006,
        blur_cabin_gate_sample_fps=4.0,
        blur_cabin_gate_min_coverage=0.5,
    )
    config = SimpleNamespace(drive=drive)
    mask = tmp_path / "day_mask.mkv"
    mask.write_bytes(b"\x00" * 32)
    return config, tmp_path / "day_raw.mp4", mask


class TestCheckDayMasterCabinCoverage:
    """Same occupant-coverage semantics as ``check_cabin_coverage``, reused on the
    drive day-master path (which builds its mask directly, not via a stage ctx)."""

    def test_noop_when_cabin_off(self, tmp_path):
        config, video, mask = _cabin_cfg(tmp_path, cabin_frac=0.0)
        assert blur_gate.check_day_master_cabin_coverage(config, video, mask) is None

    def test_noop_when_skip_blur(self, tmp_path):
        config, video, mask = _cabin_cfg(tmp_path, skip_blur=True)
        assert blur_gate.check_day_master_cabin_coverage(config, video, mask) is None

    def test_noop_when_mask_absent(self, tmp_path):
        config, video, _mask = _cabin_cfg(tmp_path)
        assert blur_gate.check_day_master_cabin_coverage(config, video, None) is None

    def test_passes_when_no_leaks(self, tmp_path, monkeypatch):
        import redactcam

        from gomoveshift import analysis

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        monkeypatch.setattr(
            analysis,
            "verify_cabin_coverage",
            lambda *a, **k: redactcam.CoverageReport(vehicle_frames=5, leaks=[]),
        )
        config, video, mask = _cabin_cfg(tmp_path)
        assert blur_gate.check_day_master_cabin_coverage(config, video, mask) is None

    def test_fails_on_exposed_cabin(self, tmp_path, monkeypatch):
        import redactcam

        from gomoveshift import analysis

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        rep = redactcam.CoverageReport(
            vehicle_frames=5, leaks=[redactcam.Leak(100, 0.0, (1, 2, 3, 4))]
        )
        monkeypatch.setattr(analysis, "verify_cabin_coverage", lambda *a, **k: rep)
        config, video, mask = _cabin_cfg(tmp_path)
        err = blur_gate.check_day_master_cabin_coverage(config, video, mask)
        assert err and "EXPOSED" in err and "--skip-blur" in err

    def test_second_call_with_identical_inputs_reuses_verdict(self, tmp_path, monkeypatch):
        import redactcam

        from gomoveshift import analysis

        calls = []

        def _stub_verify(*a, **k):
            calls.append(1)
            return redactcam.CoverageReport(vehicle_frames=5, leaks=[])

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        monkeypatch.setattr(analysis, "verify_cabin_coverage", _stub_verify)

        config, video, mask = _cabin_cfg(tmp_path)
        video.write_bytes(b"\x00" * 128)  # file_fingerprint needs a real file

        first = blur_gate.check_day_master_cabin_coverage(config, video, mask)
        second = blur_gate.check_day_master_cabin_coverage(config, video, mask)

        assert first is None and second is None
        assert len(calls) == 1, "identical video+mask+config must not re-pay the check"

    def test_changed_video_misses_the_cache(self, tmp_path, monkeypatch):
        import time

        import redactcam

        from gomoveshift import analysis

        calls = []

        def _stub_verify(*a, **k):
            calls.append(1)
            return redactcam.CoverageReport(vehicle_frames=5, leaks=[])

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        monkeypatch.setattr(analysis, "verify_cabin_coverage", _stub_verify)

        config, video, mask = _cabin_cfg(tmp_path)
        video.write_bytes(b"\x00" * 128)

        blur_gate.check_day_master_cabin_coverage(config, video, mask)
        time.sleep(0.01)
        video.write_bytes(b"\x01" * 256)  # different size + mtime -> different fingerprint
        blur_gate.check_day_master_cabin_coverage(config, video, mask)

        assert len(calls) == 2, "a changed video must miss the cached verdict"

    def test_corrupt_sidecar_falls_through_to_a_real_check(self, tmp_path, monkeypatch):
        import redactcam

        from gomoveshift import analysis

        calls = []

        def _stub_verify(*a, **k):
            calls.append(1)
            return redactcam.CoverageReport(vehicle_frames=5, leaks=[])

        monkeypatch.setattr(analysis, "resolve_vehicle_model", lambda d: Path("v.onnx"))
        monkeypatch.setattr(analysis, "resolve_plate_model", lambda d: Path("p.onnx"))
        monkeypatch.setattr(analysis, "verify_cabin_coverage", _stub_verify)

        config, video, mask = _cabin_cfg(tmp_path)
        video.write_bytes(b"\x00" * 128)
        blur_gate.cabin_verdict_path(mask).write_text("{not valid json")

        result = blur_gate.check_day_master_cabin_coverage(config, video, mask)

        assert result is None
        assert len(calls) == 1, "a corrupt sidecar must fall through to a real check, never a pass"


def _stage_ctx(tmp_path, *, blur_enabled=True):
    """A resume-complete ctx: encode stages already done, FINAL on disk."""
    cfg = MagicMock()
    cfg.drive.blur_enabled = blur_enabled
    cfg.drive.skip_blur = False
    proj = tmp_path / "proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "proj_FINAL.mp4").write_bytes(b"\x00" * 64)
    paths = PipelinePaths(
        output_base=tmp_path / "out",
        project_dir=proj,
        stages_dir=tmp_path / "stages",
        state_file=tmp_path / "s.json",
        staging_dir=tmp_path / "stg",
    )
    (tmp_path / "stages").mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        project_name="proj",
        source_video=tmp_path / "src.mp4",
        source_hash="x",
        config=cfg,
        location="A to B",
        publish_day="wed",
        paths=paths,
        current_video=tmp_path / "no_stops.mp4",
        video_properties=MagicMock(
            width=1920, height=1080, fps=30.0, bitrate_kbps=0, duration=60.0
        ),
    )
    for s in (
        PipelineContext.STAGE_COLOR_CORRECTION,
        PipelineContext.STAGE_OST_OVERLAYS,
        PipelineContext.STAGE_AUDIO_NORMALIZATION,
    ):
        ctx.stages_completed.append(s)
    return ctx


class TestStageWiring:
    def test_resume_complete_rejects_unmarked_final(self, tmp_path):
        ctx = _stage_ctx(tmp_path, blur_enabled=True)
        res = SinglePassEncodeStage().run(ctx)
        assert res.success is False and "blur marker" in res.error

    def test_resume_complete_accepts_marked_final(self, tmp_path):
        ctx = _stage_ctx(tmp_path, blur_enabled=True)
        blur_gate.mark_applied(ctx.paths.project_dir / "proj_FINAL.mp4")
        res = SinglePassEncodeStage().run(ctx)
        assert res.success is True

    def test_resume_complete_ok_when_blur_off(self, tmp_path):
        ctx = _stage_ctx(tmp_path, blur_enabled=False)
        res = SinglePassEncodeStage().run(ctx)
        assert res.success is True  # no marker needed when blur disabled


class TestSingleVideoPathHasNoSkipBlurFlag:
    """``check_mask_ready``/``check_applied`` fire only from
    ``single_pass_encode.py`` (the ``pipeline`` single-video command). Unlike
    ``drive``, that subcommand registers no ``--skip-blur`` flag — a message on
    this path pointing an operator at it sends them to an argparse rejection
    while an irreversible encode is pending. The real escape is
    ``drive.skip_blur: true`` in ``config.yaml``."""

    def test_pipeline_subcommand_registers_no_skip_blur(self):
        from gomoveshift.cli.parser import _build_parser

        top = _build_parser()
        sub = top._subparsers._group_actions[0]
        pipeline = sub.choices["pipeline"]
        opts = {opt for action in pipeline._actions for opt in action.option_strings}
        assert "--skip-blur" not in opts, (
            "pipeline now registers --skip-blur — the single-video blur_gate "
            "messages may name it directly again"
        )

    def test_check_mask_ready_points_at_config_yaml(self, tmp_path):
        ctx = _ctx(tmp_path, mask_path=None)
        err = blur_gate.check_mask_ready(ctx)
        assert err and "config.yaml" in err

    def test_check_applied_no_marker_points_at_config_yaml(self, tmp_path):
        final = tmp_path / "proj_FINAL.mp4"
        final.write_bytes(b"\x00")
        ctx = _ctx(tmp_path)
        err = blur_gate.check_applied(ctx, final)
        assert err and "config.yaml" in err


class TestFrameSamplerHasOneHome:
    """The shorts post-burn proof copied this module's frame sampler because this
    module had not merged when that one was written. Both have merged, and the
    copy had already drifted — it lost the float32 cast and the first-frame height
    pin, so the two pixel gates were reading frames by different rules."""

    def test_the_shorts_proof_reuses_this_module_s_sampler(self):
        assert ost_audit_sidecar._gray_frames is blur_gate._gray_frames

    def test_only_one_module_defines_it(self):
        pkg = Path(blur_gate.__file__).parent
        homes = sorted(
            path.name
            for path in pkg.rglob("*.py")
            if re.search(r"^def _gray_frames\(", path.read_text(), re.MULTILINE)
        )
        assert homes == ["blur_gate.py"]

    def test_frames_are_float32_with_the_height_pinned_by_the_first_read(self, monkeypatch):
        """float32 because callers difference these arrays: in uint8 a
        dark-minus-bright neighbour wraps to a large positive and reads as the
        detail a blur is supposed to have removed. A later short read yields
        ``[]`` — which every caller treats as unmeasurable and passes — rather
        than a frame reshaped to geometry the first frame did not have."""
        reads = [b"\x80" * (640 * 4), b"\x80" * (640 * 3)]
        cmds = []

        def fake_run(cmd, **kwargs):
            cmds.append(cmd)
            return SimpleNamespace(returncode=0, stdout=reads[len(cmds) - 1])

        monkeypatch.setattr(blur_gate.subprocess, "run", fake_run)
        frames = blur_gate._gray_frames(Path("v.mp4"), [1.0], 640)
        assert [(f.shape, f.dtype) for f in frames] == [((4, 640), np.float32)]
        assert "scale=640:-2,format=gray" in cmds[0]

        cmds.clear()
        assert blur_gate._gray_frames(Path("v.mp4"), [1.0, 2.0], 640) == []
