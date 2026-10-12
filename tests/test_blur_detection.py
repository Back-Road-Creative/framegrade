"""Tests for the blur-detection stage (skip/resume/detect behaviour).

The detector and video probe are mocked, so these run without the optional
``deface``/``onnxruntime`` inference deps. Timeline math lives in
the ``redactcam`` package's own suite.
"""

from unittest.mock import MagicMock

from redactcam import timeline as br
from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.stages.blur_detection import BlurDetectionStage


def _ctx(tmp_path, **drive):
    cfg = MagicMock()
    cfg.drive.blur_enabled = drive.get("blur_enabled", True)
    cfg.drive.skip_blur = drive.get("skip_blur", False)
    cfg.drive.blur_sample_fps = 8.0
    cfg.drive.blur_conf = 0.4
    cfg.drive.blur_box_dilation_px = 4
    # Real int: track-birth backfill does arithmetic on it, same reason as
    # blur_vehicle_cabin_frac below.
    cfg.drive.blur_pre_roll_frames = 5
    cfg.drive.plate_model_path = ""
    # Occupant cabin blur off by default in tests (a real float so the >0 model-resolve
    # guard doesn't choke on a MagicMock); a test opts in explicitly.
    cfg.drive.blur_vehicle_cabin_frac = drive.get("blur_vehicle_cabin_frac", 0.0)
    paths = PipelinePaths(
        output_base=tmp_path / "out",
        project_dir=tmp_path / "proj",
        stages_dir=tmp_path / "stages",
        state_file=tmp_path / "s.json",
        staging_dir=tmp_path / "stg",
    )
    (tmp_path / "stages").mkdir(parents=True, exist_ok=True)
    return PipelineContext(
        project_name="proj",
        source_video=tmp_path / "src.mp4",
        source_hash="x",
        config=cfg,
        location="A to B",
        publish_day="wed",
        paths=paths,
        current_video=tmp_path / "no_stops.mp4",
        video_properties=MagicMock(width=1920, height=1080, fps=30.0),
    )


def test_should_run_gated_by_enabled_and_skip(tmp_path):
    stage = BlurDetectionStage()
    assert stage.should_run(_ctx(tmp_path, blur_enabled=True)) is True
    assert stage.should_run(_ctx(tmp_path, blur_enabled=False)) is False
    assert stage.should_run(_ctx(tmp_path, blur_enabled=True, skip_blur=True)) is False


def test_run_detects_and_writes_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr("gomoveshift.pipeline.stop_processor.compute_file_hash", lambda p: "HASH1")
    monkeypatch.setattr(br, "probe_video", lambda p: (30.0, 0, 1920, 1080))
    # detect_and_track returns (frame_boxes, decoded_count, n_faces, n_plates);
    # decoded_count (96) takes precedence over the probe's 0.
    monkeypatch.setattr(
        "redactcam.detect_and_track",
        lambda *a, **k: ({0: [(10, 10, 20, 20)]}, 96, 1, 0),
    )
    mask_calls = []
    monkeypatch.setattr(
        "redactcam.materialize_mask_video",
        lambda tl, p, **k: (mask_calls.append(p), p)[1],
    )
    ctx = _ctx(tmp_path)
    result = BlurDetectionStage().run(ctx)
    assert result.success
    assert ctx.blur_regions_path == tmp_path / "stages" / "proj_blur_regions.json"
    saved = br.load_sidecar(ctx.blur_regions_path)
    assert saved.source_hash == "HASH1"
    assert saved.source_frame_count == 96  # decode count wins over probe's 0
    assert saved.frame_boxes[0] == [(6, 6, 28, 28)]  # the tracked box, dilated by 4
    # A fresh detection materializes the mask and exposes it for the render stage.
    assert ctx.blur_mask_path == tmp_path / "stages" / "proj_blur_mask.mkv"
    assert mask_calls == [ctx.blur_mask_path]


def test_run_resumes_on_hash_match_without_redetecting(tmp_path, monkeypatch):
    sidecar = tmp_path / "stages" / "proj_blur_regions.json"
    (tmp_path / "stages").mkdir(parents=True, exist_ok=True)
    br.write_sidecar(
        sidecar,
        br.build_timeline(
            {},
            source_hash="MATCH",
            source_fps=30.0,
            source_frame_count=10,
            frame_width=1920,
            frame_height=1080,
            box_dilation_px=4,
            sample_fps=8.0,
            conf=0.4,
        ),
    )
    monkeypatch.setattr("gomoveshift.pipeline.stop_processor.compute_file_hash", lambda p: "MATCH")

    def _boom(*a, **k):
        raise AssertionError("detect_and_track must not run on a hash match")

    monkeypatch.setattr("redactcam.detect_and_track", _boom)
    monkeypatch.setattr("redactcam.materialize_mask_video", lambda tl, p, **k: p)
    ctx = _ctx(tmp_path)
    assert BlurDetectionStage().run(ctx).success
    assert ctx.blur_regions_path == sidecar
    assert ctx.blur_mask_path == tmp_path / "stages" / "proj_blur_mask.mkv"


def test_run_reports_failure_on_detector_error(tmp_path, monkeypatch):
    monkeypatch.setattr("gomoveshift.pipeline.stop_processor.compute_file_hash", lambda p: "H")
    monkeypatch.setattr(br, "probe_video", lambda p: (30.0, 90, 1920, 1080))
    monkeypatch.setattr(
        "redactcam.detect_and_track",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model gone")),
    )
    result = BlurDetectionStage().run(_ctx(tmp_path))
    assert result.success is False and "Blur detection failed" in result.error


def test_run_hard_fails_when_cabin_on_but_no_vehicle_model(tmp_path, monkeypatch):
    """Occupant cabin blur enabled + unresolvable vehicle model → hard fail rather than
    silently disable (drivers would ship unblurred). cabin off → no such failure."""
    monkeypatch.setattr("gomoveshift.pipeline.stop_processor.compute_file_hash", lambda p: "H2")
    monkeypatch.setattr(br, "probe_video", lambda p: (30.0, 90, 1920, 1080))
    monkeypatch.setattr("gomoveshift.analysis.resolve_vehicle_model", lambda d: None)
    result = BlurDetectionStage().run(_ctx(tmp_path, blur_vehicle_cabin_frac=0.6))
    assert result.success is False and "vehicle model" in result.error


def test_blur_stage_ordered_after_stops_before_encode():
    from gomoveshift.pipeline.stages import (
        STAGE_ORDER,
        ChapterGenerationStage,
        PreRenderReviewStage,
        SinglePassEncodeStage,
        StopRemovalStage,
    )

    idx = {type(s): i for i, s in enumerate(STAGE_ORDER)}
    assert idx[StopRemovalStage] < idx[BlurDetectionStage] < idx[ChapterGenerationStage]
    assert idx[BlurDetectionStage] < idx[PreRenderReviewStage] < idx[SinglePassEncodeStage]
