"""Behaviour-parity pins for the ``redactcam`` rewire.

The privacy-blur chain moved out of this tree into the ``redactcam`` package.
Every tuned number that used to be a module constant here is now a *library
default*, which means a future redactcam release could change what this pipeline
does without a single line of this repo changing — silently, in an unattended run
that ships public video.

So: every default the service relies on WITHOUT passing it explicitly is pinned
below. If a redactcam upgrade moves one, this file goes red and the change gets
made deliberately (pass the old value at the call site, or accept the new one).
The knobs the service DOES pass explicitly, from ``config.drive``, are checked
against the package's ``ROAD_FOOTAGE`` preset instead — that preset is redactcam's
record of the values this footage was tuned on, and the two must not drift apart.
"""

import inspect

import redactcam
from redactcam import apply as redactcam_apply
from redactcam import timeline as redactcam_timeline

from gomoveshift.config import DriveConfig

# Defaults of ``detect_and_track`` that NEITHER blur call site passes
# (``pipeline/stages/blur_detection.py`` and ``pipeline/day_master_encoder.py``
# pass the same set, from ``config.drive``). These are the production tracker
# values, validated on real 4K road footage.
UNPASSED_DETECT_AND_TRACK_DEFAULTS = {
    "face_infer_shape": (640, 360),
    "plate_infer_size": 640,
    "plate_iou": 0.45,
    "track_downscale": 2,
    "track_lk_win": 31,
    "track_lk_levels": 4,
    "track_max_horizon_frames": 9,
    "track_face_coast_frames": 20,
    "track_face_coast_moving_frames": 6,
    "track_coast_move_speed": 6.0,
    "track_retire_edge_coast_margin": 3,
    "track_min_points": 6,
    "track_scale_min": 0.8,
    "track_scale_max": 1.25,
    "track_min_inliers": 4,
    "track_motion_gain": 1.0,
    "track_motion_max_px": 160,
    "track_plate_margin_floor_px": 16,
    "track_assoc_motion_gain": 1.2,
    "track_assoc_min_gate_px": 72,
    "track_dedup_iou": 0.55,
    "plate_threads": 0,
    # Crash-resume checkpointing: the day-master call site passes a path and a
    # cadence (multi-hour passes, see BLUR_CHECKPOINT_EVERY_S); the single-video
    # stage does not, so its defaults stay pinned — a future
    # redactcam that turns checkpointing ON by default fails here instead of
    # silently writing state files beside a single-video master. The overlap is
    # passed by neither and is what repairs the seam on a day-master resume.
    "checkpoint_path": None,
    "checkpoint_every_frames": 0,
    "checkpoint_overlap_frames": 90,
}

# Defaults of parameters a newer redactcam adds. Each is checked exactly like
# the table above whenever the installed redactcam has it, so this test can
# land before the dependency pin moves. Move an entry up into the table above
# once the pin that adds it has landed.
PINNED_WHEN_PRESENT_DEFAULTS = {
    # Detection heartbeat: a once-a-minute INFO line from
    # detect_and_track, which reaches drive.log through the root logger. A
    # multi-hour blur pass was otherwise silent and read as a hang.
    # Pinned so a redactcam that silences or floods it fails here.
    "progress_every_s": 60.0,
}


def _defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


class TestDetectAndTrackDefaults:
    def test_unpassed_defaults_are_pinned(self):
        got = _defaults(redactcam.detect_and_track)
        for name, expected in UNPASSED_DETECT_AND_TRACK_DEFAULTS.items():
            assert got[name] == expected, (
                f"redactcam.detect_and_track({name}=) changed from {expected!r} to "
                f"{got[name]!r}. The blur stages do not pass this — the pipeline's "
                "tracker behaviour just moved. Pass the old value explicitly or "
                "accept the new one deliberately."
            )
        for name, expected in PINNED_WHEN_PRESENT_DEFAULTS.items():
            if name in got:
                assert got[name] == expected, (
                    f"redactcam.detect_and_track({name}=) is {got[name]!r}, expected {expected!r}."
                )

    def test_every_parameter_is_either_passed_or_pinned(self):
        """A NEW redactcam parameter with a default is also a silent behaviour
        change. Fail until it is classified: passed from config, or pinned above."""
        passed_from_config = {
            "sample_fps",
            "conf",
            "face_conf",
            "face_sample_fps",
            "face_confirm_min_detections",
            "pre_roll_frames",
            "plate_model_path",
            "face_model_path",
            "tile_px",
            "tile_overlap",
            "plate_min_aspect",
            "plate_max_aspect",
            "plate_max_area_frac",
            "plate_boost_speed_px",
            "plate_boost_max_run",
            "track_face_margin_frac",
            "track_face_motion_gain",
            "track_cabin_frac",
            "vehicle_model_path",
            "vehicle_conf",
            "vehicle_min_height_frac",
            "vehicle_ignore_bottom_frac",
            "track_person",
            "person_model_path",
            "person_conf",
            "person_min_height_frac",
        }
        known = (
            passed_from_config
            | set(UNPASSED_DETECT_AND_TRACK_DEFAULTS)
            | set(PINNED_WHEN_PRESENT_DEFAULTS)
        )
        assert set(_defaults(redactcam.detect_and_track)) - known == set()


class TestRoadFootagePresetMatchesDriveConfig:
    """``ROAD_FOOTAGE`` is redactcam's record of the values this pipeline was tuned
    on. The service still drives every one of them from ``config.drive``, so the
    two are independent copies of the same tuning — and must agree."""

    def test_preset_values_match_config_defaults(self):
        cfg = DriveConfig()
        preset = redactcam.ROAD_FOOTAGE
        pairs = {
            "sample_fps": "blur_sample_fps",
            "face_sample_fps": "blur_face_sample_fps",
            "conf": "blur_conf",
            "face_conf": "blur_face_conf",
            "face_confirm_min_detections": "blur_face_confirm_min_detections",
            "plate_min_aspect": "blur_plate_min_aspect",
            "plate_max_aspect": "blur_plate_max_aspect",
            "plate_max_area_frac": "blur_plate_max_area_frac",
            "tile_px": "blur_tile_px",
            "tile_overlap": "blur_tile_overlap",
            "plate_boost_speed_px": "blur_plate_boost_speed_px",
            "plate_boost_max_run": "blur_plate_boost_max_run",
            "track_face_motion_gain": "blur_face_motion_gain",
            "track_face_margin_frac": "blur_face_margin_frac",
            "track_cabin_frac": "blur_vehicle_cabin_frac",
            "vehicle_conf": "blur_vehicle_conf",
            "vehicle_min_height_frac": "blur_vehicle_min_height_frac",
            "vehicle_ignore_bottom_frac": "blur_vehicle_ignore_bottom_frac",
            "person_conf": "blur_person_conf",
            "person_min_height_frac": "blur_person_min_height_frac",
            "pre_roll_frames": "blur_pre_roll_frames",
        }
        for preset_key, cfg_field in pairs.items():
            assert preset[preset_key] == getattr(cfg, cfg_field), (
                f"ROAD_FOOTAGE[{preset_key!r}]={preset[preset_key]!r} no longer matches "
                f"DriveConfig.{cfg_field}={getattr(cfg, cfg_field)!r}"
            )

    def test_box_dilation_and_feather_are_8(self):
        # The flat safety margin added to every box, and the mask-edge feather.
        # ``materialize_mask_video`` is called WITHOUT feather_px by both stages,
        # so its default is the production value.
        assert redactcam.ROAD_FOOTAGE_BOX_DILATION_PX == 8
        assert DriveConfig().blur_box_dilation_px == 8
        assert redactcam.ROAD_FOOTAGE_FEATHER_PX == 8
        assert _defaults(redactcam.materialize_mask_video)["feather_px"] == 8


class TestCoverageGateDefaults:
    """``blur_gate.check_cabin_coverage`` passes cabin_frac / conf / sample_fps /
    min_coverage / min_height_frac / ignore_bottom_frac / plate_max_area_frac from
    config. It does NOT pass plate_conf or infer_size — pin those."""

    def test_unpassed_verifier_defaults(self):
        got = _defaults(redactcam.verify_cabin_coverage)
        assert got["plate_conf"] == 0.4
        assert got["infer_size"] == 640

    def test_passed_verifier_defaults_still_match_config(self):
        cfg = DriveConfig()
        got = _defaults(redactcam.verify_cabin_coverage)
        assert got["cabin_frac"] == cfg.blur_vehicle_cabin_frac == 0.6
        assert got["min_coverage"] == cfg.blur_cabin_gate_min_coverage == 0.5
        assert got["sample_fps"] == cfg.blur_cabin_gate_sample_fps == 4.0
        assert got["conf"] == cfg.blur_vehicle_conf == 0.6
        assert got["min_height_frac"] == cfg.blur_vehicle_min_height_frac == 0.07
        assert got["ignore_bottom_frac"] == cfg.blur_vehicle_ignore_bottom_frac == 0.12
        assert got["plate_max_area_frac"] == cfg.blur_plate_max_area_frac == 0.006


class TestStillPhotoDefaults:
    """``photos/face_filter.py`` calls ``detect_image`` with only ``conf``, so the
    still path inherits the package's tiling and area gate."""

    def test_detect_image_defaults(self):
        got = _defaults(redactcam.detect_image)
        assert got["tile_px"] == 1280
        assert got["tile_overlap"] == 0.15
        # NOTE: 0.01, deliberately looser than the video path's 0.006 — the still
        # path has never passed this, and tightening it here would change which
        # plate boxes survive on published photos.
        assert got["plate_max_area_frac"] == 0.01

    def test_build_image_detectors_session_params(self):
        """face_filter overrides only the model paths and the plate aspect band;
        infer size / NMS iou / thread cap come from the package."""
        src = inspect.getsource(redactcam.build_image_detectors)
        assert '"face_infer_shape": (640, 360)' in src
        assert '"infer_size": 640' in src
        assert '"iou": 0.45' in src
        assert '"threads": 0' in src


class TestBlurStrength:
    """``single_pass._blur_filter`` computes the boxblur radius itself from
    ``config.drive.blur_strength``. redactcam ships the same formula; keep the two
    numerically identical so the library and the encode never disagree."""

    def test_reference_strength_matches_config(self):
        assert redactcam_apply.REFERENCE_BLUR_STRENGTH == DriveConfig().blur_strength == 12
        assert redactcam_apply.REFERENCE_WIDTH == 1920

    def test_radius_formula_matches_single_pass(self):
        for width in (1280, 1920, 3840):
            expected = max(2, round(DriveConfig().blur_strength * width / 1920))
            assert redactcam.blur_radius(width) == expected


class TestSidecarSchemaChanged:
    """DELTA, accepted: the region sidecar's schema string moved from
    ``blur_regions/v2`` to ``redactcam.timeline/v1``. ``load_sidecar`` returns None
    on a schema mismatch, so any ``*_blur_regions.json`` written before this rewire
    is ignored and detection re-runs. That costs one detection pass per in-flight
    project and cannot produce a WRONG mask — the failure mode a silent schema
    reuse would have had."""

    def test_schema_is_the_package_string(self):
        assert redactcam_timeline.SCHEMA == "redactcam.timeline/v1"

    def test_a_legacy_sidecar_is_rejected_not_reused(self, tmp_path):
        import json

        legacy = tmp_path / "proj_blur_regions.json"
        legacy.write_text(json.dumps({"schema": "blur_regions/v2", "source_hash": "abc"}))
        assert redactcam_timeline.load_sidecar(legacy) is None
