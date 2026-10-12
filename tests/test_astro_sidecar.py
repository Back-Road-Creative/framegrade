"""``.stack.json`` sidecar: numpy/Path -> JSON coercion is the point of this
suite, plus the atomic, never-raises writer mirroring
``develop.write_grade_sidecar``."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_sidecar as sidecar_mod
from gomoveshift.pipeline.photos.astro_develop import AstroDevelopResult
from gomoveshift.pipeline.photos.astro_register import RegistrationResult
from gomoveshift.pipeline.photos.astro_stack import StackResult
from gomoveshift.pipeline.photos.astro_triage import (
    FrameMetrics,
    StarShape,
    StreakClassification,
)


def _metrics(name, star_count=42, background=np.float64(0.031), fwhm=np.float32(3.2), shape=None):
    return FrameMetrics(
        path=Path(f"/cards/night1/{name}.NEF"),
        star_count=star_count,
        background_median=background,
        fwhm_px=fwhm,
        streaks=[(10.0, 10.0, 40.0, 12.0)],
        usable=True,
        shape=shape,
    )


def _star_shape(major, minor, ellipticity):
    return StarShape(
        fwhm_major_px=major,
        fwhm_minor_px=minor,
        ellipticity=ellipticity,
        orientation_deg=np.float64(118.0),
        hfr_px=np.float32(4.9),
        saturated=False,
    )


# A real capture BEFORE a refocus pause, measured over its lost
# frames: fwhm_px 10.81-11.49 at ellipticity 0.049-0.077 -- round and fat, i.e. defocus.
_DEFOCUSED = _star_shape(np.float32(11.34), np.float32(10.66), np.float64(0.06))
# The same capture AFTER that pause: fwhm_px 6.82-8.66, and
# round at that size is the unremarkable case.
_HEALTHY = _star_shape(np.float32(7.60), np.float32(7.20), np.float64(0.026))


def _registration():
    return RegistrationResult(
        transform=np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float64),
        n_matched=37,
        inlier_fraction=np.float64(0.91),
    )


def _develop_result():
    return AstroDevelopResult(
        rgb_uint16=np.zeros((4, 4, 3), dtype=np.uint16),
        black_point=np.float64(0.012),
        m_values=[np.float64(0.31), np.float64(0.24), 0.19],
        background_before=np.float64(0.08),
        background_after=0.12,
        background_gains=(np.float64(1.0), np.float64(1.02), np.float64(0.97)),
    )


def _stack_result():
    return StackResult(
        rgb=np.zeros((4, 4, 3), dtype=np.float32),
        n_frames=14,
        mean_rejected_fraction=np.float64(0.008),
    )


def _payload():
    used = [
        sidecar_mod.UsedFrame(metrics=_metrics("frame001"), registration=None),
        sidecar_mod.UsedFrame(metrics=_metrics("frame002"), registration=_registration()),
    ]
    rejected = [
        (Path("/cards/night1/frame003.NEF"), "cloud"),
        (Path("/cards/night1/frame004.NEF"), "registration failed: only 3 matched star pairs"),
    ]
    streaks = [
        StreakClassification(
            frame_index=1,
            path=Path("/cards/night1/frame002.NEF"),
            streak=(5.0, 6.0, 55.0, 61.0),
            label="satellite",
        )
    ]
    return sidecar_mod.stack_sidecar_payload(
        used_frames=used,
        rejected_frames=rejected,
        streaks=streaks,
        develop=_develop_result(),
        stack=_stack_result(),
        decoder="rawpy-linear+astrostack14",
        algo_version="astro-v1",
    )


def _min_payload(**overrides):
    # Shared minimal-args shape for the input/capture/converter cases below.
    kwargs = dict(
        used_frames=[],
        rejected_frames=[],
        develop=_develop_result(),
        stack=_stack_result(),
        decoder="d",
        algo_version="v",
    )
    kwargs.update(overrides)
    return sidecar_mod.stack_sidecar_payload(**kwargs)


def test_payload_round_trips_through_json_dumps_with_numpy_and_path_inputs():
    payload = _payload()
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["decoder"] == "rawpy-linear+astrostack14"
    assert decoded["algo_version"] == "astro-v1"


def test_used_and_rejected_frames_carry_metrics_transform_and_reason():
    payload = _payload()
    first, second = payload["frames"]["used"]
    assert first["metrics"]["path"] == "/cards/night1/frame001.NEF"
    assert first["metrics"]["star_count"] == 42
    assert isinstance(first["metrics"]["background_median"], float)
    assert "transform" not in first  # no registration for the reference frame
    assert second["transform"] == [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]
    assert second["n_matched"] == 37
    assert isinstance(second["inlier_fraction"], float)
    rejected = {r["path"]: r["reason"] for r in payload["frames"]["rejected"]}
    assert rejected["/cards/night1/frame003.NEF"] == "cloud"
    assert rejected["/cards/night1/frame004.NEF"].startswith("registration failed")


def test_frame_metrics_record_the_shape_verdict_and_the_split_behind_fwhm_px():
    """``classify_shape`` had zero production callers, so a whole night's
    ``likely_defocus`` verdict was recomputed per frame and thrown away. The sidecar
    keeps the verdict AND the major/minor/ellipticity split ``fwhm_px`` averages out."""
    payload = _min_payload(
        used_frames=[
            sidecar_mod.UsedFrame(
                metrics=_metrics("f1", fwhm=np.float32(11.0), shape=_DEFOCUSED), registration=None
            ),
            sidecar_mod.UsedFrame(
                metrics=_metrics("f2", fwhm=np.float32(7.4), shape=_HEALTHY), registration=None
            ),
        ]
    )
    defocused, healthy = json.loads(json.dumps(payload, sort_keys=True))["frames"]["used"]
    assert defocused["metrics"]["shape_verdict"] == "likely_defocus"
    assert defocused["metrics"]["shape"] == pytest.approx(
        {"fwhm_major_px": 11.34, "fwhm_minor_px": 10.66, "ellipticity": 0.06}, abs=1e-4
    )
    # Round and sharp is the normal case: a label, but not one worth acting on.
    assert healthy["metrics"]["shape_verdict"] == "indeterminate"
    assert healthy["metrics"]["shape"] == pytest.approx(
        {"fwhm_major_px": 7.60, "fwhm_minor_px": 7.20, "ellipticity": 0.026}, abs=1e-4
    )


def test_a_frame_with_no_measurable_star_shape_records_null_not_a_faked_zero():
    first, _ = _payload()["frames"]["used"]  # _metrics() defaults shape=None
    assert first["metrics"]["shape"] is None
    assert first["metrics"]["shape_verdict"] == "indeterminate"


def test_streak_classifications_and_develop_and_stack_numbers_present():
    payload = _payload()
    (streak,) = payload["streaks"]
    assert streak["label"] == "satellite" and streak["frame_index"] == 1
    assert streak["streak"] == [5.0, 6.0, 55.0, 61.0]
    develop = payload["develop"]
    assert develop["black_point"] == pytest.approx(0.012)
    assert develop["m_values"] == pytest.approx([0.31, 0.24, 0.19])
    assert develop["background_before"] == pytest.approx(0.08)
    assert develop["background_after"] == pytest.approx(0.12)
    assert develop["background_gains"] == pytest.approx([1.0, 1.02, 0.97])
    assert payload["stack"]["n_frames"] == 14
    assert payload["stack"]["mean_rejected_fraction"] == pytest.approx(0.008)


def test_develop_payload_records_peak_from_a_measured_value():
    develop = AstroDevelopResult(
        rgb_uint16=np.zeros((4, 4, 3), dtype=np.uint16),
        black_point=np.float64(0.012),
        m_values=[np.float64(0.31)],
        background_before=np.float64(0.08),
        background_after=0.12,
        background_gains=(np.float64(1.0),),
        peak=np.float64(0.734),
    )
    payload = sidecar_mod.stack_sidecar_payload(
        used_frames=[],
        rejected_frames=[],
        develop=develop,
        stack=_stack_result(),
        decoder="d",
        algo_version="v",
    )
    assert payload["develop"]["peak"] == pytest.approx(0.734)
    assert isinstance(payload["develop"]["peak"], float)


def test_develop_payload_records_the_unmeasured_zero_default_honestly():
    # ``_develop_result()`` never sets ``peak``, so it carries the dataclass's own default.
    payload = _payload()
    assert payload["develop"]["peak"] == pytest.approx(0.0)


def test_determinism_same_inputs_produce_byte_identical_json():
    first = json.dumps(_payload(), indent=2, sort_keys=True)
    second = json.dumps(_payload(), indent=2, sort_keys=True)
    assert first == second


def test_meteor_decision_defaults_off_and_records_explicit_values():
    off = {
        "mode": "off",
        "candidate_count": 0,
        "reinserted_count": 0,
        "skip_reason": None,
        "decisions": [],
    }
    assert _payload()["meteors"] == off
    payload = sidecar_mod.stack_sidecar_payload(
        used_frames=[],
        rejected_frames=[],
        develop=_develop_result(),
        stack=_stack_result(),
        decoder="d",
        algo_version="v",
        meteor_mode="reviewed",
        meteor_candidate_count=283,
        meteor_reinserted_count=1,
        meteor_skip_reason="guard",
    )
    reviewed = {
        "mode": "reviewed",
        "candidate_count": 283,
        "reinserted_count": 1,
        "skip_reason": "guard",
        "decisions": [],
    }
    assert payload["meteors"] == reviewed


def test_write_stack_sidecar_is_atomic_and_leaves_no_tmp_residue(tmp_path):
    image_path = tmp_path / "master.tiff"
    image_path.write_bytes(b"fake-tiff")
    out_path = sidecar_mod.write_stack_sidecar(_payload(), image_path)
    assert out_path == tmp_path / "master.stack.json"
    assert out_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_stack_sidecar_never_raises_on_unwritable_path(tmp_path):
    unwritable_dir = tmp_path / "does-not-exist"
    image_path = unwritable_dir / "master.tiff"  # parent dir was never created
    result = sidecar_mod.write_stack_sidecar(_payload(), image_path)
    assert result is None
    assert not unwritable_dir.exists()


def test_input_frame_paths_are_hashed_bytes_included_and_a_missing_file_stays_honest(tmp_path):
    light = tmp_path / "frame001.NEF"
    light.write_bytes(b"light-bytes")
    dark = tmp_path / "dark001.NEF"
    dark.write_bytes(b"dark-bytes")
    missing = tmp_path / "gone.NEF"
    import hashlib

    payload = _min_payload(light_paths=[light, missing], dark_paths=[dark])
    light_entry, missing_entry = payload["inputs"]["light"]
    (dark_entry,) = payload["inputs"]["dark"]
    assert light_entry["path"] == str(light)
    assert light_entry["sha256"] == hashlib.sha256(b"light-bytes").hexdigest()
    assert light_entry["hash_error"] is None
    assert dark_entry["sha256"] == hashlib.sha256(b"dark-bytes").hexdigest()
    assert missing_entry["sha256"] is None
    assert missing_entry["hash_error"]  # a reason string, not a silent None
    assert (tmp_path / "frame001.NEF").read_bytes() == b"light-bytes"  # never copied/renamed
    # bytes hash fine but Path(bytes) raises: without fsdecode, a hash attached to nothing.
    (as_bytes,) = _min_payload(light_paths=[os.fsencode(light)])["inputs"]["light"]
    assert (as_bytes["path"], as_bytes["hash_error"]) == (str(light), None)
    assert as_bytes["sha256"] == light_entry["sha256"]


def test_int_light_path_slot_degrades_honestly_and_never_touches_an_open_fd(tmp_path):
    # open() treats an int as an fd, not a path -- must never hash (and close) it.
    unrelated = tmp_path / "unrelated.NEF"
    unrelated.write_bytes(b"not-a-raw-frame")
    fd = os.open(unrelated, os.O_RDONLY)
    try:
        payload = _min_payload(light_paths=[fd])
        (entry,) = payload["inputs"]["light"]
        assert entry["sha256"] is None
        assert "int" in entry["hash_error"]
        os.read(fd, 1)  # OSError if _hash_file_bytes closed our fd
    finally:
        os.close(fd)


@pytest.mark.parametrize("bad_slot", [{"not": "a path"}, 7])
def test_wrong_type_light_path_slot_degrades_instead_of_aborting_the_whole_sidecar(bad_slot):
    payload = _min_payload(light_paths=[bad_slot])
    (entry,) = payload["inputs"]["light"]
    assert entry["path"] is None
    assert entry["sha256"] is None
    assert entry["hash_error"]  # a reason string, never a raise


def test_capture_identity_recorded_when_supplied():
    from gomoveshift.pipeline.photos.astro_calibrate import CaptureIdentity

    identity = CaptureIdentity(
        path=Path("/cards/night1/frame001.NEF"),
        camera="NIKON CORPORATION NIKON Z 6",
        raw_shape=(4016, 6048),
        cfa_pattern=(0, 1, 1, 2),
        iso=3200.0,
        exposure_s=20.0,
        temperature_c=None,  # unwritten tag -- honest null, not guessed
    )
    payload = _min_payload(capture_identity=identity)
    capture = payload["capture"]
    assert capture["available"] is True
    assert capture["path"] == "/cards/night1/frame001.NEF"
    assert capture["camera"] == "NIKON CORPORATION NIKON Z 6"
    assert capture["raw_shape"] == [4016, 6048]
    assert capture["iso"] == 3200.0
    assert capture["temperature_c"] is None


def test_capture_identity_absent_records_reason_not_silence():
    payload = _payload()
    assert payload["capture"] == {
        "available": False,
        "reason": "no CaptureIdentity supplied by the caller",
    }


def test_converter_mask_warnings_recorded_and_crop_slice_pair_stays_honest():
    # common_support_crop is the only crop producer and returns a slice pair.
    from gomoveshift.pipeline.photos.astro_register import common_support_crop

    counts = np.full((6, 6), 3, dtype=np.int32)
    payload = _min_payload(
        converter_version="rawpy-0.27.0",
        horizon_frac=0.35,
        mask_mode="operator_horizon_frac",
        crop=common_support_crop(counts, min_contributors=3),
        warnings=["dark sensor temperature 6.0C from light"],
        light_paths=[None],
    )
    decoded = json.loads(json.dumps(payload, sort_keys=True))
    assert decoded["converter_version"] == "rawpy-0.27.0"
    assert decoded["mask"] == {"horizon_frac": 0.35, "mode": "operator_horizon_frac"}
    assert decoded["crop"] == [[0, 6], [0, 6]]
    assert decoded["warnings"] == ["dark sensor temperature 6.0C from light"]
    (entry,) = decoded["inputs"]["light"]
    assert entry["path"] is None and entry["sha256"] is None
    assert entry["hash_error"]  # a reason string, never a raise


def test_used_frame_records_transform_residuals_and_match_yield():
    payload = _payload()
    _first, second = payload["frames"]["used"]
    assert second["residual_median_px"] == pytest.approx(0.0)
    assert second["residual_p95_px"] == pytest.approx(0.0)
    assert second["match_yield"] == pytest.approx(0.0)


def test_write_stack_sidecar_records_output_hash_and_survives_a_stray_value(tmp_path):
    import hashlib

    image_path = tmp_path / "master.tiff"
    image_path.write_bytes(b"real-exported-pixels")
    payload = _payload()
    payload["crop"] = slice(0, 6)  # belt-and-braces: bypasses stack_sidecar_payload's _jsonable
    out_path = sidecar_mod.write_stack_sidecar(payload, image_path)
    written = json.loads(out_path.read_text())
    assert written["output_hash"] == hashlib.sha256(b"real-exported-pixels").hexdigest()
    assert written["output_hash_error"] is None
    assert written["crop"] == "slice(0, 6, None)"


def test_v1_style_call_without_new_kwargs_still_produces_valid_json_with_safe_defaults():
    # _payload() passes only older kwargs; every v2/v3/v4 field must still default honestly.
    decoded = json.loads(json.dumps(_payload(), sort_keys=True))
    assert decoded["sidecar_version"] == "6"
    assert decoded["inputs"] == {"light": [], "dark": []}
    assert decoded["capture"]["available"] is False
    assert decoded["converter_version"] is None
    assert decoded["mask"] == {"horizon_frac": None, "mode": None}
    assert decoded["crop"] is None
    assert decoded["warnings"] == []
    assert decoded["meteors"]["decisions"] == []
    assert decoded["algo_version"] == "astro-v1"
    assert decoded["develop"]["peak"] == pytest.approx(0.0)
    assert decoded["stack"]["n_frames"] == 14
    # _stack_result() supplies no sky_validity -- coverage stays an honest "not measured".
    assert decoded["stack"]["coverage"] is None
    assert decoded["stack"]["thin_coverage_fraction"] is None


def test_stack_coverage_and_thin_fraction_reach_the_sidecar_through_the_real_builder():
    """The orchestrator suite only ever stubs ``stack_sidecar_payload`` to ``{}``, so
    nothing else drives it with a real ``StackResult.coverage``/``thin_coverage`` pair
    end to end -- this test is that missing driver."""
    from gomoveshift.pipeline.photos.astro_quality import CoverageStats

    thin = np.array([[True, False, False], [False, False, True]])  # 2 of 6 below floor
    stack = StackResult(
        rgb=np.zeros((2, 3, 3), dtype=np.float32),
        n_frames=5,
        mean_rejected_fraction=np.float64(0.01),
        coverage=CoverageStats(min_count=2, median_count=4.0, below_floor_fraction=1 / 3),
        thin_coverage=thin,
    )
    payload = sidecar_mod.stack_sidecar_payload(
        used_frames=[],
        rejected_frames=[],
        develop=_develop_result(),
        stack=stack,
        decoder="d",
        algo_version="v",
    )

    assert payload["stack"]["coverage"] == {
        "min_count": 2,
        "median_count": 4.0,
        "below_floor_fraction": pytest.approx(1 / 3),
    }
    assert payload["stack"]["thin_coverage_fraction"] == pytest.approx(float(thin.mean()))
    json.dumps(payload)  # still plain-JSON-serialisable with the new fields present


def test_used_frame_records_in_sample_radial_residuals_and_stage_yields():
    # Where the fit's matched pairs actually came from, and whether the outer
    # sensor third has real support -- labeled in_sample (fit-derived, not held-out).
    radial = {
        "center": {"n": 11, "median_px": 0.1, "p95_px": 0.2},
        "middle": {"n": 31, "median_px": 0.2, "p95_px": 0.3},
        "outer": {"n": 8, "median_px": 0.3, "p95_px": 0.5},
    }
    stage_yields = {
        "seed": 20,
        "wide": 45,
        "tighten_10px": 48,
        "tighten_4px": 49,
        "final": 50,
    }
    frame = sidecar_mod.UsedFrame(
        metrics=_metrics("frame003"),
        registration=RegistrationResult(
            transform=np.zeros((2, 3)),
            n_matched=50,
            inlier_fraction=0.9,
            radial_inlier_residuals_px=radial,
            match_yield_by_stage=stage_yields,
        ),
    )
    payload = _min_payload(used_frames=[frame])
    (used,) = payload["frames"]["used"]
    assert used["radial_inlier_residuals_px_in_sample"] == radial
    assert used["match_yield_by_stage"] == stage_yields
    json.dumps(payload)  # still plain-JSON-serialisable


def test_used_frame_residuals_not_swapped_and_meteor_decisions_recorded():
    # distinct non-zero values so a median/p95 swap or wrong attribute would fail
    frame = sidecar_mod.UsedFrame(
        metrics=_metrics("frame002"),
        registration=RegistrationResult(
            transform=np.zeros((2, 3)),
            n_matched=10,
            inlier_fraction=0.5,
            residual_median_px=0.42,
            residual_p95_px=1.87,
            match_yield=0.63,
        ),
    )
    payload = _min_payload(
        used_frames=[frame],
        meteor_reinserted_count=1,
        meteor_reinsertion_decisions=[
            {"path": Path("/cards/night1/frame002.NEF"), "accepted": True},
        ],
    )
    (used,) = payload["frames"]["used"]
    assert used["residual_median_px"] == pytest.approx(0.42)
    assert used["residual_p95_px"] == pytest.approx(1.87)
    assert used["match_yield"] == pytest.approx(0.63)
    (decision,) = payload["meteors"]["decisions"]
    assert decision["path"] == "/cards/night1/frame002.NEF"
    assert decision["accepted"] is True
