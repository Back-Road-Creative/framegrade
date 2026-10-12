"""Montage output verification + pins on two quality-hazard code paths.

From a colour/visual-quality cleanup item:

1. ``build_montage`` verifies the montage it just assembled. Previously a
   montage shipped with zero verification — a silently-empty clip extraction
   or a truncated concat rode through and only surfaced after an irreversible
   multi-hour encode. This is an upstream gate: validate right after
   generation, before the irreversible artifact.
2. The brand-logo overlay in ``ost_applier`` scaled its PNG with ffmpeg's
   default bicubic, the same softening already fixed for the shorts crop.
3. Pins on two symbols a cleanup proposed deleting. Neither could be deleted
   (both have live test references — the trap ``day_pipeline.extract_slice``
   sprang earlier), so the property that keeps each harmless
   is pinned mechanically instead.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import montage_builder


MONTAGE_LOGGER = "gomoveshift.pipeline.montage_builder"
PROD_ROOT = Path(montage_builder.__file__).resolve().parent.parent
PROBE = "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration"


def _peaks(n: int) -> list:
    return [
        SimpleNamespace(source_file_index=0, start_time=100.0 * i, end_time=100.0 * i + 40.0)
        for i in range(n)
    ]


@pytest.fixture
def montage_env(tmp_path):
    """A build_montage call whose ffmpeg work is stubbed to create its outputs,
    so the only real behaviour under test is what happens after the concat."""
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake-master")

    def _fake_trim(master_path, peak, boundaries, output, clip_duration):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"clip")
        return True

    def _fake_concat(clip_files, output_path, list_file):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"montage")
        return True

    return SimpleNamespace(
        master=master,
        out=tmp_path / "montage.mp4",
        work=tmp_path / "work",
        trim=_fake_trim,
        concat=_fake_concat,
        boundaries=[(0.0, 6000.0)],
    )


def _build(env):
    """5 peaks over a 15s budget → 3.0s each → a 15.0s montage."""
    return montage_builder.build_montage(
        master_path=env.master,
        peaks=_peaks(5),
        file_boundaries=env.boundaries,
        output_path=env.out,
        work_dir=env.work,
        total_duration=15.0,
        max_total_duration=15.0,
    )


def _run(env, probed_duration: float):
    with (
        patch.object(montage_builder, "_trim_peak_to", side_effect=env.trim),
        patch.object(montage_builder, "concat_demuxer", side_effect=env.concat),
        patch(PROBE, return_value=probed_duration),
    ):
        return _build(env)


class TestMontageOutputGate:
    """``build_montage`` verifies the assembled montage before returning it."""

    def test_correct_duration_returns_the_montage(self, montage_env):
        assert _run(montage_env, probed_duration=15.0) == montage_env.out

    def test_truncated_montage_is_rejected(self, montage_env, caplog):
        """A montage far shorter than planned means the assembly broke.

        ``build_montage`` already contracts to return ``None`` on failure and
        ``prepend_montage`` skips the LF on ``None``, so the gate reuses that
        path rather than inventing a new failure mode.
        """
        with caplog.at_level(logging.ERROR, logger=MONTAGE_LOGGER):
            result = _run(montage_env, probed_duration=2.0)

        assert result is None, "a 2.0s montage against a 15.0s plan must not ship"
        assert any(
            "montage" in r.getMessage().lower() for r in caplog.records if r.name == MONTAGE_LOGGER
        )

    def test_small_deviation_is_tolerated(self, montage_env):
        """Keyframe snapping moves cut points, so the gate carries a 10%
        tolerance — 15.9s against a 15.0s plan is normal and must pass."""
        assert _run(montage_env, probed_duration=15.9) == montage_env.out

    def test_unprobeable_montage_warns_but_ships(self, montage_env, caplog):
        """``probe_duration`` returns 0.0 as a blanket failure sentinel. An
        unavailable probe is not evidence the montage is broken, so it must not
        fail the build — but it must be visible: the gate did not run."""
        with caplog.at_level(logging.WARNING, logger=MONTAGE_LOGGER):
            result = _run(montage_env, probed_duration=0.0)

        assert result == montage_env.out
        assert any(
            "unverified" in r.getMessage().lower()
            for r in caplog.records
            if r.name == MONTAGE_LOGGER
        )

    def test_gate_uses_the_shared_validator(self, montage_env):
        """``validate_montage_output`` existed with no production caller — the
        wire-or-delete item. Pin the wiring so it cannot rot back to
        dead code."""
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        with (
            patch.object(montage_builder, "_trim_peak_to", side_effect=montage_env.trim),
            patch.object(montage_builder, "concat_demuxer", side_effect=montage_env.concat),
            patch.object(
                PipelineValidator,
                "validate_montage_output",
                wraps=PipelineValidator().validate_montage_output,
            ) as spy,
            patch(PROBE, return_value=15.0),
        ):
            _build(montage_env)

        assert spy.call_count == 1
        assert spy.call_args.kwargs["target_duration"] == pytest.approx(15.0)


def test_logo_overlay_scale_declares_lanczos():
    """ffmpeg's default scaler is bicubic, which softens the logo edges. The
    shorts crop path already pins ``flags=lanczos``; the logo branch was the
    last scale in the OST path still riding the default."""
    from gomoveshift.pipeline import ost_applier

    source = Path(ost_applier.__file__).read_text(encoding="utf-8")
    logo_scales = [ln for ln in source.splitlines() if "scale=240" in ln and "[logo]" in ln]
    assert logo_scales, "logo overlay scale filter not found — did the chain move?"
    for line in logo_scales:
        assert "flags=lanczos" in line, (
            f"logo overlay scale rides ffmpeg's default bicubic: {line.strip()!r}"
        )


def _prod_calls():
    """Yield ``(path, ast.Call)`` for every call in the shipped package."""
    for py in PROD_ROOT.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield py, node


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


class TestQualityHazardPins:
    """Two symbols a cleanup proposed deleting, pinned safe instead."""

    def test_encode_video_has_no_production_call_site(self):
        """``color_filters.encode_video`` is re-exported but never called. It
        presents itself as the complete grade+encode entry point while stamping
        limited-range metadata with no matching range conversion, so the first
        caller to trust it inherits a colour bug. It cannot be deleted (three
        test modules reference it, one asserting it exists by name), so pin that
        nothing in the package calls it.
        """
        callers = [
            f"{py.relative_to(PROD_ROOT)}:{node.lineno}"
            for py, node in _prod_calls()
            if _call_name(node) == "encode_video"
        ]
        assert callers == [], (
            f"color_filters.encode_video gained a production caller: {callers}. It "
            "stamps -color_range tv without squeezing full-range input; give it a "
            "real range conversion before wiring it up."
        )

    def test_packaged_pipeline_never_enables_clip_rendering(self):
        """``ShortsRenderer.render_clip`` encodes libx264 CRF 23 8-bit, keeps
        the audio stream, and applies no vertical framing — violating the shorts
        contract on every axis (exactly 1080x1920, ``-an``). It stays reachable
        only from ``analysis/shorts_extractor``'s ``__main__`` debug entry; the
        packaged pipeline must always pass ``render_clips=False``.
        """
        target = PROD_ROOT / "pipeline" / "shorts_extractor.py"
        tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
        flags = [
            kw.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _call_name(node) == "extract_shorts"
            for kw in node.keywords
            if kw.arg == "render_clips"
        ]

        assert flags, "pipeline/shorts_extractor.py no longer passes render_clips explicitly"
        for value in flags:
            assert isinstance(value, ast.Constant) and value.value is False, (
                "the packaged pipeline enabled ShortsRenderer.render_clip, which ships "
                "16:9 libx264 clips WITH audio — the shorts contract requires exactly "
                "1080x1920 and no audio stream."
            )
