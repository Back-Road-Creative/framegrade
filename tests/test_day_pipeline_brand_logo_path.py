"""Regression test for the brand_logo_path threading bug.

The logo-overlay capability of ost_applier.apply_ost was added but
day_pipeline.run_day_pipeline read the field from the wrong place
(`config.brand_logo_path` instead of `config.drive.brand_logo_path`),
so the resolved path was always None and the simple -vf branch was
taken with no logo input. The ost_applier "missing logo" warning never
fired because `bool(brand_logo_path)` was False before the file check.

The first real run shipped LFs without the logo
(template-match MSE 10,320, threshold <2000 for present). This test
would have caught it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from _state_helpers import make_state


def _make_config(brand_logo_path: str | None = "/tmp/fake_logo.png"):
    """Return a minimal config exposing config.drive.brand_logo_path."""
    drive = SimpleNamespace(
        brand_logo_path=brand_logo_path,
        # Other fields day_pipeline touches — keep minimal stubs
        shorts_target_count=9,
        shorts_min_gap_sec=60.0,
        skip_audio=False,
        # intermediate_folder etc. not read here
    )
    return SimpleNamespace(drive=drive)


class TestBrandLogoPathThreading:
    """run_day_pipeline must read brand_logo_path from config.drive, not config."""

    def test_apply_ost_receives_brand_logo_path_from_config_drive(self, tmp_path):
        """When config.drive.brand_logo_path is set, apply_ost gets that exact string."""
        from gomoveshift.pipeline import day_pipeline

        cfg = _make_config(brand_logo_path="/assets/brand_logo.png")

        # Minimal plumbing: one LF plan with one segment, one montage peak
        lf_plan = SimpleNamespace(
            index=1,
            segments=[SimpleNamespace(source_file_index=0, start=0.0, end=60.0)],
            montage_peaks=[SimpleNamespace(source_file_index=0, start=10.0, end=12.0)],
        )
        day_batch = SimpleNamespace(
            date="2026-02-20",
            episodes=[
                SimpleNamespace(
                    index=1,
                    concat_path=tmp_path / "ep.mp4",
                    total_duration=300.0,
                )
            ],
        )
        (tmp_path / "ep.mp4").write_bytes(b"\x00")

        captured_calls: list[dict] = []

        def fake_apply_ost(**kwargs):
            captured_calls.append(kwargs)
            # Return the (non-existent) output_path so caller's None-check passes
            return kwargs.get("output_path")

        # Stub out everything the day pipeline would otherwise do for real
        with (
            patch.object(day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"),
            patch.object(
                day_pipeline,
                "extract_and_concat_segments",
                return_value=tmp_path / "lf_01_segments.mp4",
            ),
            patch.object(
                day_pipeline, "prepend_montage", return_value=tmp_path / "lf_01_with_montage.mp4"
            ),
            patch.object(day_pipeline, "apply_ost", side_effect=fake_apply_ost),
            patch.object(day_pipeline, "plan_shorts", return_value=[]),
            patch.object(day_pipeline, "compute_day_file_boundaries", return_value=[(0.0, 300.0)]),
        ):
            (tmp_path / "master.mp4").write_bytes(b"\x00")
            (tmp_path / "lf_01_segments.mp4").write_bytes(b"\x00")
            (tmp_path / "lf_01_with_montage.mp4").write_bytes(b"\x00")
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=None,
                output_dir=tmp_path / "out",
                location="Longwood",
                config=cfg,
                lf_plans=[lf_plan],
                state=make_state(ost_audit_confirmed=True),
            )

        # At least one apply_ost call for the regular_lf with the exact path threaded through
        regular_lf_calls = [c for c in captured_calls if c.get("layout") == "regular_lf"]
        assert len(regular_lf_calls) == 1, (
            f"expected 1 regular_lf apply_ost call, got {len(regular_lf_calls)}"
        )
        call = regular_lf_calls[0]
        assert call.get("brand_logo_path") == "/assets/brand_logo.png", (
            f"apply_ost did not receive brand_logo_path from config.drive — got "
            f"brand_logo_path={call.get('brand_logo_path')!r}"
        )

    def test_missing_brand_logo_path_does_not_warn(self, tmp_path, caplog):
        """A missing brand_logo_path must NOT warn: regular_lf (the only LF
        layout) sets show_logo=False, so the brand logo is intentionally
        disabled on the YouTube surface and a missing path is expected, not a
        misconfiguration."""
        from gomoveshift.pipeline import day_pipeline

        cfg = _make_config(brand_logo_path=None)

        lf_plan = SimpleNamespace(
            index=1,
            segments=[SimpleNamespace(source_file_index=0, start=0.0, end=60.0)],
            montage_peaks=[],
        )
        day_batch = SimpleNamespace(
            date="2026-02-20",
            episodes=[
                SimpleNamespace(
                    index=1,
                    concat_path=tmp_path / "ep.mp4",
                    total_duration=300.0,
                )
            ],
        )
        (tmp_path / "ep.mp4").write_bytes(b"\x00")

        with (
            patch.object(day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"),
            patch.object(
                day_pipeline, "extract_and_concat_segments", return_value=tmp_path / "seg.mp4"
            ),
            patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
            patch.object(day_pipeline, "apply_ost", return_value=tmp_path / "lf_01.mp4"),
            patch.object(day_pipeline, "plan_shorts", return_value=[]),
            patch.object(day_pipeline, "compute_day_file_boundaries", return_value=[(0.0, 300.0)]),
            caplog.at_level("WARNING", logger="gomoveshift.pipeline.day_pipeline"),
        ):
            (tmp_path / "master.mp4").write_bytes(b"\x00")
            (tmp_path / "seg.mp4").write_bytes(b"\x00")
            (tmp_path / "wm.mp4").write_bytes(b"\x00")
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=None,
                output_dir=tmp_path / "out",
                location="Longwood",
                config=cfg,
                lf_plans=[lf_plan],
                state=make_state(ost_audit_confirmed=True),
            )

        assert not any(
            "brand_logo_path" in r.message.lower() or "brand watermark" in r.message.lower()
            for r in caplog.records
        ), (
            "a missing brand_logo_path must NOT warn (regular_lf disables the "
            f"logo); got: {[r.message for r in caplog.records]}"
        )
