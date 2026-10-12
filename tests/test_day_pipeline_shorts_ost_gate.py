"""Pre-burn OST review gate for SHORTS inside ``run_day_pipeline``.

Shorts burn text historically had NO pre-burn gate — it was only RECORDED
POST-encode, so a wrong shorts burn shipped with no chance to catch it
(asymmetric with the LF gate). These tests pin the mirror gate: the shorts
burn text is written for review and requires confirmation before the burn,
and a material change reopens it.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from _state_helpers import make_state
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError
from test_day_pipeline import TestRunDayPipeline as _StageMocks  # aliased: not a test class here
from test_day_pipeline import _make_day_batch, _make_lf_plan, _make_scores, _make_seg

DATE = "2026-02-20"


def _touch_concat(master_path, segments, file_boundaries, output_path, work_dir, **kw):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"x")
    return output_path


def _touch_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"x")
    return output_path


def _patch_and_run(tmp_path, state, *, lf_confirmed=True, shorts_confirmed=False):
    """Run ``run_day_pipeline`` through the shorts stage with heavy calls mocked.

    ``lf_confirmed`` drops the LF ``.confirmed`` so control reaches the shorts
    stage; ``shorts_confirmed`` drops the shorts ``.confirmed``. Returns
    ``(result, exc, mock_render_short, output_base)`` — only the gate signal
    ``OstAuditPendingError`` is swallowed; any other exception propagates.
    """
    from gomoveshift.pipeline import day_pipeline

    day_batch = _make_day_batch(DATE, 3, per_ep_duration=1200.0)
    day_scores = _make_scores(3600, score=0.6)
    output_base = tmp_path / "out"
    _, mocks = _StageMocks()._patch_all_stages(tmp_path)
    lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
    if lf_confirmed:
        (output_base / f"ost_audit_{DATE}.confirmed").write_text("ok")
    if shorts_confirmed:
        (output_base / f"shorts_ost_audit_{DATE}.confirmed").write_text("ok")

    with (
        mocks["encode_day_master"],
        mocks["apply_ost"],
        mocks["_render_short_ffmpeg"] as mock_render_short,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_touch_concat,
        ),
        patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=_touch_montage),
    ):
        exc = None
        result = None
        try:
            result = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test Location, SC",
                config=None,
                lf_plans=lf_plans,
                state=state,
            )
        except OstAuditPendingError as e:
            exc = e
    return result, exc, mock_render_short, output_base


def test_unconfirmed_shorts_burn_halts_before_render(tmp_path):
    """No shorts confirmation → gate blocks the burn and writes the review sidecar."""
    result, exc, mock_render_short, output_base = _patch_and_run(
        tmp_path, make_state(), lf_confirmed=True, shorts_confirmed=False
    )
    assert isinstance(exc, OstAuditPendingError)
    assert result is None
    assert mock_render_short.call_count == 0
    assert (output_base / f"shorts_ost_audit_{DATE}.json").exists()


def test_shorts_confirmed_file_lets_burn_proceed(tmp_path):
    """A ``shorts_ost_audit_<date>.confirmed`` file passes the gate."""
    result, exc, mock_render_short, _ = _patch_and_run(
        tmp_path, make_state(), lf_confirmed=True, shorts_confirmed=True
    )
    assert exc is None
    assert result is not None
    assert mock_render_short.call_count == len(result.short_paths) > 0


def test_material_change_reopens_shorts_gate(tmp_path):
    """A stale prior hash reopens the gate even with a ``.confirmed`` on disk."""
    state = make_state(
        shorts_ost_audit_passed={DATE: True},
        shorts_ost_audit_hash={DATE: "stale-hash-from-prior-run"},
    )
    result, exc, mock_render_short, _ = _patch_and_run(
        tmp_path, state, lf_confirmed=True, shorts_confirmed=True
    )
    assert isinstance(exc, OstAuditPendingError)
    assert result is None
    assert mock_render_short.call_count == 0


def test_ost_audit_confirmed_flag_bypasses_shorts_gate(tmp_path):
    """A THIS-run ``--ost-audit-confirmed`` bypasses both gates and persists the
    marker. A persisted-only confirmation no longer bypasses."""
    state = make_state(ost_audit_confirmed=True)
    result, exc, mock_render_short, _ = _patch_and_run(
        tmp_path, state, lf_confirmed=False, shorts_confirmed=False
    )
    assert exc is None
    assert result is not None
    assert mock_render_short.call_count > 0
    assert state.get("shorts_ost_audit_passed", {}).get(DATE) is True
    assert DATE in (state.get("shorts_ost_audit_hash") or {})


def test_emitted_shorts_audit_omits_measured_lufs(tmp_path):
    """The written sidecar carries the computed speed but no loudness field.

    Nothing in the pipeline measures a short's loudness — shorts ship muted —
    so a permanently ``None`` ``measured_lufs`` would read to a reviewer as
    "measured, and it was silent".
    """
    _, exc, _, output_base = _patch_and_run(
        tmp_path, make_state(), lf_confirmed=True, shorts_confirmed=True
    )
    assert exc is None
    payload = json.loads((output_base / f"shorts_ost_audit_{DATE}.json").read_text())
    assert payload["shorts"]
    for short in payload["shorts"]:
        assert "measured_lufs" not in short
        assert "median_speed_mps" in short


# ── publish runs never label a short from anything but its own clip GPS ─────


_CONFIRMED = {"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True}


def _gps_less_publish(monkeypatch, tmp_path, state):
    from test_shorts_place_source import _run_shorts

    return _run_shorts(
        monkeypatch,
        tmp_path,
        gps_points_by_date={"2026-01-01": []},
        _trim=lambda p, r: [],
        short_ost_text_fn=lambda p, s: "Should Never Be Called",
        state=state,
    )


def test_publish_run_refuses_gps_less_short_before_encode(monkeypatch, tmp_path):
    """The GPS-less anchor (override, county, drive-wide label) is not the clip's
    own place: a publish run refuses it in the derive pass, before any burn."""
    state = {**_CONFIRMED, "youtube_upload": True}
    with pytest.raises(RuntimeError, match=r"rank 1.*publish"):
        _gps_less_publish(monkeypatch, tmp_path, state)


def test_publish_intent_is_read_off_a_pipeline_state_too(monkeypatch, tmp_path):
    state = make_state(youtube_upload=True, **_CONFIRMED)
    with pytest.raises(RuntimeError, match="publish"):
        _gps_less_publish(monkeypatch, tmp_path, state)


def test_audit_run_keeps_the_gps_less_anchor(monkeypatch, tmp_path):
    state = {**_CONFIRMED, "youtube_upload": False}
    entries = _gps_less_publish(monkeypatch, tmp_path, state)
    assert entries[0]["place_source"] == "lf_label"


def test_assert_day_title_stamps_refuses_an_entry_without_start_sec():
    from gomoveshift.pipeline.social_generator import assert_day_title_stamps

    assert_day_title_stamps(DATE, [{"rank": 1, "start_sec": 3.0}, {"rank": 2, "start_sec": 9.0}])
    with pytest.raises(ValueError, match="start_sec"):
        assert_day_title_stamps(DATE, [{"rank": 1, "start_sec": 3.0}, {"rank": 2}])


def test_title_stamp_assert_runs_before_the_shorts_gate(tmp_path):
    """The stamp/capacity check fires upstream of the pre-burn review gate."""
    order = []

    def _stamps(date, entries):
        order.append(("stamps", date, len(entries)))

    def _gate(*a, **kw):
        order.append(("gate",))
        raise OstAuditPendingError("stop")

    with (
        patch("gomoveshift.pipeline.day_pipeline.assert_day_title_stamps", side_effect=_stamps),
        patch("gomoveshift.pipeline.day_pipeline.shorts_gate_or_raise", side_effect=_gate),
    ):
        _, exc, mock_render_short, _ = _patch_and_run(tmp_path, make_state())
    assert isinstance(exc, OstAuditPendingError)
    assert order[0][:2] == ("stamps", DATE) and order[0][2] > 0
    assert order[1] == ("gate",)
    assert mock_render_short.call_count == 0
