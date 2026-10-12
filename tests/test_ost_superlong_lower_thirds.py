"""Superlong recurring LOCATION lower-thirds — burned in the SAME OST
re-encode pass as the opener/end-card (no extra encode generation). See
ost_applier._prepare_lower_thirds / _build_drawtext_filter."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier as oa
from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

from ost_subprocess_harness import patch_ost_run

# Byte-identical regular_lf filtergraph pins, captured BEFORE the
# lower-thirds feature existed (no lower_thirds kwarg passed at all).
_MONTAGE_SHA = "b6b244a063feceaab85382c730ac0bd6d81e80ef792937719052a1967afd586c"  # pragma: allowlist secret (filtergraph digest, not a credential)
_LEGACY_SHA = "1ec8b169829938ce872500efe5a4cafe3598e929cf7af61aac85dd04a334b8bf"  # pragma: allowlist secret (filtergraph digest, not a credential)
_STUB_PROBE = {"width": 1920, "height": 1080, "fps": 30, "bitrate_kbps": 12000, "duration": 14400.0}


@pytest.fixture(autouse=True)
def _accept_burned_artifact(monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.pipeline.ost_applier.validate_video",
        lambda *_a, **_kw: VideoValidation(valid=True),
    )


@pytest.mark.parametrize(
    "kwargs,width,duration,expected_sha",
    [
        (dict(montage_duration=15.0, crossfade_sec=0.5), 3840, 817.0, _MONTAGE_SHA),
        ({}, 1920, 14400.0, _LEGACY_SHA),  # superlong's actual (no-montage) shape
        (dict(lower_thirds=[]), 1920, 14400.0, _LEGACY_SHA),  # explicit empty is also a no-op
    ],
)
def test_regular_lf_byte_identical(kwargs, width, duration, expected_sha):
    """No production caller passes lower_thirds — output must not change."""
    chain = oa._build_drawtext_filter(
        location="Sample Location",
        layout=oa.LAYOUTS["regular_lf"],
        font_path=oa.FONT_PATH,
        video_width=width,
        video_duration=duration,
        **kwargs,
    )
    assert hashlib.sha256(chain.encode()).hexdigest() == expected_sha


def _prep(entries, **kw):
    defaults = dict(
        video_duration=14400.0,
        opener_end_sec=10.0,
        cta_start_sec=14393.5,
        duration_sec=8.0,
        min_gap_sec=45.0,
    )
    defaults.update(kw)
    return oa._prepare_lower_thirds(entries, **defaults)


def test_n_transitions_produce_n_windows_order_independent():
    entries = [("Yemassee, SC", 1000.0), ("Beaufort, SC", 300.0)]
    assert _prep(entries) == [
        ("Beaufort, SC", 300.0, 308.0),
        ("Yemassee, SC", 1000.0, 1008.0),
    ]


@pytest.mark.parametrize(
    "entries,kw",
    [
        ([("TooEarly", 5.0), ("Fine", 500.0)], dict(opener_end_sec=10.0)),
        ([("Fine", 500.0), ("TooLate", 14390.0)], dict(cta_start_sec=14393.5)),
    ],
)
def test_never_overlaps_opener_or_cta(entries, kw):
    assert [label for label, *_ in _prep(entries, **kw)] == ["Fine"]


def test_minimum_gap_enforced_between_consecutive_entries():
    # 8s duration -> first window ends at 508; second (10s later, <45s min
    # gap) is dropped; third clears the gap from FIRST (kept), not the drop.
    prepared = _prep([("First", 500.0), ("TooSoon", 510.0), ("Ok", 560.0)])
    assert [label for label, *_ in prepared] == ["First", "Ok"]


@pytest.mark.parametrize("bad_entries", [[("", 300.0)], [("   ", 300.0)]])
def test_empty_or_whitespace_label_hard_fails(bad_entries):
    with pytest.raises(ValueError, match="OST-EMPTY-LOWER-THIRD-LABEL"):
        _prep(bad_entries)


def test_no_entries_returns_empty():
    assert _prep(None) == []


@pytest.fixture
def _apply_ost_args(tmp_path):
    in_mp4 = tmp_path / "in.mp4"
    in_mp4.write_bytes(b"\x00" * 1024)
    return dict(input_path=in_mp4, output_path=tmp_path / "out.mp4", location="Test Drive")


@patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_STUB_PROBE)
@patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True)
def test_apply_ost_hard_fails_on_empty_lower_third_label(_exists, _probe, _apply_ost_args):
    with pytest.raises(ValueError, match="OST-EMPTY-LOWER-THIRD-LABEL"):
        oa.apply_ost(layout="regular_lf", lower_thirds=[("  ", 300.0)], **_apply_ost_args)


@patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_STUB_PROBE)
@patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True)
def test_apply_ost_burns_lower_thirds_into_vfilter(_exists, _probe, _apply_ost_args):
    with patch_ost_run() as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        oa.apply_ost(layout="regular_lf", lower_thirds=[("Beaufort, SC", 300.0)], **_apply_ost_args)
    cmd = mock_run.call_args[0][0]
    assert "Beaufort, SC" in cmd[cmd.index("-vf") + 1]


# ── burn-time backstops: no repeat label, no internal separator ─────────────


def test_adjacent_duplicate_label_dropped():
    # The production pair: the same town 76s apart. The min-gap rule is a
    # strobe limiter (68s measured gap here, well over 45s) and never a dedupe,
    # so the repeat has to be caught explicitly.
    assert _prep(
        [
            ("Johnson City, Washington County", 404.0),
            ("Johnson City, Washington County", 480.0),
        ]
    ) == [("Johnson City, Washington County", 404.0, 412.0)]


def test_duplicate_survives_a_dropped_neighbour():
    # The upstream collapse in chapter_town_events structurally cannot catch
    # this: A and A are not adjacent there, and B is only dropped HERE by the
    # min-gap rule — which is why the check must exist at both sites.
    assert [label for label, *_ in _prep([("A", 500.0), ("B", 510.0), ("A", 560.0)])] == ["A"]


def test_non_adjacent_repeat_is_kept():
    # Only an immediate repeat is dropped; a genuine return to a town later in
    # the drive still gets its own lower third.
    labels = [label for label, *_ in _prep([("A", 500.0), ("B", 600.0), ("A", 700.0)])]
    assert labels == ["A", "B", "A"]


def test_semicolon_in_label_hard_fails():
    # ";" is the ffmpeg filtergraph chain separator and escape_ffmpeg_text does
    # not escape it, so an OSM concurrency join reaching a drawtext label is a
    # structural hazard, not just ugly text. Upstream now makes this
    # unreachable; this is the backstop that keeps it that way.
    with pytest.raises(ValueError, match="OST-INTERNAL-SEPARATOR-LOWER-THIRD-LABEL"):
        _prep([("I 81;US 58, Washington County", 300.0)])
