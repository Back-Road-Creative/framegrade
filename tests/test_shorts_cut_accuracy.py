"""Batched shorts cut where the scorer said, not one GOP earlier.

Legacy single-video path. `render_shorts_batched` stream-copies each scored
window out of the source and then encodes the small segment. Stream copy cannot
start on a non-keyframe, so `-ss <start> -i src -t <dur> -c copy` silently
snaps the segment start BACK to the previous keyframe — up to a full GOP
(~2.002s at 60 frames / 29.97 fps) of pre-peak footage. Phase 2 then encoded
that segment from `start_time=0.0` on the comment "the segment already starts
at the right point", which was false: every batched short opened up to two
seconds before the moment that was scored, and lost the same amount off its
tail because `-t` counted from the earlier keyframe.

The fix keeps the copy on a keyframe (it has no choice) but makes the
displacement EXPLICIT: probe the source's keyframes, extract from the keyframe
at or before the scored start with the lead-in added to the duration, then let
the phase-2 re-encode — which seeks accurately — trim the lead-in off. No
footage is lost and the shipped clip starts on the scored frame.

When the keyframe probe is unavailable the lead-in is zero and behaviour is
exactly as before, logged loudly rather than silently.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gomoveshift.pipeline import shorts_extractor as se


# A 2.002s GOP, the day master's nominal cadence.
GOP = 2.002
KEYFRAMES = [round(i * GOP, 6) for i in range(200)]


@pytest.fixture
def fake_source(tmp_path):
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00" * 1024)
    return src


def _spec(tmp_path, index=0, start_time=61.0, duration=35.0):
    return {
        "index": index,
        "start_time": start_time,
        "duration": duration,
        "location": "County Kerry, Ireland",
        "subscribe_text": "Subscribe for more",
        "output_path": str(tmp_path / f"short_{index:02d}.mp4"),
    }


# ── The lead-in calculation ──────────────────────────────────────────────


class TestKeyframeLeadIn:
    def test_lead_in_is_the_distance_back_to_the_previous_keyframe(self):
        # 61.0 sits between keyframes 30 (60.06) and 31 (62.062).
        prev = max(k for k in KEYFRAMES if k <= 61.0)
        assert se._keyframe_lead_in(61.0, KEYFRAMES) == pytest.approx(61.0 - prev)

    def test_start_already_on_a_keyframe_has_no_lead_in(self):
        assert se._keyframe_lead_in(KEYFRAMES[30], KEYFRAMES) == pytest.approx(0.0)

    def test_lead_in_never_exceeds_one_gop(self):
        for start in (0.5, 7.3, 61.0, 123.456, 300.9):
            assert 0.0 <= se._keyframe_lead_in(start, KEYFRAMES) < GOP + 1e-3

    def test_unknown_keyframes_give_no_lead_in(self):
        """Probe failure must degrade to today's behaviour, not guess."""
        assert se._keyframe_lead_in(61.0, None) == pytest.approx(0.0)
        assert se._keyframe_lead_in(61.0, []) == pytest.approx(0.0)

    def test_start_before_the_first_keyframe_gives_no_lead_in(self):
        assert se._keyframe_lead_in(0.0, KEYFRAMES) == pytest.approx(0.0)


# ── Wiring through the batched renderer ──────────────────────────────────


def _run_batched(fake_source, tmp_path, specs, keyframes):
    with (
        patch.object(se, "probe_keyframes", return_value=keyframes) as mock_probe,
        patch.object(se, "extract_segment_streamcopy", return_value=True) as mock_extract,
        patch.object(se, "render_short_ffmpeg", return_value=True) as mock_render,
        patch.object(se, "_probe_video_stream_duration", return_value=0.0),
    ):
        results = se.render_shorts_batched(
            source_video=fake_source,
            shorts_specs=specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )
    return results, mock_probe, mock_extract, mock_render


class TestBatchedRenderCutAccuracy:
    def test_extract_starts_on_the_keyframe_and_keeps_the_whole_window(self, fake_source, tmp_path):
        specs = [_spec(tmp_path, start_time=61.0, duration=35.0)]
        _, _, mock_extract, _ = _run_batched(fake_source, tmp_path, specs, KEYFRAMES)

        kwargs = mock_extract.call_args.kwargs
        prev_kf = max(k for k in KEYFRAMES if k <= 61.0)
        lead_in = 61.0 - prev_kf
        assert kwargs["start_time"] == pytest.approx(prev_kf), (
            "stream copy must start ON a keyframe, not at an arbitrary time it "
            "will silently snap back from"
        )
        assert kwargs["duration"] == pytest.approx(35.0 + lead_in), (
            "duration must absorb the lead-in or the clip loses its tail"
        )

    def test_encode_trims_the_lead_in_instead_of_starting_at_zero(self, fake_source, tmp_path):
        specs = [_spec(tmp_path, start_time=61.0, duration=35.0)]
        _, _, _, mock_render = _run_batched(fake_source, tmp_path, specs, KEYFRAMES)

        kwargs = mock_render.call_args.kwargs
        prev_kf = max(k for k in KEYFRAMES if k <= 61.0)
        assert kwargs["start_time"] == pytest.approx(61.0 - prev_kf)
        assert kwargs["duration"] == pytest.approx(35.0)

    def test_shipped_window_still_equals_the_scored_window(self, fake_source, tmp_path):
        """extract_start + encode_start == the scored start; duration unchanged."""
        specs = [_spec(tmp_path, start_time=61.0, duration=35.0)]
        _, _, mock_extract, mock_render = _run_batched(fake_source, tmp_path, specs, KEYFRAMES)

        shipped_start = (
            mock_extract.call_args.kwargs["start_time"] + mock_render.call_args.kwargs["start_time"]
        )
        assert shipped_start == pytest.approx(61.0)
        assert mock_render.call_args.kwargs["duration"] == pytest.approx(35.0)

    def test_source_is_probed_once_for_the_whole_batch(self, fake_source, tmp_path):
        specs = [_spec(tmp_path, index=i, start_time=61.0 + 60 * i) for i in range(4)]
        _, mock_probe, _, _ = _run_batched(fake_source, tmp_path, specs, KEYFRAMES)
        assert mock_probe.call_count == 1

    def test_probe_failure_keeps_the_previous_behaviour(self, fake_source, tmp_path):
        specs = [_spec(tmp_path, start_time=61.0, duration=35.0)]
        results, _, mock_extract, mock_render = _run_batched(fake_source, tmp_path, specs, None)

        assert results == {0: True}
        assert mock_extract.call_args.kwargs["start_time"] == pytest.approx(61.0)
        assert mock_extract.call_args.kwargs["duration"] == pytest.approx(35.0)
        assert mock_render.call_args.kwargs["start_time"] == pytest.approx(0.0)

    def test_probe_failure_is_logged_not_silent(self, fake_source, tmp_path, caplog):
        specs = [_spec(tmp_path, start_time=61.0)]
        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.shorts_extractor"):
            _run_batched(fake_source, tmp_path, specs, None)
        messages = [r.getMessage() for r in caplog.records]
        assert any("keyframe" in m.lower() for m in messages), messages

    def test_no_specs_does_not_probe(self, fake_source, tmp_path):
        _, mock_probe, _, _ = _run_batched(fake_source, tmp_path, [], KEYFRAMES)
        assert mock_probe.call_count == 0
