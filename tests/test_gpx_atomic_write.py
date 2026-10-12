"""GPX write-site atomicity and pre-set-gpx adoption gate strength.

``concat_episode``/GPX write sites used bare ``open(...).write(...)``,
so a crash mid-write left a truncated GPX in place. The adoption gate at
``extract_and_merge_gpx`` only checked ``exists() and size > 0``, so a
truncated-but-nonzero file was adopted forever and silently corrupted every
downstream surface (OST, chapters, day-grouping).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import gpxpy
import pytest

from gomoveshift.pipeline import _atomic_write
from gomoveshift.pipeline.gopro_gps import extract_gopro_gpx
from gomoveshift.pipeline.ingest.concatenator import (
    _extract_gpx_time_range,
    _merge_gpx_files,
    extract_and_merge_gpx,
)
from gomoveshift.pipeline.ingest.gopro_parser import Episode


def _valid_gpx_text(
    lat: float = 45.0, lon: float = -63.0, when: str = "2026-01-01T12:00:00Z"
) -> str:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)
    seg.points.append(
        gpxpy.gpx.GPXTrackPoint(
            lat, lon, time=datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
        )
    )
    return gpx.to_xml()


class TestMergeGpxFilesAtomic:
    def test_crash_mid_write_leaves_prior_merged_gpx_intact(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A crash while swapping in the merged GPX must not truncate it."""
        output_dir = tmp_path
        episode = Episode(index=1, date="2026-01-01")

        part = output_dir / "part_a.gpx"
        part.write_text(_valid_gpx_text(), encoding="utf-8")

        merged_path = output_dir / "episode_01_2026-01-01.gpx"
        old_content = _valid_gpx_text(lat=1.0, lon=2.0)
        merged_path.write_text(old_content, encoding="utf-8")

        monkeypatch.setattr(
            _atomic_write.os, "replace", MagicMock(side_effect=RuntimeError("simulated crash"))
        )

        with pytest.raises(RuntimeError, match="simulated crash"):
            _merge_gpx_files([part], output_dir, episode)

        assert merged_path.read_text(encoding="utf-8") == old_content
        # Still parseable — never truncated.
        with open(merged_path, encoding="utf-8") as f:
            gpxpy.parse(f)


class TestExtractGpxTimeRangeAtomic:
    def test_crash_mid_write_leaves_prior_slice_gpx_intact(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        anchor = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        source_gpx = tmp_path / "source.gpx"
        source_gpx.write_text(_valid_gpx_text(when="2026-01-01T12:00:05Z"), encoding="utf-8")

        out_path = tmp_path / "slice.gpx"
        old_content = _valid_gpx_text(lat=9.0, lon=9.0)
        out_path.write_text(old_content, encoding="utf-8")

        monkeypatch.setattr(
            _atomic_write.os, "replace", MagicMock(side_effect=RuntimeError("simulated crash"))
        )

        result = _extract_gpx_time_range(
            source_gpx, 0.0, 10.0, tmp_path, "slice.gpx", anchor_time=anchor
        )

        # The function's own try/except swallows the crash; result is None,
        # but that must not mean a half-written file was left behind.
        assert result is None
        assert out_path.read_text(encoding="utf-8") == old_content


def _gopro_overlay_mock_modules(gpx_xml: str) -> dict:
    mock_ffmpeg_inst = MagicMock()
    mock_ffmpeg_inst.ffprobe.return_value.invoke.return_value.stdout = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "index": 0,
                    "disposition": {"default": 1},
                    "width": 1920,
                    "height": 1080,
                    "duration": "30.0",
                    "nb_frames": "900",
                    "avg_frame_rate": "30/1",
                },
                {"codec_type": "audio", "index": 1, "disposition": {"default": 1}},
                {
                    "codec_type": "data",
                    "codec_tag_string": "gpmd",
                    "index": 2,
                    "time_base": "1/600",
                },
            ]
        }
    )
    mock_gpx_obj = MagicMock()
    mock_gpx_obj.to_xml.return_value = gpx_xml
    mock_ffmpeg_mod = MagicMock()
    mock_ffmpeg_mod.FFMPEG.return_value = mock_ffmpeg_inst
    mock_ffmpeg_gopro_mod = MagicMock()
    mock_ffmpeg_gopro_mod.FFMPEGGoPro.return_value.find_frame_duration.return_value = MagicMock()
    mock_framemeta_gpx_mod = MagicMock()
    mock_framemeta_gpx_mod.framemeta_to_gpx.return_value = mock_gpx_obj
    return {
        "gopro_overlay": MagicMock(),
        "gopro_overlay.ffmpeg": mock_ffmpeg_mod,
        "gopro_overlay.ffmpeg_gopro": mock_ffmpeg_gopro_mod,
        "gopro_overlay.framemeta_gpmd": MagicMock(),
        "gopro_overlay.framemeta_gpx": mock_framemeta_gpx_mod,
        "gopro_overlay.gpmf": MagicMock(),
        "gopro_overlay.gpmd_filters": MagicMock(),
        "gopro_overlay.counter": MagicMock(),
        "gopro_overlay.dimensions": MagicMock(),
        "gopro_overlay.timeunits": MagicMock(),
        "gopro_overlay.units": MagicMock(),
    }


class TestExtractGoproGpxAtomic:
    def test_crash_mid_write_leaves_prior_gpx_intact_and_returns_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        video_path = tmp_path / "gopro.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"
        output_dir.mkdir()

        # Pre-existing GPX at the exact path extract_gopro_gpx will target.
        gpx_path = output_dir / f"{video_path.stem}_gopro.gpx"
        old_content = _valid_gpx_text(lat=3.0, lon=4.0)
        gpx_path.write_text(old_content, encoding="utf-8")

        mock_modules = _gopro_overlay_mock_modules('<?xml version="1.0"?><gpx></gpx>')

        monkeypatch.setattr(
            _atomic_write.os, "replace", MagicMock(side_effect=RuntimeError("simulated crash"))
        )

        mock_gpx_obj = mock_modules["gopro_overlay.framemeta_gpx"].framemeta_to_gpx.return_value
        with (
            patch.dict("sys.modules", mock_modules),
            patch("gomoveshift.pipeline.gopro_gps.build_stamped_gpx", return_value=mock_gpx_obj),
        ):
            result = extract_gopro_gpx(video_path, output_dir)

        # extract_gopro_gpx swallows all exceptions and returns None.
        assert result is None
        assert gpx_path.read_text(encoding="utf-8") == old_content


class TestAdoptionGateRejectsTruncatedGpx:
    def test_truncated_but_nonzero_preset_gpx_on_sub_episode_raises(self, tmp_path: Path) -> None:
        """A sub-episode's corrupt-but-nonzero pre-set GPX must fail loudly —
        it has no chapter recordings to fall back to."""
        gpx_path = tmp_path / "sub_gpx.gpx"
        # Nonzero size, no trackpoints — the shape a crash-truncated write leaves.
        gpx_path.write_text('<?xml version="1.0"?><gpx><trk><trkseg>', encoding="utf-8")

        episode = Episode(
            index=2,
            recordings=[],
            date="2026-01-01",
            gpx_path=gpx_path,
            _explicit_total_duration=60.0,
        )

        with pytest.raises(Exception):
            extract_and_merge_gpx(episode, tmp_path)

    def test_truncated_but_nonzero_preset_gpx_on_chapter_derived_episode_falls_back(
        self, tmp_path: Path
    ) -> None:
        """A chapter-derived episode can re-extract, so a corrupt pre-set GPX
        is discarded rather than raised — never returned as adopted."""
        gpx_path = tmp_path / "external.gpx"
        gpx_path.write_text('<?xml version="1.0"?><gpx><trk><trkseg>', encoding="utf-8")

        episode = Episode(index=3, recordings=[], date="2026-01-01", gpx_path=gpx_path)

        result = extract_and_merge_gpx(episode, tmp_path)

        # No chapters to extract from either, so the fallback also yields
        # nothing usable — but the corrupt path itself must never be handed
        # back to callers as a "good" adopted GPX.
        assert result != gpx_path
