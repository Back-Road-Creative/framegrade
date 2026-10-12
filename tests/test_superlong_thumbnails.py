"""Each superlong part gets its OWN thumbnail, stamped with the part number
— reusing the day's LF thumbnail would show identical
images for every part of a multi-part day in subscriptions, worse than the
YouTube auto-picked frame. Mirrors the LF two-part shape: generation runs
upstream of the upload, attach runs after (needs a video ID).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline import youtube_sidecar
from gomoveshift.pipeline.drive_phases.phase_d import (
    _attach_superlong_thumbnails,
    _generate_superlong_thumbnails,
)

from _state_helpers import make_state
from conftest import fake_generate_lf_thumbnail  # type: ignore[import-not-found]

GEN_PATH = "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail"


def _part_record(tmp_path, idx, part_count, location="Kerry Coast"):
    path = tmp_path / f"sl_{idx:02d}.mp4"
    path.write_bytes(b"x")
    return {
        "path": str(path),
        "part_index": idx,
        "part_count": part_count,
        "location": location,
        "title": f"{location} — Part {idx} of {part_count}",
        "duration_sec": 3600.0,
    }


def _state(tmp_path, n=3, date="2026-03-01"):
    records = [_part_record(tmp_path, i, n) for i in range(1, n + 1)]
    return make_state(superlong_parts={date: {"part_count": n, "records": records}})


class TestGenerateSuperlongThumbnails:
    def test_empty_plan_returns_silently(self, tmp_path):
        _generate_superlong_thumbnails(state=make_state(), output_base=tmp_path)
        assert not (tmp_path / "thumbnails").exists()

    def test_each_part_gets_a_distinct_thumbnail_stamped_with_its_number(self, tmp_path):
        state = _state(tmp_path, n=3)
        subtitles = []

        def _fake_gen(*, output_dir, subtitle=None, **kw):
            subtitles.append(subtitle)
            output_dir.mkdir(parents=True, exist_ok=True)
            variant = output_dir / "v.jpg"
            variant.write_bytes(f"bytes-{subtitle}".encode())
            return variant

        with patch(GEN_PATH, side_effect=_fake_gen):
            _generate_superlong_thumbnails(state=state, output_base=tmp_path)

        thumbs = sorted((tmp_path / "thumbnails").glob("sl_2026-03-01_*.jpg"))
        assert len(thumbs) == 3
        assert len({p.read_bytes() for p in thumbs}) == 3  # each part distinct
        assert subtitles == ["Part 1 of 3", "Part 2 of 3", "Part 3 of 3"]

    def test_filename_cannot_collide_with_lf_thumbnails(self, tmp_path):
        state = _state(tmp_path, n=1)
        with patch(GEN_PATH, side_effect=fake_generate_lf_thumbnail):
            _generate_superlong_thumbnails(state=state, output_base=tmp_path)
        names = [p.name for p in (tmp_path / "thumbnails").glob("*.jpg")]
        assert names == ["sl_2026-03-01_01.jpg"]


class TestAttachSuperlongThumbnails:
    def _write_thumb(self, tmp_path, date, idx):
        thumb = tmp_path / "thumbnails" / f"sl_{date}_{idx:02d}.jpg"
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"jpegbytes")
        return thumb

    def test_attach_uses_the_right_video_id_for_the_right_part(self, tmp_path):
        state = _state(tmp_path, n=2)
        t1 = self._write_thumb(tmp_path, "2026-03-01", 1)
        t2 = self._write_thumb(tmp_path, "2026-03-01", 2)
        youtube_sidecar.save_superlong_sidecar(
            tmp_path,
            {
                "2026-03-01:01": {"video_id": "VID-1"},
                "2026-03-01:02": {"video_id": "VID-2"},
            },
        )
        yt_mock = MagicMock()

        _attach_superlong_thumbnails(state=state, output_base=tmp_path, yt=yt_mock)

        yt_mock.set_thumbnail.assert_any_call("VID-1", str(t1))
        yt_mock.set_thumbnail.assert_any_call("VID-2", str(t2))
        assert yt_mock.set_thumbnail.call_count == 2

    def test_no_upload_entry_or_no_thumbnail_skips_quietly(self, tmp_path):
        state = _state(tmp_path, n=1)
        self._write_thumb(tmp_path, "2026-03-01", 1)  # thumb exists, no sidecar entry
        yt_mock = MagicMock()

        _attach_superlong_thumbnails(state=state, output_base=tmp_path, yt=yt_mock)

        yt_mock.set_thumbnail.assert_not_called()

    def test_attach_failure_is_recorded_not_raised(self, tmp_path):
        state = _state(tmp_path, n=1)
        self._write_thumb(tmp_path, "2026-03-01", 1)
        youtube_sidecar.save_superlong_sidecar(tmp_path, {"2026-03-01:01": {"video_id": "VID-1"}})
        yt_mock = MagicMock()
        yt_mock.set_thumbnail.side_effect = RuntimeError("quota exhausted")

        _attach_superlong_thumbnails(state=state, output_base=tmp_path, yt=yt_mock)  # no raise

        entry = youtube_sidecar.load_superlong_sidecar(tmp_path)["2026-03-01:01"]
        assert "quota exhausted" in entry["thumbnail_attach_failed"]


class TestAttachSuperlongThumbnailsMasterIdentity:
    """The thumbnail entry gets the same cut-from-master identity +
    attestation cross-check as the LF provenance sidecar — best-effort,
    never gating."""

    def _write_thumb(self, tmp_path, date, idx):
        thumb = tmp_path / "thumbnails" / f"sl_{date}_{idx:02d}.jpg"
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"jpegbytes")
        return thumb

    def _write_attestation(self, tmp_path, date, master_fingerprint=None):
        day_dir = tmp_path / "_stages" / f"day_{date}"
        day_dir.mkdir(parents=True)
        payload = {"schema": "gms.privacy_attestation/v1", "date": date}
        if master_fingerprint is not None:
            payload["master_fingerprint"] = master_fingerprint
        (day_dir / f"day_{date}_privacy_attestation.json").write_text(json.dumps(payload))

    def _run(self, tmp_path, *, per_day_completed=None):
        state = _state(tmp_path, n=1)
        state.per_day_completed = per_day_completed or {}
        self._write_thumb(tmp_path, "2026-03-01", 1)
        youtube_sidecar.save_superlong_sidecar(tmp_path, {"2026-03-01:01": {"video_id": "VID-1"}})
        _attach_superlong_thumbnails(state=state, output_base=tmp_path, yt=MagicMock())
        return youtube_sidecar.load_superlong_sidecar(tmp_path)["2026-03-01:01"]

    def test_matching_fingerprint_verifies_true(self, tmp_path):
        self._write_attestation(tmp_path, "2026-03-01", master_fingerprint="123:456")
        entry = self._run(
            tmp_path, per_day_completed={"2026-03-01": {"master_fingerprint": "123:456"}}
        )
        assert entry["cut_from_master_fingerprint"] == "123:456"
        assert entry["master_attestation_verified"] is True

    def test_mismatched_fingerprint_verifies_false(self, tmp_path):
        self._write_attestation(tmp_path, "2026-03-01", master_fingerprint="999:999")
        entry = self._run(
            tmp_path, per_day_completed={"2026-03-01": {"master_fingerprint": "123:456"}}
        )
        assert entry["master_attestation_verified"] is False
        assert entry["null_reasons"].get("master_attestation_verified")

    def test_no_attestation_on_disk_is_none(self, tmp_path):
        entry = self._run(
            tmp_path, per_day_completed={"2026-03-01": {"master_fingerprint": "123:456"}}
        )
        assert entry["master_attestation_verified"] is None
        assert entry["null_reasons"].get("master_attestation_verified")

    def test_no_per_day_record_is_none_with_reason(self, tmp_path):
        entry = self._run(tmp_path, per_day_completed={})
        assert entry["cut_from_master_fingerprint"] is None
        assert entry["null_reasons"].get("cut_from_master_fingerprint")
