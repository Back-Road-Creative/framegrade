"""The trail film's opening replaces the peak montage on the LF it was filmed from.

The film is rendered from the finished LF, so the swap is a post-film step: the LF's recorded
montage lead is cut off, the film's satellite opening + first map leg (capped at
``montage_max_duration_sec``) is burned with the audited OST and put in its place, and a marker
folded into the LF's fingerprint makes a resume neither redo nor lose it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import trail_opening as to
from gomoveshift.pipeline.inserts import Insert, MediaInfo
from gomoveshift.pipeline.media_api import (
    Deliverables,
    TrailFilm,
    deliverables_from_state,
    deliverables_to_state,
)
from gomoveshift.pipeline.route_story.assembly import FilmTimeline

LEAD, TOTAL, KEYFRAMES = 15.0, 100.0, [0.0, 2.0, 16.0, 18.0, 40.0]
CONFIG = SimpleNamespace(drive=SimpleNamespace(montage_max_duration_sec=15.0))


class _Rig:
    """A day with one rendered LF, its film, and fake ffmpeg/ffprobe/OST steps."""

    def __init__(self, tmp_path: Path, *, end_frame=300, fps=30, inserts=()):
        self.lf_dir = tmp_path / "day_2026-05-01" / "lf"
        self.lf_dir.mkdir(parents=True)
        self.lf = self.lf_dir / "lf_01.mp4"
        self.lf.write_bytes(b"old-lf")
        (self.lf_dir / "lf_01.ost.sha256").write_text("base")
        self.receipt = tmp_path / "film.receipt.json"
        self.receipt.write_text(
            json.dumps({"opening": {"fps": fps, "width": 1920, "height": 1080,
                                    "end_frame": end_frame}})
        )  # fmt: skip
        film = tmp_path / "film.mp4"
        film.write_bytes(b"film")
        self.film = TrailFilm(lf_index=1, path=film, receipt=self.receipt)
        self.d = Deliverables(
            date="2026-05-01", master_path=None, regular_lf_paths=[self.lf], regular_lf_indices=[1]
        )
        self.argvs: list[list[str]] = []
        self.burns: list[dict] = []
        self.burn_ok = True
        self.video_s = None  # the swapped file's video-stream length; None = as planned
        to.record_lead(self.lf_dir, 1, lead_s=LEAD, location="Greene County, NY",
                       crossfade_sec=0.5, brand_logo_path=None, inserts=list(inserts))  # fmt: skip

    def run(self, argv, **_kw):
        self.argvs.append(list(argv))
        if argv[0] == "ffprobe" and "v:0" in argv:
            seen = self.expected if self.video_s is None else self.video_s
            return SimpleNamespace(returncode=0, stdout=f"{seen}\n", stderr="")
        if argv[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="0\n", stderr="")
        Path(argv[-1]).write_bytes(b"x")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def probe(self, path):
        if Path(path).name == "swapped.mp4":
            return MediaInfo(1920, 1080, 30000 / 1001, self.expected)
        return MediaInfo(1920, 1080, 30000 / 1001, TOTAL)

    expected = TOTAL - LEAD + 10.0

    def burn(self, **kw):
        self.burns.append(kw)
        if not self.burn_ok:
            return None
        Path(kw["output_path"]).write_bytes(b"burned")
        return Path(kw["output_path"])

    def apply(self):
        return to.apply_trail_opening(
            self.d, self.film, CONFIG, run=self.run, probe=self.probe,
            keyframes=lambda _p: list(KEYFRAMES), burn=self.burn,
        )  # fmt: skip

    def attach(self):
        return to.attach_trail_opening(
            self.d, self.film, CONFIG, run=self.run, probe=self.probe,
            keyframes=lambda _p: list(KEYFRAMES), burn=self.burn,
        )  # fmt: skip


def test_trail_opening_replaces_montage_when_film_exists(tmp_path):
    rig = _Rig(tmp_path)
    assert rig.attach() is True
    assert rig.lf.read_bytes() == b"x"  # replaced by the joined file
    (burn,) = rig.burns
    assert burn["location"] == "Greene County, NY" and burn["layout"] == "regular_lf"
    assert burn["montage_duration"] == 10.0 and burn["website_cta"] is False
    copies = [
        a for a in rig.argvs if a[0] == "ffmpeg" and "-c" in a and a[a.index("-c") + 1] == "copy"
    ]
    body = next(a for a in copies if str(rig.lf) in a and "-ss" in a)
    assert body[body.index("-ss") + 1] == "16.0"  # first keyframe at or after the 15 s lead
    seam = next(
        a for a in rig.argvs if "-ss" in a and str(rig.lf) in a and "libx265" in " ".join(a)
    )
    assert seam[seam.index("-ss") + 1] == "15.0" and seam[seam.index("-t") + 1] == "1.0"
    assert rig.d.trail_openings == [
        {"lf_index": 1, "opening_s": 10.0, "lead_s": 15.0, "insert_delta_s": -5.0}
    ]
    assert rig.d.trail_skips == []


def test_opening_is_capped_at_the_montage_max_duration(tmp_path):
    rig = _Rig(tmp_path, end_frame=900)  # 30 s of opening + first leg
    rig.expected = TOTAL - LEAD + 15.0
    rig.apply()
    assert rig.burns[0]["montage_duration"] == 15.0


def test_swap_folds_into_the_fingerprint_and_resume_is_a_no_op(tmp_path):
    rig = _Rig(tmp_path)
    assert to.expected_fingerprint(rig.lf_dir, 1, "base") == "base"  # no film: unchanged
    rig.apply()
    stored = (rig.lf_dir / "lf_01.ost.sha256").read_text()
    assert stored != "base" and stored == to.expected_fingerprint(rig.lf_dir, 1, "base")
    rig.argvs.clear()
    rig.burns.clear()
    assert rig.apply() is None  # resume: already swapped
    assert rig.argvs == [] and rig.burns == []
    assert to.expected_fingerprint(rig.lf_dir, 1, "another") != stored  # re-burn is detected


def test_reset_gives_the_re_burned_lf_its_montage_back(tmp_path):
    rig = _Rig(tmp_path)
    rig.apply()
    to.reset(rig.lf_dir, 1)
    assert to.expected_fingerprint(rig.lf_dir, 1, "base") == "base"
    assert not (rig.lf_dir / "lf_01.lead.json").exists()
    assert not (rig.lf_dir / "lf_01.trail_opening.json").exists()
    assert rig.attach() is False and "no montage lead" in rig.d.trail_skips[0]  # montage again


def test_crash_after_the_replace_finishes_without_redoing(tmp_path):
    rig = _Rig(tmp_path)
    key = to._opening_seconds(rig.receipt, 15.0)[1]
    (rig.lf_dir / "lf_01.trail_opening.json").write_text(
        json.dumps({"status": "pending", "key": key, "base_fp": "base",
                    "new_duration": TOTAL, "new_size": rig.lf.stat().st_size})  # replace landed
    )  # fmt: skip
    assert rig.apply() == {
        "lf_index": 1, "opening_s": 10.0, "lead_s": 15.0, "insert_delta_s": -5.0,
    }  # fmt: skip
    assert rig.burns == []
    assert (rig.lf_dir / "lf_01.ost.sha256").read_text() == to.fold("base", key)


def test_crash_before_the_replace_redoes_the_swap_even_at_the_same_length(tmp_path):
    # The opening can be as long as what it replaces, so length alone cannot tell a landed
    # replace from one that never happened; the pending marker's file size can.
    rig = _Rig(tmp_path)
    key = to._opening_seconds(rig.receipt, 15.0)[1]
    (rig.lf_dir / "lf_01.trail_opening.json").write_text(
        json.dumps({"status": "pending", "key": key, "base_fp": "base",
                    "new_duration": TOTAL, "new_size": rig.lf.stat().st_size + 1})
    )  # fmt: skip
    rig.apply()
    assert rig.burns != []  # the swap ran again


def test_unlocatable_opening_is_a_recorded_skip_and_leaves_the_lf(tmp_path):
    rig = _Rig(tmp_path)
    rig.receipt.write_text(json.dumps({"preset": {}}))
    assert rig.attach() is False
    assert rig.lf.read_bytes() == b"old-lf" and rig.burns == []
    (reason,) = rig.d.trail_skips
    assert "opening span" in reason and "keeps its montage" in reason
    rig.attach()
    assert len(rig.d.trail_skips) == 1  # a resume does not stack the same reason


def test_failed_burn_is_a_recorded_skip_and_leaves_the_lf(tmp_path):
    rig = _Rig(tmp_path)
    rig.burn_ok = False
    assert rig.attach() is False
    assert rig.lf.read_bytes() == b"old-lf"
    assert "OST burn" in rig.d.trail_skips[0]
    assert not (rig.lf_dir / ".lf_01_opening").exists()


def test_lf_with_a_pinned_clip_keeps_its_montage(tmp_path):
    clip = Insert(media=Path("clip.mp4"), at_s=120.0)
    rig = _Rig(tmp_path, inserts=[clip])
    assert rig.attach() is False
    assert "insert-clip" in rig.d.trail_skips[0] and rig.burns == []


def test_photo_inserts_ride_with_the_body(tmp_path):
    still = Insert(media=Path("a.jpg"), at_s=40.0)
    rig = _Rig(tmp_path, inserts=[still])
    assert rig.attach() is True  # the stills are in the final LF and are stream-copied with it
    body = next(
        a for a in rig.argvs if str(rig.lf) in a and "-c" in a and a[a.index("-c") + 1] == "copy"
    )
    assert body[body.index("-to") + 1] == str(TOTAL)


def test_insert_inside_the_lead_keeps_the_montage(tmp_path):
    rig = _Rig(tmp_path, inserts=[Insert(media=Path("a.jpg"), at_s=5.0)])
    assert rig.attach() is False and "inside the montage lead" in rig.d.trail_skips[0]


def test_lf_without_a_recorded_lead_is_a_loud_skip(tmp_path):
    rig = _Rig(tmp_path)
    (rig.lf_dir / "lf_01.lead.json").unlink()
    assert rig.attach() is False and "no montage lead" in rig.d.trail_skips[0]


def test_trail_openings_survive_the_state_round_trip(tmp_path):
    d = Deliverables(
        date="2026-05-01", master_path=None, trail_openings=[{"lf_index": 1, "opening_s": 10.0}]
    )
    assert deliverables_from_state(deliverables_to_state(d)).trail_openings == d.trail_openings


def test_timeline_marks_only_the_first_leg():
    t = FilmTimeline()
    t.mark_opening_end(120)
    t.mark_opening_end(900)
    assert t.opening_end_frame == 120


def _burn_filter(tmp_path, **kw):
    """The drawtext chain ``apply_ost`` builds for a 60 s LF, captured before ffmpeg runs."""
    from gomoveshift.pipeline import ost_applier

    clip = tmp_path / "in.mp4"
    clip.write_bytes(b"x")
    built = []

    class _StopError(Exception):
        pass

    real = ost_applier._build_drawtext_filter

    def spy(*a, **k):
        built.append(real(*a, **k))
        raise _StopError

    info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0, "bitrate": 1}
    with (
        patch.object(ost_applier, "_probe_video_info", return_value=info),
        patch.object(ost_applier, "_build_drawtext_filter", spy),
        pytest.raises(_StopError),
    ):
        ost_applier.apply_ost(clip, tmp_path / "out.mp4", "Greene County, NY", "regular_lf",
                              font_path=str(clip), **kw)  # fmt: skip
    return built[0], ost_applier._WEBSITE_TEXT


def test_ost_burn_can_drop_the_website_cta(tmp_path):
    kept, cta = _burn_filter(tmp_path)
    centred = f"text='{cta}':"  # the closing CTA is centred; the corner URL watermark stays
    assert kept.count(cta) == 2 and "x=(w-text_w)/2" in kept.split(centred)[2]
    dropped, _ = _burn_filter(tmp_path, website_cta=False)
    assert dropped.count(cta) == 1 and "x=w-text_w-40" in dropped.split(centred)[1]
    assert "Greene County" in dropped


def test_the_day_pipeline_reads_and_records_the_swap(tmp_path):
    """Burn records the lead; a swapped LF is fresh on resume; a re-burn gives the montage back."""
    from test_drive_inserts_splice import _run_day

    def probe(p):
        return MediaInfo(1920, 1080, 30.0, 130.0 if "with_montage" in Path(p).name else 100.0)

    lf_dir = tmp_path / "out" / "day_2026-02-20" / "regular_lfs"
    with patch("gomoveshift.pipeline.day_pipeline.probe_media", side_effect=probe):
        _, calls, _, _ = _run_day(tmp_path, None)
        assert calls == ["apply_ost"]
        assert json.loads((lf_dir / "lf_01.lead.json").read_text())["lead_s"] == 30.0
        base = (lf_dir / "lf_01.ost.sha256").read_text()
        to._write_json(
            lf_dir / "lf_01.trail_opening.json", {"status": "done", "key": "k", "base_fp": base}
        )
        (lf_dir / "lf_01.ost.sha256").write_text(to.fold(base, "k"))  # what the swap leaves
        _, calls, _, _ = _run_day(tmp_path, None)
        assert calls == []  # the swapped LF is fresh: not re-burned, marker kept
        assert (lf_dir / "lf_01.trail_opening.json").exists()
        (lf_dir / "lf_01.ost.sha256").unlink()  # a stale LF is re-burned
        _, calls, _, _ = _run_day(tmp_path, None)
    assert calls == ["apply_ost"]
    assert not (lf_dir / "lf_01.trail_opening.json").exists()  # the montage is back
    assert (lf_dir / "lf_01.lead.json").exists()  # and its lead recorded afresh
    assert (lf_dir / "lf_01.ost.sha256").read_text() == base


def test_a_video_only_timestamp_collapse_fails_the_swap_and_keeps_the_lf(tmp_path):
    """The format duration still reads right when the LF has audio; the video stream does not."""
    rig = _Rig(tmp_path)
    rig.video_s = 8.0  # the join collapsed the body's video timestamps
    assert rig.attach() is False
    assert rig.lf.read_bytes() == b"old-lf"
    assert "video stream is 8.00s" in rig.d.trail_skips[0]
    assert not (rig.lf_dir / "lf_01.trail_opening.json").exists()
    assert not (rig.lf_dir / ".lf_01_opening").exists()


def test_a_re_rendered_film_swaps_in_place_with_no_re_run(tmp_path):
    """The film step runs just before the day is recorded finished, and a resume never re-burns
    a finished day: a re-rendered film must replace the earlier opening at once, cutting that
    opening's own length rather than the montage's."""
    rig = _Rig(tmp_path)
    assert rig.attach() is True  # the 15 s montage lead became a 10 s opening
    rig.receipt.write_text(
        json.dumps({"opening": {"fps": 30, "width": 1920, "height": 1080, "end_frame": 360}})
    )  # a re-rendered film: a different receipt hash, a 12 s opening
    key = to._opening_seconds(rig.receipt, 15.0)[1]
    rig.expected, rig.argvs = 12.0 + (TOTAL - 10.0), []
    assert rig.attach() is True and rig.d.trail_skips == []
    assert rig.burns[1]["montage_duration"] == 12.0
    seam = next(a for a in rig.argvs if "-ss" in a and "libx265" in " ".join(a) and "-t" in a)
    assert seam[seam.index("-ss") + 1] == "10.0"  # the earlier opening ends at 10 s
    assert (rig.lf_dir / "lf_01.ost.sha256").read_text() == to.fold("base", key)
    assert rig.d.trail_openings == [
        {"lf_index": 1, "opening_s": 12.0, "lead_s": 10.0, "insert_delta_s": -3.0}
    ]
    assert rig.apply() is None and len(rig.burns) == 2  # a resume keeps it


def test_a_failed_re_swap_says_the_lf_keeps_its_earlier_opening(tmp_path):
    # An already-swapped LF no longer has a montage: the skip must name what it really keeps.
    rig = _Rig(tmp_path)
    assert rig.attach() is True
    rig.receipt.write_text(
        json.dumps({"opening": {"fps": 30, "width": 1920, "height": 1080, "end_frame": 360}})
    )
    rig.burn_ok = False
    assert rig.attach() is False
    (reason,) = rig.d.trail_skips
    assert "keeps its earlier trail opening" in reason and "montage" not in reason


def test_the_opening_keeps_the_lf_frame_rate_as_an_exact_rational(tmp_path):
    """fps=29.97003 gives the opening a 1/2997003 timebase; the stream-copied body (1/30000)
    then joins with collapsed timestamps -- real ffmpeg, a 40 s LF plays as 8 s of video."""
    rig = _Rig(tmp_path)
    rig.apply()
    cut = next(a for a in rig.argvs if str(rig.film.path) in a and "-vf" in a)
    assert "fps=30000/1001" in cut[cut.index("-vf") + 1]


def test_the_opening_burn_gets_the_brands_media_profile(tmp_path, monkeypatch):
    """The phases call the swap without a profile; the brand's font and text must still apply."""
    profile = object()
    monkeypatch.setattr(to, "load_media_profile", lambda slug: profile)
    monkeypatch.setattr(to, "current_brand", lambda: "gomoveshift")
    rig = _Rig(tmp_path)
    rig.apply()
    assert rig.burns[0]["media_profile"] is profile


def test_the_recorded_insert_times_move_with_the_new_lead(tmp_path):
    """Phase D's insert chapter markers read the planned times: a swapped lead moves them."""
    from _state_helpers import make_state
    from gomoveshift.pipeline.drive_phases import phase_d_metadata

    still = Insert(media=Path("a.jpg"), at_s=40.0)
    rig = _Rig(tmp_path, inserts=[still])
    rig.d.inserts = {1: [{"kind": "still", "media": "a.jpg", "at_s": 40.0, "style": "full"}]}
    assert rig.attach() is True  # a 10 s opening replaces a 15 s lead
    wire = json.loads(json.dumps(deliverables_to_state(rig.d)))
    state = make_state(per_day_completed={"2026-05-01": {"deliverables": wire}})
    assert phase_d_metadata._lf_inserts(state)[1][0]["at_s"] == 35.0
