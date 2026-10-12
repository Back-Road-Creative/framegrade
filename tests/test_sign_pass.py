"""Phase A road-sign OCR evidence pass.

The pass records evidence ONLY — it never writes a location anywhere. These
tests are fully hermetic: every heavy seam (frame grab, OCR, claim parsing) is
injected, so nothing here decodes video, shells out to ffmpeg, loads EasyOCR,
or touches the network. The one real collaborator is the cheap colour/shape
sign-region finder, which runs against small synthetic numpy frames.

Covered:
  * day-master offsets: an episode second in the SECOND episode of a day maps
    to ``ts_day = first_ep_duration + t_local`` and keeps the GLOBAL episode
    index (the pair the whole verification graph buckets on);
  * the per-day sample cap is never exceeded and samples stride evenly across
    the whole day rather than clustering at the start;
  * stationary seconds are excluded when the episode has GPS;
  * a GPS-less episode still yields hits, with ``lat``/``lon`` ``None``;
  * hits with zero parsed claims are discarded (no OCR noise persisted);
  * a grab/OCR failure on one second is skipped, never raised;
  * ``SignHit`` round-trips through the state serializer unchanged;
  * Phase A wiring: flag off / flag on / already-populated back-fill skip.
"""

from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gomoveshift.analysis import sign_pass as sign_pass_mod
from gomoveshift.analysis.sign_evidence import (
    SignClaim,
    SignHit,
    hits_by_date_from_state,
    hits_by_date_to_dicts,
)
from gomoveshift.analysis.sign_pass import collect_sign_hits
from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import phase_a as phase_a_mod
from gomoveshift.pipeline.drive_phases.phase_a import PhaseA
from gomoveshift.pipeline.drive_state_model import PipelineState

# ── config proxy ──────────────────────────────────────────────────────────
# ``DriveConfig`` is a pydantic model that refuses assignment of a field it
# does not declare, and the sign-verify knobs land separately. Proxying
# keeps these tests decoupled from that timing AND exercises the defensive
# ``getattr(config.drive, ...)`` reads in the pass itself.


class _DriveProxy:
    def __init__(self, drive, overrides):
        self._drive = drive
        self._overrides = dict(overrides)

    def __getattr__(self, name):
        overrides = self.__dict__["_overrides"]
        if name in overrides:
            return overrides[name]
        return getattr(self.__dict__["_drive"], name)


class _ConfigProxy:
    def __init__(self, base, **drive_overrides):
        self._base = base
        self.drive = _DriveProxy(base.drive, drive_overrides)

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)


def _config(**drive_overrides):
    return _ConfigProxy(Config(), **drive_overrides)


# ── fixtures / builders ───────────────────────────────────────────────────
def _episode(index, concat_path, duration, creation_time):
    return SimpleNamespace(
        index=index,
        concat_path=str(concat_path),
        gpx_path=None,
        date=creation_time.strftime("%Y-%m-%d"),
        total_duration=float(duration),
        creation_time=creation_time,
        gpx_source="gpmf",
    )


def _plan(episodes):
    return SimpleNamespace(episodes=list(episodes))


def _gps(start_ts, coords):
    """``(GPSData-like, gps_offset)`` from a per-second ``(lat, lon)`` list."""
    trackpoints = [
        SimpleNamespace(timestamp=float(start_ts + i), lat=float(lat), lon=float(lon))
        for i, (lat, lon) in enumerate(coords)
    ]
    return (SimpleNamespace(trackpoints=trackpoints), float(start_ts))


def _track(n, *, lat0=32.5, lon0=-80.5, deg_per_sec=0.0002):
    """``n`` per-second fixes marching north. 0.0002 deg/s ~= 22 m/s (moving);
    0.000009 deg/s ~= 1 m/s (a stationary-for-our-purposes crawl)."""
    return [(lat0 + i * deg_per_sec, lon0) for i in range(n)]


def _sign_frame(width=400, height=200):
    """A synthetic frame with one solid green rectangle inside the sign band."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 60), (160, 100), (0, 255, 0), thickness=-1)
    return frame


def _blank_frame(width=400, height=200):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _ocr_ok(_roi):
    return ("Welcome to Colleton County", 0.91)


def _parse_county(text):
    return [SignClaim(kind="county_boundary", value="Colleton", raw_text=text)]


def _parse_nothing(_text):
    return []


class _GrabSpy:
    """Records every (path, t_local) grab and serves a frame per rule."""

    def __init__(self, rule):
        self.calls = []
        self._rule = rule

    def __call__(self, video_path, t_local):
        self.calls.append((str(video_path), float(t_local)))
        return self._rule(str(video_path), float(t_local))

    @property
    def times(self):
        return [t for _p, t in self.calls]


# ── the pass ──────────────────────────────────────────────────────────────
class TestCollectSignHits:
    def test_second_episode_of_day_maps_to_day_master_and_global_index(self):
        """ts_day = first_ep_duration + t_local; episode_index stays GLOBAL."""
        day_a = _episode(0, "/v/ep0.mp4", 30.0, datetime(2026, 5, 23, 9, 0, 0))
        day_b_first = _episode(1, "/v/ep1.mp4", 100.0, datetime(2026, 5, 24, 9, 0, 0))
        day_b_second = _episode(2, "/v/ep2.mp4", 50.0, datetime(2026, 5, 24, 11, 0, 0))
        episode_files = [day_a.concat_path, day_b_first.concat_path, day_b_second.concat_path]

        grab = _GrabSpy(
            lambda path, t: (
                _sign_frame() if (path == day_b_second.concat_path and t == 10.0) else None
            )
        )
        hits = collect_sign_hits(
            _config(),
            _plan([day_a, day_b_first, day_b_second]),
            episode_files,
            [(None, 0.0)] * 3,
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        assert hits["2026-05-23"] == []
        day = hits["2026-05-24"]
        assert len(day) == 1
        hit = day[0]
        assert hit.episode_index == 2, "must be the GLOBAL episode_files index"
        assert hit.t_local == 10.0
        assert hit.ts_day == 110.0, "100s first episode + 10s into the second"
        assert hit.date == "2026-05-24"
        assert hit.raw_text == "Welcome to Colleton County"
        assert hit.ocr_conf == pytest.approx(0.91)
        assert hit.parsed_claims()[0].value == "Colleton"

    def test_per_day_sample_cap_is_never_exceeded_and_spreads(self):
        ep = _episode(0, "/v/long.mp4", 600.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, _t: None)

        collect_sign_hits(
            _config(sign_verify_max_samples_per_day=10),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        times = grab.times
        assert len(times) == 10, "the per-day cap is a hard ceiling"
        assert times == sorted(times)
        assert times[0] < 30.0, "sampling must start near the day's beginning"
        assert times[-1] > 500.0, "sampling must reach the day's end, not cluster at the start"
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert max(gaps) - min(gaps) <= 1.0, "stride must be even across the day"

    def test_stationary_seconds_excluded_when_gps_present(self):
        crawl = _track(30, deg_per_sec=0.000009)  # ~1 m/s — parked/creeping
        rolling = _track(30, lat0=crawl[-1][0], deg_per_sec=0.0002)  # ~22 m/s
        ep = _episode(0, "/v/mixed.mp4", 60.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, _t: None)

        collect_sign_hits(
            _config(sign_verify_min_speed_mps=5.0),
            _plan([ep]),
            [ep.concat_path],
            [_gps(1_000_000.0, crawl + rolling)],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        times = grab.times
        assert times, "the moving half must still be sampled"
        assert not [t for t in times if t <= 20.0], "stationary seconds must be skipped"
        assert [t for t in times if t >= 40.0], "moving seconds must survive"

    def test_gps_less_episode_yields_hits_with_null_coordinates(self):
        ep = _episode(0, "/v/nogps.mp4", 20.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, t: _sign_frame() if t == 5.0 else None)

        hits = collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        day = hits["2026-05-24"]
        assert len(day) == 1
        assert day[0].lat is None and day[0].lon is None
        assert day[0].t_local == 5.0

    def test_hit_carries_nearest_gps_fix_and_skips_null_island(self):
        coords = _track(20)
        coords[5] = (0.0, 0.0)  # cold-start null-island sentinel at the hit second
        ep = _episode(0, "/v/gps.mp4", 20.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, t: _sign_frame() if t == 5.0 else None)

        hits = collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [_gps(1_000_000.0, coords)],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        hit = hits["2026-05-24"][0]
        assert hit.lat is not None and hit.lon is not None
        assert (hit.lat, hit.lon) != (0.0, 0.0), "null-island fixes are never recorded"
        assert hit.lat == pytest.approx(coords[4][0]) or hit.lat == pytest.approx(coords[6][0])

    def test_hits_without_claims_are_discarded(self):
        ep = _episode(0, "/v/nogps.mp4", 20.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, _t: _sign_frame())

        hits = collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_nothing,
        )

        assert hits["2026-05-24"] == [], "OCR noise with no parsed claim is not persisted"

    def test_no_sign_region_means_no_ocr_call(self):
        ep = _episode(0, "/v/nogps.mp4", 10.0, datetime(2026, 5, 24, 9, 0, 0))
        ocr_calls = []

        def _ocr(roi):
            ocr_calls.append(roi)
            return _ocr_ok(roi)

        collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=_GrabSpy(lambda _p, _t: _blank_frame()),
            ocr_fn=_ocr,
            parse_claims_fn=_parse_county,
        )

        assert ocr_calls == [], "a frame with no sign region must never reach OCR"

    def test_grab_and_ocr_failures_are_skipped_not_raised(self):
        ep = _episode(0, "/v/nogps.mp4", 10.0, datetime(2026, 5, 24, 9, 0, 0))

        def _grab(_path, t):
            if t == 3.0:
                raise RuntimeError("ffmpeg exploded")
            return _sign_frame()

        def _ocr(roi):
            _ocr.seen += 1
            if _ocr.seen == 1:
                raise RuntimeError("easyocr exploded")
            return _ocr_ok(roi)

        _ocr.seen = 0

        hits = collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=_grab,
            ocr_fn=_ocr,
            parse_claims_fn=_parse_county,
        )

        day = hits["2026-05-24"]
        assert len(day) == 8, "10 seconds minus one grab failure and one OCR failure"
        assert 3.0 not in [h.t_local for h in day]
        assert day == sorted(day, key=lambda h: h.ts_day)

    def test_broken_episode_does_not_abort_the_day(self):
        good = _episode(0, "/v/good.mp4", 5.0, datetime(2026, 5, 24, 9, 0, 0))
        bad = _episode(1, "/v/missing.mp4", 5.0, datetime(2026, 5, 24, 10, 0, 0))
        # ``bad`` is absent from episode_files → no GLOBAL index → skipped.
        hits = collect_sign_hits(
            _config(),
            _plan([good, bad]),
            [good.concat_path],
            [(None, 0.0)],
            grab_frame_fn=_GrabSpy(lambda _p, t: _sign_frame() if t == 1.0 else None),
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        day = hits["2026-05-24"]
        assert [h.episode_index for h in day] == [0]

    def test_defaults_do_not_require_the_sibling_config_knobs(self):
        """Real ``Config()`` (knobs may not exist yet) must still run."""
        ep = _episode(0, "/v/nogps.mp4", 5.0, datetime(2026, 5, 24, 9, 0, 0))
        hits = collect_sign_hits(
            Config(),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=_GrabSpy(lambda _p, _t: None),
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )
        assert hits == {"2026-05-24": []}

    def test_hits_round_trip_through_state_serialization(self):
        ep = _episode(0, "/v/gps.mp4", 20.0, datetime(2026, 5, 24, 9, 0, 0))
        hits = collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [_gps(1_000_000.0, _track(20))],
            grab_frame_fn=_GrabSpy(lambda _p, t: _sign_frame() if t == 7.0 else None),
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        as_dicts = hits_by_date_to_dicts(hits)
        assert isinstance(as_dicts["2026-05-24"][0], dict)
        rehydrated = hits_by_date_from_state(as_dicts)

        before = hits["2026-05-24"][0]
        after = rehydrated["2026-05-24"][0]
        assert isinstance(after, SignHit)
        assert after == before

    def test_state_shape_is_json_native(self):
        import json

        ep = _episode(0, "/v/nogps.mp4", 10.0, datetime(2026, 5, 24, 9, 0, 0))
        hits = collect_sign_hits(
            _config(),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=_GrabSpy(lambda _p, t: _sign_frame() if t == 2.0 else None),
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )
        json.dumps(hits_by_date_to_dicts(hits))  # must not raise


# ── Phase A wiring ────────────────────────────────────────────────────────
class _StubScorer:
    def __init__(self, *a, **k):
        pass

    def score_file(self, concat_path, **kwargs):
        return np.zeros((5, 9), dtype=np.float64)

    def merge_file_scores(self, file_scores, file_durations):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.merge_file_scores(self, file_scores, file_durations)

    def save_scores(self, scores, path, metadata):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        UnifiedSegmentScorer.save_scores(self, scores, path, metadata)

    def load_scores(self, path):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.load_scores(self, path)


def _patch_phase_a(monkeypatch, episodes):
    """Neutralise every Phase A collaborator except the sign-pass gate."""
    monkeypatch.setattr(phase_a_mod, "UnifiedSegmentScorer", _StubScorer)
    monkeypatch.setattr(phase_a_mod, "create_ingest_plan", lambda *a, **k: object())
    monkeypatch.setattr(
        phase_a_mod,
        "execute_ingest",
        lambda plan, out, **k: SimpleNamespace(episodes=episodes),
    )
    monkeypatch.setattr(phase_a_mod, "_get_shared_geocoder", lambda: None)
    monkeypatch.setattr(phase_a_mod, "_get_route_analyzer", lambda g: None)
    monkeypatch.setattr(
        phase_a_mod,
        "build_route_profile",
        lambda **k: SimpleNamespace(to_dict=lambda: {}, title=lambda: ""),
    )
    monkeypatch.setattr(phase_a_mod, "_extract_episode_ost_meta", lambda p: {})
    monkeypatch.setattr(phase_a_mod, "differentiate_duplicate_ost", lambda locs, meta: locs)
    monkeypatch.setattr(phase_a_mod, "derive_drive_ost_text", lambda *a, **k: "")
    monkeypatch.setattr(phase_a_mod, "load_overrides", lambda base: None)
    monkeypatch.setattr(phase_a_mod, "validate_ost", lambda t, fmt: (True, ""))
    monkeypatch.setattr(phase_a_mod, "classify_video", lambda *a: phase_a_mod.VideoType.REGULAR_LF)


class _CollectSpy:
    def __init__(self, hits):
        self.calls = 0
        self._hits = hits

    def __call__(self, *a, **k):
        self.calls += 1
        return self._hits


@pytest.fixture
def phase_a_env(tmp_path, monkeypatch):
    src = tmp_path / "src"
    out = tmp_path / "out"
    stages = tmp_path / "stages"
    for d in (src, out, stages):
        d.mkdir()
    concat = tmp_path / "ep_0.mp4"
    concat.write_bytes(b"x" * 128)
    episodes = [_episode(0, concat, 30.0, datetime(2026, 5, 24, 9, 0, 0))]
    _patch_phase_a(monkeypatch, episodes)

    def _state():
        return PipelineState(
            source_dir=src,
            output_base=out,
            stages_base=stages,
            region="usa",
            publish_day="tue",
        )

    return SimpleNamespace(state=_state, stages=stages)


class TestPhaseASignPassWiring:
    def _spy(self, monkeypatch):
        spy = _CollectSpy({"2026-05-24": [SignHit(0, 5.0, 5.0, "2026-05-24", "Colleton County")]})
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        return spy

    def test_flag_off_leaves_state_untouched(self, phase_a_env, monkeypatch):
        spy = self._spy(monkeypatch)
        state = phase_a_env.state()
        PhaseA(_config(sign_verify=False)).run(state)
        assert state.sign_hits_by_date is None
        assert spy.calls == 0

    def test_flag_on_populates_state(self, phase_a_env, monkeypatch):
        spy = self._spy(monkeypatch)
        state = phase_a_env.state()
        PhaseA(_config(sign_verify=True)).run(state)
        assert spy.calls == 1
        assert state.sign_hits_by_date == {
            "2026-05-24": [
                {
                    "episode_index": 0,
                    "t_local": 5.0,
                    "ts_day": 5.0,
                    "date": "2026-05-24",
                    "raw_text": "Colleton County",
                    "ocr_conf": 0.0,
                    "lat": None,
                    "lon": None,
                    "claims": [],
                }
            ]
        }

    def test_already_populated_state_is_not_re_decoded(self, phase_a_env, monkeypatch):
        spy = self._spy(monkeypatch)
        state = phase_a_env.state()
        state.sign_hits_by_date = {"2026-05-24": []}
        PhaseA(_config(sign_verify=True)).run(state)
        assert spy.calls == 0, "a resume must never re-decode evidence it already has"
        assert state.sign_hits_by_date == {"2026-05-24": []}

    def test_pass_runs_before_the_cached_score_early_return(self, phase_a_env, monkeypatch):
        """A resume that reuses drive_scores.npz must still back-fill evidence."""
        spy = self._spy(monkeypatch)
        # First run populates the score cache (sign flag OFF).
        PhaseA(_config(sign_verify=False)).run(phase_a_env.state())
        assert (phase_a_env.stages / "drive_scores.npz").exists()
        assert spy.calls == 0

        # Second run hits the cached-score branch — the sign pass must still run.
        state = phase_a_env.state()
        PhaseA(_config(sign_verify=True)).run(state)
        assert spy.calls == 1
        assert state.sign_hits_by_date is not None

    def test_pass_failure_never_fails_the_run(self, phase_a_env, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("sign pass exploded")

        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", _boom)
        state = phase_a_env.state()
        PhaseA(_config(sign_verify=True)).run(state)
        assert state.sign_hits_by_date is None
        assert state.drive_scores is not None, "the run itself must complete"

    def test_phase_a_code_version_not_bumped_by_sign_pass(self):
        """The sign pass is keyed independently of the 3-4h score checkpoint.

        Version 2 is the GPS media-time origin bump, not the sign pass.
        """
        assert phase_a_mod.PHASE_A_CODE_VERSION == 2
