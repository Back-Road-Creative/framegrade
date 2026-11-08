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
        """On a day with a trace the flat cap is the ceiling (a GPS-less day is
        sampled by period instead — see the gpsless test below)."""
        ep = _episode(0, "/v/long.mp4", 600.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, _t: None)

        collect_sign_hits(
            _config(sign_verify_max_samples_per_day=10),
            _plan([ep]),
            [ep.concat_path],
            [_gps(1_000_000.0, _track(600))],
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

    def test_a_gpsless_day_is_sampled_every_period_not_at_the_flat_cap(self):
        """No GPS means the signs are the drive's only evidence, so a GPS-less day
        is read every ``sign_verify_gpsless_sample_period_s`` seconds; the flat
        per-day cap only governs days that have a trace to lean on."""
        ep = _episode(0, "/v/nogps-long.mp4", 600.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, _t: None)

        collect_sign_hits(
            _config(sign_verify_max_samples_per_day=10, sign_verify_gpsless_sample_period_s=10.0),
            _plan([ep]),
            [ep.concat_path],
            [(None, 0.0)],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        times = grab.times
        assert len(times) == 60, "600 s / 10 s period = 60 samples, not the flat cap of 10"
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert max(gaps) - min(gaps) <= 1.0, "stride must stay even across the day"

    def test_a_gps_day_keeps_the_flat_cap(self):
        ep = _episode(0, "/v/gps-long.mp4", 600.0, datetime(2026, 5, 24, 9, 0, 0))
        grab = _GrabSpy(lambda _p, _t: None)

        collect_sign_hits(
            _config(sign_verify_max_samples_per_day=10, sign_verify_gpsless_sample_period_s=10.0),
            _plan([ep]),
            [ep.concat_path],
            [_gps(1_000_000.0, _track(600))],
            grab_frame_fn=grab,
            ocr_fn=_ocr_ok,
            parse_claims_fn=_parse_county,
        )

        assert len(grab.times) == 10, "a day with a trace is still bounded by the flat cap"

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
class TestSignPassWanted:
    """When the pass runs: an operator decision wins; undecided, a drive with a
    GPS-less episode reads its signs because nothing else can say where it was."""

    @pytest.mark.parametrize(
        "config_flag, state_flag, gps, wanted",
        [
            (False, None, [(None, 0.0)], True),  # undecided, no GPS: signs are the only evidence
            (
                False,
                None,
                [(None, 0.0), _gps(1.0, _track(3))],
                True,
            ),  # one GPS-less episode is enough
            (False, None, [_gps(1.0, _track(3))], False),  # undecided, GPS everywhere: stays opt-in
            (False, None, [], False),  # nothing to read
            (False, False, [(None, 0.0)], False),  # --no-sign-verify is respected even without GPS
            (True, None, [_gps(1.0, _track(3))], True),  # --sign-verify (config mirror) wins
            (False, True, [_gps(1.0, _track(3))], True),  # a saved --sign-verify wins on resume
        ],
    )
    def test_decision(self, config_flag, state_flag, gps, wanted):
        assert sign_pass_mod.sign_pass_wanted(config_flag, state_flag, gps) is wanted


class _StubScorer:
    def __init__(self, *a, **k):
        pass

    def score_file(self, concat_path, **kwargs):
        return np.zeros((5, 9), dtype=np.float64)

    def merge_file_scores(self, file_scores, file_durations):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.merge_file_scores(self, file_scores, file_durations)

    def save_scores(self, scores, path, metadata, extra=None):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        UnifiedSegmentScorer.save_scores(self, scores, path, metadata, extra)

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

    def test_explicit_no_sign_verify_leaves_state_untouched(self, phase_a_env, monkeypatch):
        spy = self._spy(monkeypatch)
        state = phase_a_env.state()
        state.sign_verify = False  # --no-sign-verify, saved by the flag registry
        PhaseA(_config(sign_verify=False)).run(state)
        assert state.sign_hits_by_date is None
        assert spy.calls == 0

    def test_a_gpsless_drive_runs_the_pass_without_the_flag(self, phase_a_env, monkeypatch):
        """The fixture drive has no GPX, so its signs are read by default."""
        spy = self._spy(monkeypatch)
        state = phase_a_env.state()
        assert state.sign_verify is None, "the operator did not decide"
        PhaseA(_config(sign_verify=False)).run(state)
        assert spy.calls == 1
        assert state.sign_hits_by_date is not None

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
        # First run populates the score cache (sign pass explicitly OFF).
        first = phase_a_env.state()
        first.sign_verify = False
        PhaseA(_config(sign_verify=False)).run(first)
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


# ── a resume with Phase A checkpointed still reads the signs ─────────────
# ``PhaseA.run`` is never re-entered once ``phase_a`` sits in
# ``phases_completed`` (the orchestrator's resumed-phase branch runs only the
# OST gate), so a GPS-less drive parked past Phase A before the pass became
# the default would never have its signs read. ``sign_pass.run_sign_pass`` is
# the one A6 body both callers share; the orchestrator back-fills with it.


def _resumed_state(**over):
    base = dict(
        sign_hits_by_date=None,
        sign_verify=None,
        gps_per_episode=[(None, 0.0)],
        ingest_plan=SimpleNamespace(episodes=[]),
        episode_files=["/mock/ep0.mp4"],
        episode_dates=["2026-03-05"],
        file_durations=[100.0],
        gpx_files=[None],
        episode_locations=[""],
        phases_completed=["phase_a"],
    )
    base.update(over)
    return SimpleNamespace(**base)


_ONE_HIT = {"2026-03-05": [SignHit(0, 5.0, 5.0, "2026-03-05", "Colleton County")]}


class TestRunSignPass:
    def _call(self, state, config=None):
        return sign_pass_mod.run_sign_pass(
            config or _config(sign_verify=False),
            state,
            state.ingest_plan,
            state.episode_files,
            state.gps_per_episode,
        )

    def test_a_gpsless_drive_is_read_and_its_hits_recorded(self, monkeypatch):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state()
        assert self._call(state) is True
        assert spy.calls == 1
        assert state.sign_hits_by_date == {
            "2026-03-05": [SignHit(0, 5.0, 5.0, "2026-03-05", "Colleton County").to_dict()]
        }

    def test_footage_already_read_is_not_decoded_again(self, monkeypatch):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(sign_hits_by_date={"2026-03-05": []})
        assert self._call(state) is False
        assert spy.calls == 0
        assert state.sign_hits_by_date == {"2026-03-05": []}

    def test_the_operator_s_no_sign_verify_wins(self, monkeypatch):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(sign_verify=False)
        assert self._call(state) is False
        assert spy.calls == 0 and state.sign_hits_by_date is None

    def test_a_located_drive_without_the_flag_is_left_alone(self, monkeypatch):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(gps_per_episode=[(object(), 0.0)])
        assert self._call(state) is False
        assert spy.calls == 0

    def test_a_failing_pass_is_advisory(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("sign pass exploded")

        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", _boom)
        state = _resumed_state()
        assert self._call(state) is False
        assert state.sign_hits_by_date is None


class TestZeroDaysIsNotEvidence:
    """``run_sign_pass`` never records ``{}``, and never trusts one it finds.

    A pass that grouped zero days decoded nothing; a saved ``{}`` (what the
    first rec0063 back-fill left behind) must read as unread so the next
    resume retries instead of trusting empty evidence.
    """

    def _call(self, state, config=None):
        return sign_pass_mod.run_sign_pass(
            config or _config(sign_verify=False),
            state,
            state.ingest_plan,
            state.episode_files,
            state.gps_per_episode,
        )

    def test_a_pass_that_grouped_zero_days_leaves_the_field_unset(self, monkeypatch):
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", _CollectSpy({}))
        state = _resumed_state()
        assert self._call(state) is False
        assert state.sign_hits_by_date is None, "{} would read as evidence on the next resume"

    def test_a_checkpoint_holding_an_empty_dict_is_read_again(self, monkeypatch):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(sign_hits_by_date={})
        assert self._call(state) is True
        assert spy.calls == 1
        assert state.sign_hits_by_date == {
            "2026-03-05": [h.to_dict() for h in _ONE_HIT["2026-03-05"]]
        }

    def test_a_read_but_empty_day_still_counts_as_read(self, monkeypatch):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(sign_hits_by_date={"2026-03-05": []})
        assert self._call(state) is False
        assert spy.calls == 0


class TestBackfillRehydratesTheResumedPlan:
    """``backfill_sign_pass_on_resume`` with the plan a JSON round-trip yields.

    ``IngestPlan.from_dict`` is a shallow rehydrate: ``episodes`` come back as
    the raw ``list[dict]`` they were serialized as. ``group_episodes_by_day``
    reads them with ``getattr``, every episode is silently dropped, and the
    pass reports "0 hit(s) across 0 day(s)" in no time at all — the exact
    failure nctopa-2026-03-05-rec0063 showed on its first back-fill. The
    back-fill must hand the pass a plan it can group.
    """

    def _state(self, tmp_path):
        from gomoveshift.pipeline.ingest.gopro_parser import IngestPlan

        ep_path = str(tmp_path / "episode_01_2026-03-05.mp4")
        plan = IngestPlan.from_dict(
            {
                "source_dir": str(tmp_path),
                "episodes": [
                    {
                        "index": 1,
                        "recordings": [],
                        "location": "",
                        "date": "2026-03-05",
                        "concat_path": ep_path,
                        "gpx_path": None,
                        "gpx_source": "gpmf",
                    }
                ],
            }
        )
        assert isinstance(plan.episodes[0], dict), "the shape under test is the shallow one"
        return _resumed_state(
            ingest_plan=plan,
            episode_files=[ep_path],
            episode_dates=["2026-03-05"],
            file_durations=[13208.3],
            gpx_files=[None],
        )

    def test_the_pass_is_handed_a_plan_it_can_group_by_day(self, monkeypatch, tmp_path):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day
        from gomoveshift.pipeline.drive_orchestrator import backfill_sign_pass_on_resume

        handed: list = []

        def spy(config, ingest_plan, episode_files, gps_per_episode, **k):
            handed.append(ingest_plan)
            return {"2026-03-05": []}

        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = self._state(tmp_path)
        assert backfill_sign_pass_on_resume(_config(sign_verify=False), state) is True
        batches = group_episodes_by_day(handed[0])
        assert [b.date for b in batches] == ["2026-03-05"], (
            "dict episodes group into 0 days; the back-fill must rehydrate them"
        )
        (ep,) = batches[0].episodes
        assert str(ep.concat_path) == state.episode_files[0]
        assert ep.total_duration == pytest.approx(13208.3)

    def test_the_real_pass_sees_the_footage_not_zero_days(self, monkeypatch, tmp_path):
        """End to end through ``collect_sign_hits``: the one sampled second is OCR'd."""
        from gomoveshift.pipeline.drive_orchestrator import backfill_sign_pass_on_resume

        sampled: list = []
        monkeypatch.setattr(
            sign_pass_mod,
            "_hit_for_candidate",
            lambda cand, date, **k: sampled.append((cand.episode_index, cand.t_local)) or None,
        )
        state = self._state(tmp_path)
        assert backfill_sign_pass_on_resume(_config(sign_verify=False), state) is True
        assert state.sign_hits_by_date == {"2026-03-05": []}
        assert sampled, (
            "a 13208 s GPS-less episode yields zero sampled seconds only if it was dropped"
        )
        assert {idx for idx, _ in sampled} == {0}


class TestResumedPhaseAReadsSigns:
    """``DrivePipelineOrchestrator._execute_phases`` with ``phase_a`` resumed."""

    def _run(self, monkeypatch, tmp_path, state, *, label_dry_run=False):
        import time

        from gomoveshift.pipeline import drive_orchestrator as orch_mod
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_phases import lf_listing

        monkeypatch.setattr(orch_mod, "run_phase_a_ost_gate", lambda s: None)
        seen_at_dry_run: list = []
        monkeypatch.setattr(
            lf_listing,
            "print_label_dry_run",
            lambda cfg, st: seen_at_dry_run.append(st.sign_hits_by_date) and False,
        )
        orch = DrivePipelineOrchestrator(_config(sign_verify=False))
        saves: list = []
        monkeypatch.setattr(orch, "_save_state", lambda f, s: saves.append(f))
        monkeypatch.setattr(orch, "_emit_pipeline_summary", lambda **k: None)
        state_file = tmp_path / "pipeline_state.json"
        orch._execute_phases(
            state=state,
            state_file=state_file,
            phase_defs=[
                ("phase_a", lambda s: pytest.fail("a checkpointed phase_a must not re-run"))
            ],
            resumed_phases=["phase_a"],
            source_dir=tmp_path,
            project_dir=tmp_path,
            location="",
            list_lfs=False,
            label_dry_run=label_dry_run,
            start_time=time.time(),
            stages_completed=["phase_a"],
            errors=[],
            timings={},
        )
        return saves, state_file, seen_at_dry_run

    def test_a_checkpointed_gpsless_drive_has_its_signs_read_and_saved(self, monkeypatch, tmp_path):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state()
        saves, state_file, _ = self._run(monkeypatch, tmp_path, state)
        assert spy.calls == 1
        assert state.sign_hits_by_date is not None
        assert saves == [state_file], (
            "the evidence must be checkpointed, or the next resume re-decodes"
        )

    def test_a_checkpoint_that_already_holds_evidence_is_not_decoded_or_saved_again(
        self, monkeypatch, tmp_path
    ):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(sign_hits_by_date={"2026-03-05": []})
        saves, _, _ = self._run(monkeypatch, tmp_path, state)
        assert spy.calls == 0
        assert saves == []

    def test_the_label_dry_run_sees_the_back_filled_evidence(self, monkeypatch, tmp_path):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state()
        _, _, seen = self._run(monkeypatch, tmp_path, state, label_dry_run=True)
        assert spy.calls == 1
        assert seen and seen[0] is not None, "the dry-run printed before the signs were read"

    def test_a_checkpoint_with_no_ingest_plan_is_left_alone(self, monkeypatch, tmp_path):
        spy = _CollectSpy(_ONE_HIT)
        monkeypatch.setattr(sign_pass_mod, "collect_sign_hits", spy)
        state = _resumed_state(ingest_plan=None)
        saves, _, _ = self._run(monkeypatch, tmp_path, state)
        assert spy.calls == 0 and saves == []
