"""The sign pass says how far it is and where its samples died.

``nctopa-2026-03-05-rec0063`` (220 min, GPS-less, 4K HEVC on a slow mount,
2026-11-08): the A6 pass sampled 1320 seconds at ~7 s each and wrote NOTHING
to the log for two hours, then one line — "recorded 0 hit(s)". An operator
could not tell a hang from progress, and "0 hits" could not be told apart
from "0 frames decoded" or "0 sign regions found" without a separate probe.
Now the pass logs progress every ``PROGRESS_EVERY`` samples (done/total, hits,
elapsed, ETA) and counts each stage — sampled, grabbed, region, ocr_text,
claims — per day; the counters ride on the A6 line and persist on the state
as ``sign_pass_stats`` beside the evidence they explain.

Hermetic like ``test_sign_pass``: grab / OCR / claim parsing are injected; the
sign-region finder runs for real on small synthetic frames.
"""

import logging
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np

from gomoveshift.analysis import sign_pass as sign_pass_mod
from gomoveshift.analysis.sign_evidence import SignClaim
from gomoveshift.analysis.sign_pass import collect_sign_hits
from gomoveshift.config import Config
from gomoveshift.pipeline.drive_state_model import PipelineState

DATE = "2026-03-05"


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


def _episode(index, concat_path, duration):
    return SimpleNamespace(
        index=index,
        concat_path=str(concat_path),
        gpx_path=None,
        date=DATE,
        total_duration=float(duration),
        creation_time=datetime(2026, 3, 5, 9, 0, 0),
        gpx_source="gpmf",
    )


def _sign_frame(width=400, height=200):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 60), (160, 100), (0, 255, 0), thickness=-1)
    return frame


def _blank_frame(width=400, height=200):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _parse_county_only(text):
    if "Colleton" in text:
        return [SignClaim(kind="county_boundary", value="Colleton", raw_text=text)]
    return []


class _Ocr:
    """First region reads a county sign, every later one an exit sign."""

    def __init__(self):
        self.calls = 0

    def __call__(self, _roi):
        self.calls += 1
        return ("Welcome to Colleton County", 0.9) if self.calls == 1 else ("EXIT 283", 0.8)


def _grab(_path, t):
    # 60 s GPS-less episode, cap 6 -> ``_even_stride`` samples 0,10,...,50.
    # Three decode (0, 10, 20); of those, 0 and 10 carry a sign region, 20 is blank.
    if t in (0.0, 10.0):
        return _sign_frame()
    if t == 20.0:
        return _blank_frame()
    return None


EXPECTED = {"candidates": 60, "sampled": 6, "grabbed": 3, "region": 2, "ocr_text": 2, "claims": 1}


def _run(stats=None, **cfg):
    ep = _episode(0, "/v/nogps.mp4", 60.0)
    return collect_sign_hits(
        _config(sign_verify_max_samples_per_day=6, **cfg),
        SimpleNamespace(episodes=[ep]),
        [ep.concat_path],
        [(None, 0.0)],
        grab_frame_fn=_grab,
        ocr_fn=_Ocr(),
        parse_claims_fn=_parse_county_only,
        stats=stats,
    )


def test_stage_counters_say_where_each_sample_died():
    stats = {}
    hits = _run(stats=stats)

    assert len(hits[DATE]) == 1
    day = stats[DATE]
    assert {k: day[k] for k in EXPECTED} == EXPECTED
    assert day["elapsed_s"] >= 0.0


def test_the_day_summary_line_carries_the_counters(caplog):
    with caplog.at_level(logging.INFO, logger=sign_pass_mod.__name__):
        _run()

    summary = [
        r.getMessage() for r in caplog.records if "1 hit(s) from 6 sampled" in r.getMessage()
    ]
    assert summary, caplog.text
    assert "grabbed=3 region=2 ocr_text=2 claims=1" in summary[0]


def test_progress_is_logged_every_n_samples_with_an_eta(monkeypatch, caplog):
    monkeypatch.setattr(sign_pass_mod, "PROGRESS_EVERY", 2)
    with caplog.at_level(logging.INFO, logger=sign_pass_mod.__name__):
        _run()

    progress = [r.getMessage() for r in caplog.records if "progress" in r.getMessage()]
    assert [p.split("progress ")[1].split(" ")[0] for p in progress] == ["2/6", "4/6", "6/6"]
    assert all("elapsed" in p and "ETA" in p and DATE in p for p in progress)


def test_progress_stays_quiet_below_the_default_period(caplog):
    assert sign_pass_mod.PROGRESS_EVERY == 50
    with caplog.at_level(logging.INFO, logger=sign_pass_mod.__name__):
        _run()

    assert not [r for r in caplog.records if "progress" in r.getMessage()]


def test_run_sign_pass_persists_the_counters_and_prints_them_on_the_a6_line(
    tmp_path, monkeypatch, caplog
):
    ep = _episode(0, tmp_path / "ep0.mp4", 60.0)
    state = PipelineState(
        source_dir=tmp_path / "src",
        output_base=tmp_path / "out",
        stages_base=tmp_path / "out" / "_stages",
        region="usa",
        publish_day="tue",
    )
    state.ingest_plan = SimpleNamespace(episodes=[ep])
    state.episode_files = [ep.concat_path]
    state.gps_per_episode = [(None, 0.0)]
    monkeypatch.setattr(sign_pass_mod, "_default_grab_frame", _grab)
    monkeypatch.setattr(sign_pass_mod, "_LazyEasyOCR", lambda config: _Ocr())
    monkeypatch.setattr(
        sign_pass_mod, "_LazyClaimParser", lambda fn: lambda text: _parse_county_only(text)
    )

    with caplog.at_level(logging.INFO, logger=sign_pass_mod.__name__):
        assert sign_pass_mod.run_sign_pass(
            _config(sign_verify=False, sign_verify_max_samples_per_day=6),
            state,
            state.ingest_plan,
            state.episode_files,
            state.gps_per_episode,
        )

    assert {k: state.sign_pass_stats[DATE][k] for k in EXPECTED} == EXPECTED
    a6 = [r.getMessage() for r in caplog.records if "A6: sign pass recorded" in r.getMessage()]
    assert a6, caplog.text
    assert f"{DATE}: sampled=6 grabbed=3 region=2 ocr_text=2 claims=1" in a6[0]
