"""YouTube titles learn from YouTube reach.

The nightly ``metrics collect`` joins each shipped title's chosen template id
(``social_posts.json`` for a Short, ``youtube_uploads.json`` for a long-form) to
the YouTube video id it shipped under, asks Waypost's ``title_evidence`` for per
template click-through rates from the collected reach rows, and writes
``title_evidence.json`` beside the queue. Every caller that decides a shipped title
hands that snapshot to ``decide_title``; no file keeps today's baseline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from seo_manager.analytics import metric_source
from seo_manager.analytics.title_selection import EvidenceSnapshot, TemplateObservation
from seo_manager.collection.contracts import Observation
from seo_manager.collection.store import CollectionStore
from seo_manager.collection.youtube_reach import COMBINED_SOURCE

from gomoveshift.pipeline import longform_metadata
from gomoveshift.pipeline.social_generator import SocialGenerator, TitleFacts
from gomoveshift.pipeline.title_decision import approve_retained_title
from gomoveshift.tracking.schema import ShortRecord, TrackingRecord
from gomoveshift.tracking.title_evidence import (
    EVIDENCE_FILE,
    load_title_evidence,
    title_attribution,
    title_evidence_report_lines,
    write_title_evidence,
)

NOW = datetime.now(timezone.utc)
TENANT = "gomoveshift"


def _decision(fmt, chosen_template, reason="insufficient_evidence", others=()):
    candidates = [{"index": 0, "template_id": chosen_template, "title": "t"}]
    candidates += [{"index": i, "template_id": t, "title": None} for i, t in enumerate(others, 1)]
    return {
        "format": fmt,
        "chosen_index": 0,
        "chosen_title": "t",
        "reason": reason,
        "candidates": candidates,
    }


def _project(base, posts=(), uploads=None):
    """A project's output dir: Shorts manifest per LF and the LF upload sidecar."""
    lf = base / "shorts" / "lf_01"
    lf.mkdir(parents=True)
    (lf / "social_posts.json").write_text(json.dumps(list(posts)))
    if uploads is not None:
        (base / "youtube_uploads.json").write_text(json.dumps(uploads))
    return base


def _record(*indexes_and_ids):
    shorts = [
        ShortRecord(index=i, platform_ids={"youtube": vid} if vid else {})
        for i, vid in indexes_and_ids
    ]
    return TrackingRecord(project="proj", shorts=shorts)


# --- attribution ----------------------------------------------------------------


def test_a_short_and_a_longform_are_attributed_to_their_chosen_template(tmp_path):
    base = _project(
        tmp_path,
        posts=[
            {"short_index": 1, "title_decision": _decision("short", "place")},
            {
                "short_index": 2,
                "title_decision": {
                    **_decision("short", "place", others=("highlight_place",)),
                    "chosen_index": 1,
                },
            },
        ],
        uploads={
            "1": {
                "video_id": "LF1",
                "title_decision": _decision("longform", "region_only_lf"),
            }
        },
    )
    got = title_attribution([_record((1, "SV1"), (2, "SV2"))], lambda project: base)
    assert {k: (v.template_id, v.format) for k, v in got.items()} == {
        "SV1": ("place", "short"),
        "SV2": ("highlight_place", "short"),
        "LF1": ("region_only_lf", "longform"),
    }


def test_facts_unavailable_and_template_less_decisions_are_not_evidence(tmp_path):
    base = _project(
        tmp_path,
        posts=[
            {
                "short_index": 1,
                "title_decision": {**_decision("short", "place"), "reason": "facts_unavailable"},
            },
            {
                "short_index": 2,
                "title_decision": {**_decision("short", "place"), "candidates": [{"index": 0}]},
            },
            {"short_index": 3, "title_decision": None},
            {"short_index": 4},
            {"short_index": 5, "title_decision": _decision("short", "place")},
        ],
        uploads={
            "1": {"video_id": "LF1", "title_decision": {"reason": "facts_unavailable"}},
            "2": {"video_id": "LF2"},
            "__plan_fingerprint__": "abc",
        },
    )
    rec = _record((1, "SV1"), (2, "SV2"), (3, "SV3"), (4, "SV4"), (5, None))
    assert title_attribution([rec], lambda project: base) == {}
    rec = _record((1, "SV1"), (5, "SV5"))
    assert set(title_attribution([rec], lambda project: base)) == {"SV5"}


def test_a_project_with_no_output_dir_or_unreadable_files_costs_only_itself(tmp_path):
    assert title_attribution([_record((1, "SV1"))], lambda project: None) == {}
    (tmp_path / "shorts" / "lf_01").mkdir(parents=True)
    (tmp_path / "shorts" / "lf_01" / "social_posts.json").write_text("{not json")
    (tmp_path / "youtube_uploads.json").write_text("{not json")
    assert title_attribution([_record((1, "SV1"))], lambda project: tmp_path) == {}


# --- the evidence file ------------------------------------------------------------


def _snapshot(*rows):
    return EvidenceSnapshot(
        as_of=NOW.isoformat(),
        observations=tuple(
            TemplateObservation(
                template_id=t,
                format=fmt,
                traffic_source=traffic,
                metric_source=metric_source.YOUTUBE_THUMBNAIL_IMPRESSION_CTR,
                rate=rate,
                n=n,
                window_start=(NOW - timedelta(days=10)).isoformat(),
                window_end=(NOW - timedelta(days=1)).isoformat(),
            )
            for t, fmt, traffic, rate, n in rows
        ),
    )


def test_the_evidence_file_round_trips_and_a_bad_file_is_no_evidence(tmp_path):
    from seo_manager.analytics.title_evidence import TitleEvidenceResult

    snap = _snapshot(("place", "short", "browse", 0.05, 450))
    path = write_title_evidence(tmp_path, TitleEvidenceResult(snapshot=snap))
    assert path == tmp_path / EVIDENCE_FILE
    assert load_title_evidence(tmp_path) == snap
    assert [p.name for p in tmp_path.iterdir()] == [EVIDENCE_FILE]  # no temp left behind

    (tmp_path / EVIDENCE_FILE).write_text("{not json")
    assert load_title_evidence(tmp_path) is None
    (tmp_path / EVIDENCE_FILE).write_text(json.dumps({"as_of": "not a time", "observations": []}))
    assert load_title_evidence(tmp_path) is None  # Waypost refuses it
    (tmp_path / EVIDENCE_FILE).unlink()
    assert load_title_evidence(tmp_path) is None


def test_report_lines_name_each_format_traffic_pair_and_the_skip_counters():
    from seo_manager.analytics.title_evidence import TitleEvidenceResult

    result = TitleEvidenceResult(
        snapshot=_snapshot(
            ("place", "short", "browse", 0.05, 450),
            ("highlight_place", "short", "browse", 0.06, 120),
        ),
        rows_used=7,
        rows_unattributed=2,
    )
    lines = title_evidence_report_lines(result)
    assert "TITLE_EVIDENCE format=short traffic=browse templates=2 impressions=570" in lines
    assert "TITLE_EVIDENCE format=longform traffic=search templates=0 impressions=0" in lines
    (skips,) = [x for x in lines if x.startswith("TITLE_EVIDENCE_ROWS")]
    assert "used=7" in skips and "unattributed=2" in skips and "dropped_rate_over_1=0" in skips


# --- the nightly writer -----------------------------------------------------------


def _reach_store(path, video_id, impressions, ctr, traffic="3"):
    store = CollectionStore(path)
    store.init()
    end = NOW - timedelta(days=1)
    start = end - timedelta(days=1)
    subject = f"{video_id}|{traffic}|-|-|-"
    rows = [
        Observation(TENANT, COMBINED_SOURCE, subject, metric, start, end, value, NOW)
        for metric, value in (
            ("video_thumbnail_impressions", impressions),
            ("video_thumbnail_impressions_ctr", ctr),
        )
    ]
    store.write_batch(TENANT, rows)
    return store


def _config(storage):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=storage))


@pytest.fixture
def nightly(tmp_path, monkeypatch):
    """A project with one decided Short, a sidecar the queue lookup resolves, and
    a tracking record that names the Short's video."""
    from test_cli_enqueue_plan import _record as plan_record
    from test_cli_enqueue_plan import _sidecar

    from gomoveshift.cli.handlers import metrics

    _project(
        tmp_path,
        posts=[{"short_index": 1, "title_decision": _decision("short", "place")}],
    )
    _sidecar(tmp_path, plan_record(short_index=1))
    rec = _record((1, "SV1"))
    monkeypatch.setattr(
        metrics, "TrackingStore", lambda: SimpleNamespace(load_all_records=lambda: [rec])
    )
    return metrics, tmp_path


def test_nightly_collect_writes_the_evidence_file_from_the_reach_store(
    nightly, monkeypatch, capsys
):
    metrics, tmp_path = nightly
    db = tmp_path / "reach" / "collection.sqlite3"
    db.parent.mkdir()
    _reach_store(db, "SV1", impressions=1000, ctr=0.05)
    monkeypatch.setenv("SEO_COLLECTION_DB", str(db))

    metrics._write_title_evidence(_config(tmp_path))

    (only,) = load_title_evidence(tmp_path).observations
    assert (only.template_id, only.format, only.traffic_source) == ("place", "short", "browse")
    assert (only.n, round(only.rate, 3)) == (1000, 0.05)
    out = capsys.readouterr().out
    assert "TITLE_EVIDENCE format=short traffic=browse templates=1 impressions=1000" in out


def test_a_missing_reach_store_is_a_logged_skip_and_nothing_is_created(
    nightly, monkeypatch, capsys, caplog
):
    metrics, tmp_path = nightly
    db = tmp_path / "not-yet" / "collection.sqlite3"
    monkeypatch.setenv("SEO_COLLECTION_DB", str(db))

    with caplog.at_level("INFO"):
        metrics._write_title_evidence(_config(tmp_path))

    assert not db.parent.exists() and not (tmp_path / EVIDENCE_FILE).exists()
    assert "no reach store" in capsys.readouterr().out
    assert "no reach store" in caplog.text


def test_a_failed_evidence_write_never_fails_the_collection(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import metrics

    db = tmp_path / "collection.sqlite3"
    CollectionStore(db).init()
    monkeypatch.setenv("SEO_COLLECTION_DB", str(db))

    def boom():
        raise OSError("store unreadable")

    monkeypatch.setattr(metrics, "TrackingStore", boom)
    metrics._write_title_evidence(_config(tmp_path))
    assert load_title_evidence(tmp_path) is None


# --- shipped titles read the snapshot ---------------------------------------------


@pytest.fixture
def storage(tmp_path, monkeypatch):
    seo = tmp_path / "seo"
    seo.mkdir()
    monkeypatch.setattr(longform_metadata, "DEFAULT_SEO_STORAGE", seo)
    return seo


def _write(storage, *rows):
    from seo_manager.analytics.title_evidence import TitleEvidenceResult

    write_title_evidence(storage, TitleEvidenceResult(snapshot=_snapshot(*rows)))


def _short_decision():
    gen = SocialGenerator(location="Washington County, PA", project_name="u5c", series_part=1)
    variant = gen._per_short_variant("Washington County, PA", ["Washington County, PA"])
    facts = TitleFacts(date="2026-03-04", ordinal=1, secs=42, place_ok=True)
    return variant.generate_post(short_index=1, title_facts=facts).title_decision


def test_a_generated_short_carries_the_loaded_snapshot(storage):
    _write(storage, ("place", "short", "browse", 0.05, 450))
    decision = _short_decision()
    baseline = decision["candidates"][0]
    assert decision["evidence_as_of"] == load_title_evidence(storage).as_of
    assert (baseline["n"], baseline["score"]) == (450, 0.05)
    # A single-place Short has no second renderable template, so it cannot move.
    assert decision["reason"] == "insufficient_evidence" and decision["chosen_index"] == 0


def test_no_evidence_file_keeps_the_baseline_as_insufficient_evidence(storage):
    decision = _short_decision()
    assert decision["reason"] == "insufficient_evidence"
    assert decision["candidates"][0]["n"] is None


def _region_route():
    from test_region_only_waypost_title import _route

    return _route()


def test_a_longform_title_moves_once_baseline_and_alternative_each_have_400_impressions(storage):
    _write(
        storage,
        ("region_only_lf", "longform", "search", 0.04, 500),
        ("region_drive_lf", "longform", "search", 0.06, 500),
    )
    decision = approve_retained_title("Ashby County, SC", _region_route(), None)
    assert decision["reason"] == "lift_over_baseline"
    assert decision["chosen_title"] == "Ashby County, SC Drive in 4K"


def test_a_longform_with_no_evidence_file_ships_the_baseline(storage):
    decision = approve_retained_title("Ashby County, SC", _region_route(), None)
    assert decision["reason"] == "insufficient_evidence"
    assert decision["chosen_title"] == "Ashby County, SC in 4K"
