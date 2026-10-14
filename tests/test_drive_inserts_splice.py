"""Inserts in the drive run: the day pipeline splices them after the OST burn, the
burn fingerprint covers them, a resume plans against every day, and Phase D shows them.
ffmpeg, the geocoder and the track are injected."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import inserts as ins_mod
from gomoveshift.pipeline.inserts import (
    Insert,
    InsertError,
    MediaInfo,
    insert_request,
    reconcile_insert_request,
    shift_photo_inserts,
    splice_day_lf_inserts,
)

DATE = "2026-10-03"


def _config():
    from gomoveshift.config import Config

    return Config()


def _drive_cfg(tmp_path, clip):
    config = _config()
    config.drive.insert_clips = [f"{clip}@00:30"]
    return config


# ── shift_photo_inserts / splice_day_lf_inserts ──────────────────────────────────────


def test_shift_photo_inserts_moves_stills_only_and_clamps_at_zero():
    still = Insert(10.0, Path("a.JPG"))
    clip = Insert(10.0, Path("c.mp4"))
    out = shift_photo_inserts([still, clip], 2.5)
    assert [i.at_s for i in out] == [12.5, 10.0]
    assert shift_photo_inserts([still, clip], -30.0)[0].at_s == 0.0
    assert shift_photo_inserts([still, clip], -30.0)[1].at_s == 10.0
    assert still.at_s == 10.0  # the inputs are not touched


def test_splice_day_lf_inserts_shifts_stills_by_the_montage_lead_error(tmp_path):
    durations = {"prep.mp4": 130.0, "cat.mp4": 100.0}  # the montage really added 30 s

    def probe(p):
        return MediaInfo(1920, 1080, 30.0, durations[Path(p).name])

    seen = {}

    def fake_splice(lf, inserts, **kw):
        seen.update(lf=lf, inserts=list(inserts), kw=kw)
        return "result"

    with patch.object(ins_mod, "splice_lf_inserts", side_effect=fake_splice):
        out = splice_day_lf_inserts(
            tmp_path / "lf_01.mp4",
            [Insert(50.0, Path("a.jpg")), Insert(50.0, Path("c.mp4"))],
            config="cfg",
            label="Colleton County, SC",
            profile="prof",
            work_dir=tmp_path / "w",
            montage_planned_s=25.0,  # planned 25 s, got 30 s: stills move +5
            prepended=tmp_path / "prep.mp4",
            concatenated=tmp_path / "cat.mp4",
            probe=probe,
        )
    assert out == "result"
    assert [i.at_s for i in seen["inserts"]] == [55.0, 50.0]
    assert seen["kw"]["label"] == "Colleton County, SC"
    assert seen["kw"]["config"] == "cfg"


# ── run_day_pipeline: fingerprint fold, splice ordering, gate hash ───────────────────


def _run_day(tmp_path, inserts_by_lf, *, location="NC, USA", splice_effect=None, state=None):
    """One run_day_pipeline pass over a seeded master; returns (artifacts, calls, mocks)."""
    from test_lf_freshness_resume_reburn import (
        _make_day_batch,
        _make_lf_plan,
        _make_scores,
        _make_seg,
        _seed_master,
    )

    from _state_helpers import make_state
    from gomoveshift.pipeline import day_pipeline

    output_base = tmp_path / "out"
    master = _seed_master(output_base)
    calls: list[str] = []

    def asm(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"concat")
        return output_path

    def prep(raw_lf_path, peaks, output_path, work_dir, **kw):
        output_path.write_bytes(b"montage")
        return output_path

    def ost(input_path, output_path, location, layout, **kw):
        calls.append("apply_ost")
        output_path.write_bytes(b"ost")
        return output_path

    def render(source_video, output_path, start_time, duration, location, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"short")
        return True

    def splice(lf, inserts, **kw):
        calls.append("splice")
        if splice_effect:
            raise splice_effect
        return None

    with (
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master),
        patch("gomoveshift.pipeline.day_pipeline.extract_and_concat_segments", side_effect=asm),
        patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=prep),
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=ost),
        patch("gomoveshift.pipeline.day_pipeline._render_short_ffmpeg", side_effect=render),
        patch("gomoveshift.pipeline.day_pipeline.splice_day_lf_inserts", side_effect=splice) as spl,
    ):
        art = day_pipeline.run_day_pipeline(
            day_batch=_make_day_batch("2026-02-20", 2),
            day_scores=_make_scores(2400),
            output_dir=output_base,
            location=location,
            config=None,
            lf_plans=[_make_lf_plan(1, [_make_seg(0, 100, 200)], [])],
            state=state or make_state(ost_audit_confirmed=True),
            inserts_by_lf=inserts_by_lf,
        )
    return art, calls, spl, output_base


def test_post_burn_splice_runs_after_apply_ost_on_the_burned_result(tmp_path):
    ins = {1: [Insert(30.0, Path("a.jpg"))]}
    art, calls, spl, out = _run_day(tmp_path, ins)
    assert calls == ["apply_ost", "splice"]
    args, kw = spl.call_args
    assert args[0] == out / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4"
    assert kw["prepended"].name == "lf_01_with_montage.mp4"
    assert kw["concatenated"].name == "lf_01_segments.mp4"
    assert kw["montage_planned_s"] == 15.0  # config=None: the 15 s default montage lead
    assert art.inserts == ins_mod.insert_records(ins)


def test_no_inserts_means_no_splice(tmp_path):
    _, calls, spl, _ = _run_day(tmp_path, None)
    assert calls == ["apply_ost"]
    spl.assert_not_called()


def test_a_failed_splice_leaves_no_fingerprint_so_the_resume_reburns(tmp_path):
    ins = {1: [Insert(30.0, Path("a.jpg"))]}
    with pytest.raises(InsertError):
        _run_day(tmp_path, ins, splice_effect=InsertError("refused"))
    lf_dir = tmp_path / "out" / "day_2026-02-20" / "regular_lfs"
    assert not (lf_dir / "lf_01.ost.sha256").exists()


def test_the_burn_fingerprint_folds_the_inserts_in(tmp_path):
    a = {1: [Insert(30.0, Path("a.jpg"))]}
    same = {1: [Insert(30.0, Path("a.jpg"))]}
    moved = {1: [Insert(31.0, Path("a.jpg"))]}
    _, calls, _, _ = _run_day(tmp_path, a)
    assert calls == ["apply_ost", "splice"]
    _, calls, _, _ = _run_day(tmp_path, same)  # unchanged: fresh final, nothing re-burned
    assert calls == []
    _, calls, _, _ = _run_day(tmp_path, moved)  # a moved insert: re-burn and re-splice
    assert calls == ["apply_ost", "splice"]
    _, calls, _, _ = _run_day(tmp_path, None)  # inserts dropped: the final is stale too
    assert calls == ["apply_ost"]


def test_the_per_day_gate_hash_covers_the_inserts(tmp_path):
    """A confirmation hashed without the inserts is stale once they exist; one hashed with passes."""
    from _state_helpers import make_state
    from gomoveshift.pipeline.ost_audit_sidecar import (
        OstAuditPendingError,
        compute_ost_audit_hash,
        format_ost_audit,
    )

    day = "2026-02-20"
    ins = {1: [Insert(30.0, Path("a.jpg"))]}
    audit_state = {
        "per_lf_ost_text": {},
        "per_lf_titles": {},
        "per_lf_recording_date": {},
        "per_lf_locations": {},
    }

    def hash_of(inserts):
        payload = format_ost_audit(day, audit_state, [1], location="NC, USA", inserts=inserts)
        return compute_ost_audit_hash(payload)

    assert hash_of(None) != hash_of(ins_mod.insert_records(ins))
    stale = make_state(ost_audit_confirmed=True, ost_audit_hash={day: hash_of(None)})
    with pytest.raises(OstAuditPendingError):
        _run_day(tmp_path, ins, state=stale)
    fresh = make_state(
        ost_audit_confirmed=True, ost_audit_hash={day: hash_of(ins_mod.insert_records(ins))}
    )
    art, _, _, _ = _run_day(tmp_path / "ok", ins, state=fresh)
    assert art is not None


# ── reconcile_insert_request ─────────────────────────────────────────────────────────


def _drive(photos="", clips=()):
    return SimpleNamespace(insert_photos=photos, insert_clips=list(clips))


def test_insert_request_is_none_without_flags_and_canonical_with_them():
    assert insert_request(_drive()) is None
    assert insert_request(_drive("/p", ["b@0:10", "a@0:20"])) == {
        "photos": "/p",
        "clips": ["a@0:20", "b@0:10"],
    }


@pytest.mark.parametrize(
    "saved,current,done,refuses",
    [
        (None, None, ["d1"], False),
        ({"photos": "/p", "clips": []}, _drive("/p"), ["d1"], False),
        (None, _drive("/p"), ["d1"], True),  # a finished day was rendered without them
        ({"photos": "/p", "clips": []}, _drive(), ["d1"], True),  # flags dropped on resume
        ({"photos": "/p", "clips": []}, _drive("/q"), ["d1"], True),
        (None, _drive("/p"), [], False),  # nothing finished: free to change
        ({"photos": "/p", "clips": []}, _drive(), [], False),
    ],
)
def test_reconcile_insert_request_matrix(saved, current, done, refuses):
    state = SimpleNamespace(insert_request=saved)
    if refuses:
        with pytest.raises(InsertError, match="--fresh"):
            reconcile_insert_request(state, current, done)
        return
    reconcile_insert_request(state, current, done)
    assert state.insert_request == insert_request(current)


# ── run_all_days: plan against every day, render some ────────────────────────────────


def _run_all(tmp_path, spy, planner, config, state=None, **extra):
    import numpy as np

    from gomoveshift.pipeline.day_grouper import DayBatch
    from gomoveshift.pipeline.day_pipeline import run_all_days

    ep = SimpleNamespace(
        index=1,
        concat_path=Path("/fake/ep.mp4"),
        total_duration=600.0,
        date=DATE,
        creation_time=None,
    )
    scores = np.zeros((600, 9))
    scores[:, 0] = np.arange(600)
    out = tmp_path / "stages"
    out.mkdir(exist_ok=True)
    plans = {DATE: [SimpleNamespace(index=1, segments=[], montage_peaks=[])]}
    with (
        patch("gomoveshift.pipeline.day_pipeline.run_day_pipeline", side_effect=spy),
        patch("gomoveshift.pipeline.day_pipeline.plan_drive_inserts", side_effect=planner),
    ):
        run_all_days(
            day_batches=[DayBatch(date=DATE, episodes=[ep])],
            drive_scores=scores,
            file_boundaries=[(0.0, 600.0)],
            output_base=out,
            location="Colleton County, SC",
            config=config,
            episode_paths=["/fake/ep.mp4"],
            lf_plans_by_date=plans,
            per_lf_ost_text_by_date={DATE: {1: "A to B via R1, SC"}},
            state={"ost_audit_confirmed_this_run": True} if state is None else state,
            **extra,
        )
    return out


def placed_one(clip):
    return {DATE: {1: [Insert(30.0, clip, capture=(1.0, 2.0))]}}


def test_run_all_days_threads_inserts_into_the_audit_and_the_day_pipeline(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    config = _drive_cfg(tmp_path, clip)
    seen = []
    _run_all(
        tmp_path,
        lambda **kw: seen.append(kw["inserts_by_lf"]) or None,
        lambda *a, **k: placed_one(clip),
        config,
    )
    assert [[i.media.name for i in v[1]] for v in seen] == [["c.mp4"]]
    with pytest.raises(OstAuditPendingError):  # unconfirmed: the sidecar is written, then it stops
        _run_all(tmp_path, lambda **kw: None, lambda *a, **k: placed_one(clip), config, state={})
    audit = json.loads((tmp_path / "stages" / f"ost_audit_{DATE}.json").read_text())
    assert audit["lfs"][0]["inserts"][0]["media"] == "c.mp4"


def test_run_all_days_refuses_an_ungated_insert_before_the_audit_and_any_encode(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    config = _drive_cfg(tmp_path, clip)
    calls = []

    def refuse(*a, **k):
        raise InsertError("c.mp4: no capture position, cannot check the LF's counties")

    with pytest.raises(InsertError):
        _run_all(tmp_path, lambda **kw: calls.append(kw), refuse, config)
    assert calls == []
    assert not (tmp_path / "stages" / f"ost_audit_{DATE}.json").exists()


def test_run_all_days_plans_against_every_day_and_gates_only_the_rendered_ones(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    config = _drive_cfg(tmp_path, clip)
    done = SimpleNamespace(date="2026-10-02", episodes=[])
    gps = {"2026-10-02": [(0.0, 1.0, 2.0)]}
    planned = {}

    def planner(cfg, batches, **kw):
        planned.update(batches=list(batches), kw=kw)
        return placed_one(clip)

    _run_all(
        tmp_path,
        lambda **kw: None,
        planner,
        config,
        all_day_batches=[done, SimpleNamespace(date=DATE, episodes=[])],
        all_gps_points_by_date=gps,
    )
    assert [b.date for b in planned["batches"]] == ["2026-10-02", DATE]
    assert planned["kw"]["render_dates"] == {DATE}
    assert planned["kw"]["gps_points_by_date"] == gps


# ── phase BC: the resume ─────────────────────────────────────────────────────────────


def _phase(state, cfg, fake_run_all_days):
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

    geocoder = MagicMock()
    geocoder.reverse_geocode.return_value = None

    class _Probe:
        def __init__(self, _m, *, timeout=30):
            pass

        def probe_duration(self, path):
            return 1500.0

    with (
        patch("gomoveshift.pipeline.gps_enrichment._build_shared_geocoder", return_value=geocoder),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=fake_run_all_days,
        ) as run_mock,
        patch("gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient", _Probe),
    ):
        PhaseBCDayFirst(cfg).run(state, probe_duration_fn=lambda p: 100.0)
    return run_mock


def _resumed_state(tmp_path, saved_request):
    from test_phase_bc_fast_path_master_integrity import _make_state, _write_finals

    state, _, stages = _make_state(tmp_path)
    done = "2026-02-20"
    master = stages / f"day_{done}_master.mp4"
    master.write_bytes(b"x" * 1024)
    _write_finals(stages / f"day_{done}")
    rec = {"kind": "still", "media": "a.jpg", "at_s": 95.0, "style": "full", "hold_s": 4.0}
    state["per_day_completed"] = {
        done: {
            "master_path": str(master),
            "deliverables": {"date": done, "inserts": {"1": [rec]}},
        }
    }
    state["insert_request"] = saved_request
    return state, stages, rec


def _fresh_day(stages):
    def fake(**kw):
        out = []
        for b in kw["day_batches"]:
            m = stages / f"day_{b.date}_master.mp4"
            m.write_bytes(b"fresh")
            out.append(
                SimpleNamespace(date=b.date, master_path=m, regular_lf_paths=[], short_paths=[])
            )
        return out

    return fake


def test_a_resume_without_the_original_insert_flags_refuses_loudly(tmp_path):
    state, stages, _ = _resumed_state(tmp_path, {"photos": "/p", "clips": []})
    with pytest.raises(InsertError, match="--insert-photos|/p"):
        _phase(state, _config(), _fresh_day(stages))


def test_a_resume_with_the_same_flags_plans_every_day_and_keeps_the_finished_inserts(tmp_path):
    state, stages, rec = _resumed_state(tmp_path, {"photos": "/p", "clips": []})
    cfg = _config()
    cfg.drive.insert_photos = "/p"
    run_mock = _phase(state, cfg, _fresh_day(stages))
    kw = run_mock.call_args.kwargs
    assert [b.date for b in kw["day_batches"]] == ["2026-02-21"]
    assert [b.date for b in kw["all_day_batches"]] == ["2026-02-20", "2026-02-21"]
    assert state.insert_request == {"photos": "/p", "clips": []}
    kept = state.per_day_completed["2026-02-20"]["deliverables"]["inserts"]
    assert kept == {1: [rec]} or kept == {"1": [rec]}


def test_a_resume_with_no_flags_and_none_before_is_untouched(tmp_path):
    state, stages, _ = _resumed_state(tmp_path, None)
    run_mock = _phase(state, _config(), _fresh_day(stages))
    assert run_mock.call_args.kwargs.get("all_day_batches") is None
    assert state.insert_request is None


# ── Phase D ──────────────────────────────────────────────────────────────────────────


def test_chapter_markers_and_description_count():
    from gomoveshift.pipeline.chapters.marker_merger import add_insert_markers
    from gomoveshift.pipeline.longform_metadata import _format_insert_line

    chapters = [
        {"timestamp": "0:00", "title": "Start"},
        {"timestamp": "2:00", "title": "Town"},
    ]
    out = add_insert_markers(chapters, [30.0, 31.0, 125.0])
    stamps = [c["timestamp"] for c in out]
    assert stamps == sorted(stamps, key=lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
    marks = [c for c in out if c["title"].startswith("\U0001f4f8")]
    assert [c["timestamp"] for c in marks] == [
        "0:30"
    ]  # 31 s and 125 s sit within 10 s of a chapter
    assert add_insert_markers(chapters, []) == chapters
    assert _format_insert_line(0) == ""
    assert _format_insert_line(1) == "\U0001f4f8 1 photo/clip insert\n"
    assert _format_insert_line(3) == "\U0001f4f8 3 photo/clip inserts\n"


def test_inserts_reach_the_chapters_and_the_description_in_phase_d(tmp_path):
    from _state_helpers import make_state
    from gomoveshift.pipeline.drive_phases import phase_d, phase_d_metadata
    from gomoveshift.pipeline.media_api import Deliverables, deliverables_to_state

    recs = {1: [{"kind": "still", "media": "a.jpg", "at_s": 95.0, "style": "full", "hold_s": 4.0}]}
    d = Deliverables(
        date="2026-10-03",
        master_path=None,
        regular_lf_paths=[tmp_path / "lf_01.mp4"],
        regular_lf_indices=[1],
        inserts=recs,
    )
    wire = json.loads(json.dumps(deliverables_to_state(d)))  # state is JSON on disk
    state = make_state(per_day_completed={"2026-10-03": {"deliverables": wire}})
    assert phase_d_metadata._lf_inserts(state) == recs  # int keys come back

    state.per_lf_gpx_path = {1: str(tmp_path / "t.gpx")}
    (tmp_path / "t.gpx").write_text("x")
    lf = SimpleNamespace(index=1, total_duration=300.0, segments=[])
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {"title": "t", "description": "d"}
    plan = SimpleNamespace(regular_lfs=[lf])
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        meta = phase_d_metadata._build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=plan,
            regular_lf_paths=["/v/lf.mp4"],
            state=state,
            location="x",
            publish_day_full="Wed",
            project_base="p",
            output_base=tmp_path,
        )
        assert gen.generate.call_args.kwargs["insert_count"] == 1
        chapters = [{"timestamp": f"{m}:00", "title": f"c{m}"} for m in (0, 2, 4)]
        with patch.object(
            phase_d,
            "_generate_chapters_for_lf",
            return_value=(chapters, "0:00 c0\n2:00 c2\n4:00 c4"),
        ):
            phase_d_metadata._generate_chapters_all_lfs(
                config=_config(),
                drive_plan=plan,
                state=state,
                metadata_per_lf=meta,
                location="x",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
    assert "1:35 \U0001f4f8 Photo stop" in meta[1]["chapter_text"]
    assert any(c["timestamp"] == "1:35" for c in meta[1]["chapters"])
