"""YouTube Shorts title/hashtag arms: stratified assignment, the 72 h outcome,
and the per-arm decision record (hold / demote / pause / INSUFFICIENT_EVIDENCE)."""

from collections import Counter
from datetime import date, timedelta
from types import SimpleNamespace as Ns

import pytest

from gomoveshift.publishing.brand_policy import load_arm_thresholds, load_brand_policy
from gomoveshift.publishing.experiments import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_N,
    DEFAULT_PAUSE_GAP,
    PAUSE_RETEST_DAYS,
    SHORTS_FACTORS,
    ArmThresholds,
    ShortOutcome,
    assign_stratified_arm,
    decide_shorts_arms,
    decision_lines,
    parse_arm_thresholds,
    parse_experiments,
    pause_retest_due,
    shorts_arm_outcomes,
    youtube_outcome,
)

TODAY = date(2026, 12, 20)
ARMS = ("A", "B")


def test_defaults_are_the_conservative_jp_approved_values():
    assert (DEFAULT_MIN_N, DEFAULT_MIN_GAP, PAUSE_RETEST_DAYS) == (36, 3.0, 28)
    assert DEFAULT_PAUSE_GAP > DEFAULT_MIN_GAP
    assert SHORTS_FACTORS == ("yt_title_body", "yt_hashtag_set")


# --- stratified assignment ---------------------------------------------------


def _assign_stratum(arm_ids, stratum, keys):
    assigned = []
    for key in keys:
        assigned.append(assign_stratified_arm("yt_title_body", arm_ids, stratum, key, assigned))
    return assigned


@pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 11])
def test_arms_balance_within_one_in_every_project_slot_stratum(n):
    for stratum in (("p1", "mon-18"), ("p1", "tue-18"), ("p2", "mon-18")):
        counts = Counter(_assign_stratum(("A", "B", "C"), stratum, range(1, n + 1)))
        sizes = [counts.get(a, 0) for a in "ABC"]
        assert max(sizes) - min(sizes) <= 1


def test_assignment_is_deterministic_and_independent_of_process():
    first = _assign_stratum(ARMS, ("p", "s"), range(1, 30))
    assert first == _assign_stratum(ARMS, ("p", "s"), range(1, 30))
    # the first arm of a stratum is not always "A": the tie-break is a hash
    firsts = {_assign_stratum(ARMS, ("p", f"slot{i}"), [1])[0] for i in range(40)}
    assert firsts == {"A", "B"}


def test_factors_are_assigned_independently():
    a = _assign_stratum(ARMS, ("p", "s"), range(1, 40))
    b = []
    for key in range(1, 40):
        b.append(assign_stratified_arm("yt_hashtag_set", ARMS, ("p", "s"), key, b))
    assert a != b


# --- the 72 h outcome --------------------------------------------------------


YT = "2026-11-01T12:00:00+00:00"  # YouTube's own publish time


def _entry(hours, views, pct, status="measured", yt=YT, post_id="yt1"):
    from datetime import datetime

    return {
        "status": status,
        "post_id": post_id,
        "event_at": "2026-11-01T20:00:00+00:00",  # copied from posted_at (a later platform)
        "fetched_at": (datetime.fromisoformat(yt) + timedelta(hours=hours)).isoformat(),
        "cumulative": {"views": views, "avg_view_percentage": pct},
    }


def _short(index, log, posted="2026-11-01T20:00:00+00:00", youtube_id="yt1"):
    return Ns(
        index=index,
        posted_at=posted,
        platform_ids={"youtube": youtube_id} if youtube_id else {},
        observation_log={"youtube": log},
        metrics={},
    )


def test_outcome_reads_the_snapshot_nearest_72h_with_views_and_retention():
    s = _short(1, [_entry(24, 10, 80.0), _entry(71, 120, 61.5), _entry(168, 400, 50.0)])
    assert youtube_outcome(s, YT, TODAY) == (120, 61.5)


def test_outcome_is_unknown_without_a_snapshot_near_72h_or_while_unsettled():
    assert (
        youtube_outcome(_short(1, [_entry(24, 10, 80.0), _entry(168, 400, 50.0)]), YT, TODAY)
        is None
    )
    assert (
        youtube_outcome(_short(1, [_entry(72, 5, 1.0, status="rate_limited")]), YT, TODAY) is None
    )
    fresh = _short(1, [_entry(72, 50, 60.0)])
    assert youtube_outcome(fresh, YT, date(2026, 11, 5)) is None  # posted+3d > today-2d
    assert youtube_outcome(fresh, YT, date(2026, 11, 6)) == (50, 60.0)


def test_unreported_retention_stays_unknown_never_zero():
    assert youtube_outcome(_short(1, [_entry(72, 50, None)]), YT, TODAY) == (50, None)


def test_outcome_rows_join_sidecar_tags_for_a_factor():
    records = [
        Ns(
            project="p",
            shorts=[
                _short(1, [_entry(72, 10, 50.0)]),
                _short(2, [_entry(72, 20, 60.0)]),
                _short(3, [_entry(72, 30, 70.0)]),
            ],
        )
    ]
    tags = {1: {"yt_title_body": "A"}, 2: {"yt_title_body": "B"}, 3: {"yt_hashtag_set": "X"}}
    times = {("p", i): YT for i in (1, 2, 3)}
    rows = shorts_arm_outcomes("yt_title_body", records, lambda project: tags, times, TODAY)
    assert [(r.arm, r.views_72h, r.avg_view_pct) for r in rows] == [
        ("A", 10, 50.0),
        ("B", 20, 60.0),
    ]
    assert shorts_arm_outcomes("yt_title_body", records, lambda p: None, times, TODAY) == []
    # no YouTube time known: no outcome, never a fall back to posted_at
    assert shorts_arm_outcomes("yt_title_body", records, lambda p: tags, {}, TODAY) == []


# --- decisions on a recorded-shape fixture ------------------------------------


def _fixture(a_views, b_views, projects=6, per=6, a_pct=60.0, b_pct=60.0):
    """Within every project, ``per`` Shorts of arm A then ``per`` of arm B, posted
    on consecutive days; ``*_views`` are the per-project base views."""
    rows = []
    for p in range(projects):
        for i in range(per):
            day = date(2026, 11, 1) + timedelta(days=p * 2 + i // 3)
            rows.append(ShortOutcome(f"proj{p}", "A", day, a_views + i, a_pct))
            rows.append(ShortOutcome(f"proj{p}", "B", day, b_views + i, b_pct))
    return rows


def _by_arm(rows, thresholds=None):
    return {
        d.arm: d for d in decide_shorts_arms("yt_title_body", rows, thresholds or ArmThresholds())
    }


def test_below_min_sample_is_insufficient_evidence_even_with_a_huge_gap():
    got = _by_arm(_fixture(1000, 10, projects=5, per=6))  # 30 Shorts per arm < 36
    assert {d.decision for d in got.values()} == {"INSUFFICIENT_EVIDENCE"}
    assert got["B"].reason == "n_below_min"


def test_clear_repeated_gap_demotes_only_the_lagging_arm():
    got = _by_arm(_fixture(1000, 100))  # 36 per arm, A is ~10x B
    # ~10x gap >= pause_gap and retention equal (no better): the exact decision is pause
    assert (got["B"].decision, got["B"].reason) == ("pause", "clear_gap_repeated")
    assert got["A"].decision == "hold"


def test_gap_below_three_times_holds():
    got = _by_arm(_fixture(1000, 400))  # 2.5x
    assert {d.decision for d in got.values()} == {"hold"}


def test_pause_needs_the_larger_gap_and_no_better_retention():
    assert _by_arm(_fixture(1000, 100, b_pct=40.0))["B"].decision == "pause"
    # a 4x gap with retention is only a demote
    assert _by_arm(_fixture(1000, 250, b_pct=40.0))["B"].decision == "demote"
    # retention better than the other arm's: demote, not pause
    assert _by_arm(_fixture(1000, 100, b_pct=80.0))["B"].decision == "demote"
    # retention unreported: demote, not pause
    assert _by_arm(_fixture(1000, 100, b_pct=None))["B"].decision == "demote"


def test_a_gap_that_does_not_repeat_across_the_arms_two_halves_holds():
    rows = _fixture(1000, 100)
    # B recovers to A's level over the latest 40% of days: the overall gap
    # stands but the later half of B's Shorts does not show it again
    days = sorted({r.posted for r in rows})
    later = set(days[int(len(days) * 0.6) :])
    rows = [
        ShortOutcome(r.project, r.arm, r.posted, 1000 + (r.views_72h % 100), r.avg_view_pct)
        if r.arm == "B" and r.posted in later
        else r
        for r in rows
    ]
    got = _by_arm(rows)
    assert got["B"].decision == "hold" and got["B"].reason == "gap_not_repeated"


def test_decision_uses_within_project_medians_not_pooled_volume():
    """B wins every project head to head but lives in the low-view projects: a
    pooled median would call it a laggard; within-project it is a winner."""
    rows = []
    for p in range(6):
        base = 10_000 if p < 3 else 100
        for i in range(6):
            day = date(2026, 11, 1) + timedelta(days=p + i // 3)
            rows.append(ShortOutcome(f"p{p}", "A", day, base + i, 60.0))
            rows.append(ShortOutcome(f"p{p}", "B", day, base * 2 + i, 60.0))
    got = _by_arm(rows)
    assert got["A"].decision == "hold" and got["B"].decision == "hold"


def test_projects_with_one_arm_or_a_zero_median_do_not_count():
    rows = _fixture(1000, 100)
    rows += [ShortOutcome("solo", "A", date(2026, 11, 3), 1, 50.0) for _ in range(40)]
    rows += [ShortOutcome("zero", a, date(2026, 11, 3), 0, None) for a in ARMS for _ in range(40)]
    got = _by_arm(rows)
    assert got["A"].n == 36 and got["B"].n == 36


def test_single_arm_is_insufficient_evidence_and_input_order_is_irrelevant():
    only_a = [r for r in _fixture(1000, 100) if r.arm == "A"]
    assert decide_shorts_arms("f", only_a, ArmThresholds()) == []  # no within-project comparison
    rows = _fixture(1000, 100)
    assert decide_shorts_arms("f", rows, ArmThresholds()) == decide_shorts_arms(
        "f", list(reversed(rows)), ArmThresholds()
    )


def test_thresholds_are_configurable():
    rows = _fixture(1000, 400)  # 2.5x
    assert _by_arm(rows, ArmThresholds(min_gap=2.0))["B"].decision == "demote"
    assert _by_arm(rows, ArmThresholds(min_n=37))["B"].decision == "INSUFFICIENT_EVIDENCE"


def test_pause_retest_is_on_a_fixed_28_day_schedule():
    paused = date(2026, 12, 1)
    assert not pause_retest_due(paused, paused + timedelta(days=27))
    assert pause_retest_due(paused, paused + timedelta(days=28))
    assert not pause_retest_due(paused, paused + timedelta(days=5), ArmThresholds(retest_days=10))


def test_decision_lines_name_every_arm_and_the_retest_for_a_pause():
    lines = decision_lines(
        "yt_title_body",
        decide_shorts_arms("yt_title_body", _fixture(1000, 100, b_pct=40.0), ArmThresholds()),
    )
    assert lines[0].startswith("ARM_DECISION factor=yt_title_body arm=A decision=hold n=36 ")
    assert "arm=B decision=pause n=36" in lines[1] and "retest_days=28" in lines[1]
    assert "retest_days" not in lines[0]
    none = decision_lines("yt_title_body", decide_shorts_arms("yt_title_body", [], ArmThresholds()))
    assert none == [
        "ARM_DECISION factor=yt_title_body arm=- decision=INSUFFICIENT_EVIDENCE n=0 reason=no_outcomes"
    ]


# --- config ------------------------------------------------------------------


def test_parse_arm_thresholds_defaults_validates_and_rejects_unknown_keys():
    assert parse_arm_thresholds(None) == ArmThresholds()
    assert parse_arm_thresholds({"min_n": 50, "min_gap": 4}) == ArmThresholds(min_n=50, min_gap=4.0)
    for bad in (
        {"min_n": 0},
        {"min_gap": 1.0},
        {"pause_gap": 2.0},
        {"nope": 1},
        {"min_n": True},
        "x",
    ):
        with pytest.raises(ValueError):
            parse_arm_thresholds(bad)


def _brand(tmp_path, monkeypatch, policy_yaml):
    import gomoveshift.publishing.brand_policy as mod

    d = tmp_path / "brands" / "exp"
    d.mkdir(parents=True)
    (d / "brand.yaml").write_text(policy_yaml)
    monkeypatch.setattr(mod, "_SHIPPED_BRANDS_DIR", tmp_path / "brands")


def test_arm_thresholds_load_in_the_report_path_only(tmp_path, monkeypatch):
    _brand(tmp_path, monkeypatch, "policy:\n  arm_thresholds:\n    min_n: 40\n")
    assert load_arm_thresholds("exp") == ArmThresholds(min_n=40)
    assert load_arm_thresholds("missing-brand") == ArmThresholds()
    load_brand_policy("exp")  # a known key, not flagged unknown


def test_a_bad_arm_thresholds_value_never_breaks_load_brand_policy(tmp_path, monkeypatch):
    _brand(
        tmp_path, monkeypatch, "policy:\n  weekly_ceiling: 5\n  arm_thresholds:\n    min_gap: 0.5\n"
    )
    assert load_brand_policy("exp").weekly_ceiling == 5
    with pytest.raises(ValueError, match="min_gap"):
        load_arm_thresholds("exp")


def test_thresholds_reject_a_min_n_too_small_to_split_and_non_finite_gaps():
    for bad in ({"min_n": 1}, {"min_n": 3}, {"min_gap": float("nan")}, {"pause_gap": float("inf")}):
        with pytest.raises(ValueError):
            parse_arm_thresholds(bad)
    assert parse_arm_thresholds({"min_n": 4}).min_n == 4


def test_an_experiment_cannot_take_a_shorts_factor_name():
    for name in SHORTS_FACTORS:
        with pytest.raises(ValueError, match="Shorts factor"):
            parse_experiments([{"name": name, "platforms": ["facebook"], "start": "2026-11-01"}])


# --- review fixes: YouTube's own clock, id match, zero reference, unseen arms -


def test_window_is_timed_from_youtube_not_from_the_later_platform():
    # YouTube 10:00, Facebook registered 18:00 (posted_at): real YouTube ages
    # 65 h and 89 h. The 89 h snapshot is 17 h off 72 h: no outcome from it.
    yt = "2026-11-01T10:00:00+00:00"
    late = "2026-11-01T18:00:00+00:00"
    assert youtube_outcome(_short(1, [_entry(89, 70, 50.0, yt=yt)], posted=late), yt, TODAY) is None
    both = [_entry(65, 60, 55.0, yt=yt), _entry(89, 90, 45.0, yt=yt)]
    assert youtube_outcome(_short(1, both, posted=late), yt, TODAY) == (60, 55.0)
    # timed from posted_at (the old bug) the 89 h entry would read as 81 h: in window
    assert (
        youtube_outcome(_short(1, [_entry(89, 70, 50.0, yt=yt)], posted=late), None, TODAY) is None
    )


def test_an_old_video_ids_entry_is_not_picked_after_a_reupload():
    old = _short(1, [_entry(72, 99, 50.0, post_id="old")], youtube_id="new")
    assert youtube_outcome(old, YT, TODAY) is None
    assert youtube_outcome(_short(1, [_entry(72, 99, 50.0)], youtube_id=None), YT, TODAY) is None


def test_an_arm_equal_to_a_zero_reference_holds_and_is_never_paused():
    from gomoveshift.publishing.experiments import _gap

    assert _gap(0.0, 0.0) == 0.0
    assert _gap(1.0, 0.0) == float("inf")
    assert _gap(2.0, 1.0) == 2.0


def test_every_arm_seen_gets_a_line_even_with_no_countable_short():
    rows = [r for r in _fixture(1000, 100) if r.arm == "A"]
    got = decide_shorts_arms("yt_title_body", rows, ArmThresholds(), seen_arms=["A", "B"])
    b = {d.arm: d for d in got}["B"]
    assert (b.decision, b.n) == ("INSUFFICIENT_EVIDENCE", 0)
    lines = decision_lines("yt_title_body", got)
    assert any("arm=B decision=INSUFFICIENT_EVIDENCE n=0" in line for line in lines)


# --- the metrics handler report -----------------------------------------------


def _handler_world(tmp_path, monkeypatch, youtube_at):
    import json as _json

    from gomoveshift.cli.handlers import metrics as handler
    from gomoveshift.publishing.queue import open_queue_db

    conn = open_queue_db(tmp_path)
    conn.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, "
        "platform_upload_id, published_at, published_at_source, created_at, updated_at) "
        "VALUES ('proj', 1, 'youtube', 'h', 'published', 'yt1', ?, 'platform_confirmed', 'x', 'x')",
        (youtube_at,),
    )
    conn.commit()
    conn.close()
    plan = tmp_path / "proj_publish_plan.json"
    plan.write_text(_json.dumps({}))
    short = Ns(
        index=1,
        posted_at="2026-11-01T20:00:00+00:00",
        platform_ids={"youtube": "yt1"},
        observation_log={"youtube": [_entry(72, 50, 60.0)]},
        metrics={},
    )
    monkeypatch.setattr(
        handler,
        "load_publish_plan",
        lambda path: Ns(shorts=[Ns(short_index=1, experiment_arms={"yt_title_body": "A"})]),
    )
    monkeypatch.setattr(
        handler,
        "TrackingStore",
        lambda: Ns(load_all_records=lambda: [Ns(project="proj", shorts=[short])]),
    )
    monkeypatch.setattr(handler, "load_arm_thresholds", lambda brand: ArmThresholds())
    return handler, Ns(seo=Ns(storage_path=str(tmp_path)))


def test_report_prints_one_line_per_arm_seen(tmp_path, monkeypatch, capsys):
    handler, config = _handler_world(tmp_path, monkeypatch, YT)
    handler._report_shorts_arm_decisions(config, "brand")
    out = capsys.readouterr().out
    assert "ARM_DECISION factor=yt_title_body arm=A decision=INSUFFICIENT_EVIDENCE n=0" in out
    assert "ARM_DECISION factor=yt_hashtag_set arm=- " in out  # no tagged arm: one placeholder


def test_report_failure_is_logged_and_swallowed(tmp_path, monkeypatch, capsys, caplog):
    handler, config = _handler_world(tmp_path, monkeypatch, YT)

    def boom(brand):
        raise ValueError("policy.arm_thresholds.min_gap bad")

    monkeypatch.setattr(handler, "load_arm_thresholds", boom)
    with caplog.at_level("WARNING"):
        handler._report_shorts_arm_decisions(config, "brand")  # must not raise
    assert "ARM_DECISION" not in capsys.readouterr().out
    assert "shorts arm decision report skipped" in caplog.text
