"""YouTube Shorts title/hashtag arms: stratified assignment, the 72 h outcome,
and the per-arm decision record (hold / demote / pause / INSUFFICIENT_EVIDENCE)."""

from collections import Counter
from datetime import date, timedelta
from types import SimpleNamespace as Ns

import pytest

from gomoveshift.publishing.brand_policy import load_brand_policy
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


def _entry(hours, views, pct, status="measured", posted="2026-11-01T12:00:00+00:00"):
    from datetime import datetime

    event = datetime.fromisoformat(posted)
    return {
        "status": status,
        "event_at": posted,
        "fetched_at": (event + timedelta(hours=hours)).isoformat(),
        "cumulative": {"views": views, "avg_view_percentage": pct},
    }


def _short(index, log, posted="2026-11-01T12:00:00+00:00"):
    return Ns(index=index, posted_at=posted, observation_log={"youtube": log}, metrics={})


def test_outcome_reads_the_snapshot_nearest_72h_with_views_and_retention():
    s = _short(1, [_entry(24, 10, 80.0), _entry(71, 120, 61.5), _entry(168, 400, 50.0)])
    assert youtube_outcome(s, TODAY) == (120, 61.5)


def test_outcome_is_unknown_without_a_snapshot_near_72h_or_while_unsettled():
    assert youtube_outcome(_short(1, [_entry(24, 10, 80.0), _entry(168, 400, 50.0)]), TODAY) is None
    assert youtube_outcome(_short(1, [_entry(72, 5, 1.0, status="rate_limited")]), TODAY) is None
    fresh = _short(1, [_entry(72, 50, 60.0)])
    assert youtube_outcome(fresh, date(2026, 11, 5)) is None  # posted+3d > today-2d
    assert youtube_outcome(fresh, date(2026, 11, 6)) == (50, 60.0)


def test_unreported_retention_stays_unknown_never_zero():
    assert youtube_outcome(_short(1, [_entry(72, 50, None)]), TODAY) == (50, None)


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
    rows = shorts_arm_outcomes("yt_title_body", records, lambda project: tags, TODAY)
    assert [(r.arm, r.views_72h, r.avg_view_pct) for r in rows] == [
        ("A", 10, 50.0),
        ("B", 20, 60.0),
    ]
    assert shorts_arm_outcomes("yt_title_body", records, lambda project: None, TODAY) == []


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
    assert got["B"].decision == "demote" or got["B"].decision == "pause"
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


def test_brand_policy_carries_the_thresholds(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as mod

    d = tmp_path / "brands" / "exp"
    d.mkdir(parents=True)
    (d / "brand.yaml").write_text("policy:\n  arm_thresholds:\n    min_n: 40\n")
    monkeypatch.setattr(mod, "_SHIPPED_BRANDS_DIR", tmp_path / "brands")
    assert load_brand_policy("exp").arm_thresholds == ArmThresholds(min_n=40)
    assert load_brand_policy("missing-brand").arm_thresholds == ArmThresholds()
