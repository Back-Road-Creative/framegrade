"""Caption experiment arms: policy block, deterministic assignment, hashtag modes,
sidecar tag round-trip, per-arm Facebook reach report."""

from datetime import date

import pytest

from gomoveshift.publishing.brand_policy import load_brand_policy
from gomoveshift.publishing.experiments import (
    arm_report_lines,
    arm_results,
    assign_arms,
    parse_experiments,
)

START = date(2026, 11, 2)


def _block(**over):
    block = {
        "name": "fb_hashtags",
        "platforms": ["facebook"],
        "start": "2026-11-02",
        "min_n": 2,
        "stop_after_days": 45,
        "arms": {
            "A": {"hashtag_mode": "none"},
            "B": {"hashtag_mode": "location"},
            "C": {"hashtag_mode": "june"},
        },
    }
    block.update(over)
    return block


def _write_policy(tmp_path, monkeypatch, yaml_text):
    import gomoveshift.publishing.brand_policy as mod

    d = tmp_path / "brands" / "exp"
    d.mkdir(parents=True)
    (d / "brand.yaml").write_text(yaml_text)
    monkeypatch.setattr(mod, "_SHIPPED_BRANDS_DIR", tmp_path / "brands")


def test_policy_loads_experiments(tmp_path, monkeypatch):
    _write_policy(
        tmp_path,
        monkeypatch,
        "policy:\n  experiments:\n"
        "    - name: fb_hashtags\n      platforms: [facebook]\n      start: 2026-11-02\n"
        "      min_n: 20\n      stop_after_days: 45\n      arms:\n"
        "        A: {hashtag_mode: none}\n        B: {hashtag_mode: location}\n"
        "        C: {hashtag_mode: june}\n",
    )
    (exp,) = load_brand_policy("exp").experiments
    assert exp.name == "fb_hashtags"
    assert [a.id for a in exp.arms] == ["A", "B", "C"]
    assert exp.min_n == 20 and exp.start == START


def test_no_experiments_by_default():
    assert load_brand_policy("no-such-brand").experiments == ()


@pytest.mark.parametrize(
    "bad, needle",
    [
        ({"platforms": ["tiktok"]}, "platforms"),
        ({"min_n": 0}, "min_n"),
        ({"stop_after_days": 0}, "stop_after_days"),
        ({"start": "soon"}, "start"),
        ({"arms": {"A": {"hashtag_mode": "none"}}}, "arms"),
        ({"arms": {"A": {"hashtag_mode": "none"}, "B": {"hashtag_mode": "loud"}}}, "hashtag_mode"),
        (
            {"arms": {"A": {"hashtag_mode": "none"}, "B": {"hashtag_mode": "none", "weight": 0}}},
            "weight",
        ),
        ({"name": ""}, "name"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_invalid_experiment_block_fails_loudly(bad, needle):
    with pytest.raises(ValueError, match=rf"policy\.experiments.*{needle}"):
        parse_experiments([_block(**bad)])


def test_duplicate_experiment_name_refused():
    with pytest.raises(ValueError, match="duplicate"):
        parse_experiments([_block(), _block()])


def test_assignment_is_deterministic_and_balanced():
    exps = parse_experiments([_block()])
    first = [assign_arms(exps, "proj", i, START) for i in range(1, 301)]
    assert first == [assign_arms(exps, "proj", i, START) for i in range(1, 301)]
    counts = {a: sum(1 for t in first if t["fb_hashtags"] == a) for a in "ABC"}
    assert all(60 <= n <= 140 for n in counts.values()), counts


def test_assignment_stops_after_window_and_before_start():
    exps = parse_experiments([_block(stop_after_days=45)])
    assert assign_arms(exps, "proj", 1, date(2026, 12, 16))  # day 44, last active day
    assert assign_arms(exps, "proj", 1, date(2026, 12, 17)) == {}
    assert assign_arms(exps, "proj", 1, date(2026, 11, 1)) == {}


def test_weights_bias_assignment():
    exps = parse_experiments(
        [_block(arms={"A": {"hashtag_mode": "none", "weight": 9}, "B": {"hashtag_mode": "june"}})]
    )
    tags = [assign_arms(exps, "p", i, START)["fb_hashtags"] for i in range(1, 201)]
    assert tags.count("A") > tags.count("B") * 3


# --- report -------------------------------------------------------------

TODAY = date(2026, 12, 1)


def _obs(arm, reach, posted="2026-11-10T12:00:00"):
    return (arm, posted, reach)


def test_report_waits_for_min_n_in_every_arm_and_never_counts_unknown():
    exp = parse_experiments([_block(min_n=2)])[0]
    obs = [
        _obs("A", 100),
        _obs("A", 300),
        _obs("A", None),  # unknown reach: not n
        _obs("B", 50),
        _obs("B", 70),
        _obs("C", 10),  # C below min_n
        _obs("C", 20, posted="2026-11-29T10:00:00"),  # window not settled: not n
    ]
    res = {r.arm: r for r in arm_results(exp, obs, TODAY)}
    assert (res["A"].n, res["B"].n, res["C"].n) == (2, 2, 1)
    assert all(r.median_reach is None for r in res.values())  # C short: nothing compared yet
    obs.append(_obs("C", 30))
    res = {r.arm: r for r in arm_results(exp, obs, TODAY)}
    assert (res["A"].median_reach, res["B"].median_reach, res["C"].median_reach) == (200, 60, 20)


def test_report_lines_format_and_unknown_is_not_zero():
    exp = parse_experiments([_block(min_n=2)])[0]
    lines = arm_report_lines(arm_results(exp, [_obs("A", 0), _obs("A", 0)], TODAY))
    assert lines[0] == "EXPERIMENT fb_hashtags platform=facebook arm=A n=2 median_reach=unknown"
    assert lines[1] == "EXPERIMENT fb_hashtags platform=facebook arm=B n=0 median_reach=unknown"
    done = arm_report_lines(arm_results(exp, [_obs(a, 0) for a in "AABBCC"], TODAY))
    assert done[0].endswith("n=2 median_reach=0")  # a measured zero is a value


def test_arm_observations_join_sidecar_tags_to_facebook_reach():
    from types import SimpleNamespace as Ns

    from gomoveshift.publishing.experiments import arm_observations

    exp = parse_experiments([_block()])[0]
    short = lambda i, reach: Ns(  # noqa: E731
        index=i, posted_at="2026-11-10T12:00:00", metrics={"facebook": Ns(reach=reach)}
    )
    records = [Ns(project="p", shorts=[short(1, 5), short(2, None), short(3, 9)])]
    tags = {1: {"fb_hashtags": "A"}, 2: {"fb_hashtags": "B"}, 3: {}}
    rows = arm_observations(exp, records, lambda project: tags)
    assert rows == [("A", "2026-11-10T12:00:00", 5), ("B", "2026-11-10T12:00:00", None)]
    assert arm_observations(exp, records, lambda project: None) == []


def test_shipped_gomoveshift_brand_runs_the_fb_hashtag_experiment(monkeypatch):
    monkeypatch.delenv("GMS_BRANDS_DIR", raising=False)
    (exp,) = load_brand_policy("gomoveshift").experiments
    assert exp.name == "fb_hashtags"
    assert exp.platforms == ("facebook",)
    assert exp.start == date(2026, 11, 2)
    assert (exp.min_n, exp.stop_after_days) == (20, 45)
    assert [(a.id, a.hashtag_mode) for a in exp.arms] == [
        ("A", "none"),
        ("B", "location"),
        ("C", "june"),
    ]
