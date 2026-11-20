"""Caption experiments declared in a brand's ``policy.experiments`` block.

An experiment only chooses WHICH caption treatment a short gets; how many
shorts post and when stays the cadence's (Waypost's) decision. Each arm names
a ``hashtag_mode`` the caption generator honours on the experiment's platforms:

* ``none``     -- the composed caption, no hashtags;
* ``location`` -- the composed caption plus 2 location hashtags;
* ``june``     -- June 2026 shape: ``#POVDriving #ScenicDrive #RoadTrip`` + the two location tags.

Assignment is a stable hash of ``experiment name + project + short index``,
never random at runtime, and stops for good ``stop_after_days`` after
``start``. The tag travels in the publish-plan sidecar
(``PublishRecord.experiment_arms``); ``deliveries`` has no column for it and
adding one would need a migration.

The outcome is Facebook Reel initial plays (``PlatformMetrics.reel_initial_plays``):
Graph answers no per-Reel reach for the live Page, so every Reel's ``reach`` is
null (``docs/facebook-metrics.md``). Plays not yet measured are unknown, never 0,
and a short counts toward an arm's ``n`` only once its day-7 window is settled
(``tracking.day7.window_settled``).

YouTube Shorts arms (title body, hashtag set) are the second half of this
module: ``assign_stratified_arm`` balances arms within a (project, slot)
stratum, ``youtube_outcome`` reads a Short's 72 h views and average view
percentage from the stored observation series, and ``decide_shorts_arms``
turns the settled outcomes into one decision record per arm -- ``hold``,
``demote``, ``pause`` or ``INSUFFICIENT_EVIDENCE``. Every decision is a pure
function of recorded numbers and :class:`ArmThresholds`; nothing here picks
text to send (the caption/title generator does), and no LLM is involved.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HASHTAG_MODES = frozenset({"none", "location", "june"})
#: Facebook Reel plays are the only experiment outcome measured today.
EXPERIMENT_PLATFORMS = frozenset({"facebook"})
_EXPERIMENT_KEYS = frozenset({"name", "platforms", "start", "min_n", "stop_after_days", "arms"})
_ARM_KEYS = frozenset({"hashtag_mode", "weight"})


@dataclass(frozen=True)
class Arm:
    id: str
    hashtag_mode: str
    weight: int = 1


@dataclass(frozen=True)
class Experiment:
    name: str
    platforms: Tuple[str, ...]
    arms: Tuple[Arm, ...]
    start: date
    min_n: int
    stop_after_days: int


@dataclass(frozen=True)
class ArmResult:
    experiment: str
    platform: str
    arm: str
    n: int
    #: ``None`` (unknown) until every arm of the experiment holds ``min_n``.
    median_plays: Optional[float]


def _bad(index: int, key: str, why: str) -> ValueError:
    return ValueError(f"policy.experiments[{index}].{key} {why}")


def _positive_int(raw: Mapping, index: int, key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _bad(index, key, f"must be an int >= 1, got {value!r}")
    return value


def _parse_arms(raw: Mapping, index: int) -> Tuple[Arm, ...]:
    block = raw.get("arms")
    if not isinstance(block, dict) or len(block) < 2:
        raise _bad(index, "arms", f"must be a mapping of at least two arms, got {block!r}")
    arms = []
    for arm_id, spec in block.items():
        if not isinstance(arm_id, str) or not arm_id:
            raise _bad(index, "arms", f"ids must be non-empty strings, got {arm_id!r}")
        if not isinstance(spec, dict) or set(spec) - _ARM_KEYS:
            raise _bad(index, f"arms.{arm_id}", f"must be a mapping of {sorted(_ARM_KEYS)}")
        mode = spec.get("hashtag_mode")
        if mode not in HASHTAG_MODES:
            raise _bad(
                index,
                f"arms.{arm_id}.hashtag_mode",
                f"must be one of {sorted(HASHTAG_MODES)}, got {mode!r}",
            )
        weight = spec.get("weight", 1)
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise _bad(index, f"arms.{arm_id}.weight", f"must be an int >= 1, got {weight!r}")
        arms.append(Arm(arm_id, mode, weight))
    return tuple(arms)


def parse_experiments(block: Any) -> Tuple[Experiment, ...]:
    """Validate the raw ``policy.experiments`` list; raise ``ValueError`` naming the key."""
    if block is None:
        return ()
    if not isinstance(block, (list, tuple)):
        raise ValueError(f"policy.experiments must be a list (or omitted), got {block!r}")
    experiments: List[Experiment] = []
    for index, raw in enumerate(block):
        if not isinstance(raw, dict):
            raise ValueError(f"policy.experiments[{index}] must be a mapping, got {raw!r}")
        unknown = set(raw) - _EXPERIMENT_KEYS
        if unknown:
            raise _bad(index, sorted(unknown)[0], "is not a recognized experiment key")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _bad(index, "name", f"must be a non-empty string, got {name!r}")
        if name in SHORTS_FACTORS:
            raise _bad(index, "name", f"{name!r} is a Shorts factor name, not a caption experiment")
        if any(e.name == name for e in experiments):
            raise _bad(index, "name", f"duplicate experiment name {name!r}")
        platforms = raw.get("platforms")
        if (
            not isinstance(platforms, (list, tuple))
            or not platforms
            or not set(platforms) <= EXPERIMENT_PLATFORMS
        ):
            raise _bad(
                index, "platforms", f"must be a non-empty subset of {sorted(EXPERIMENT_PLATFORMS)}"
            )
        start = raw.get("start")
        if isinstance(start, str):
            try:
                start = date.fromisoformat(start)
            except ValueError:
                pass
        if not isinstance(start, date):
            raise _bad(index, "start", f"must be an ISO date, got {raw.get('start')!r}")
        experiments.append(
            Experiment(
                name=name,
                platforms=tuple(platforms),
                arms=_parse_arms(raw, index),
                start=start,
                min_n=_positive_int(raw, index, "min_n"),
                stop_after_days=_positive_int(raw, index, "stop_after_days"),
            )
        )
    return tuple(experiments)


def is_active(experiment: Experiment, today: date) -> bool:
    """True from ``start`` through ``start + stop_after_days - 1``."""
    return experiment.start <= today < experiment.start + timedelta(days=experiment.stop_after_days)


def assign_arm(experiment: Experiment, project_name: str, short_index: int) -> str:
    """The short's arm: a weighted, stable hash -- the same inputs always agree."""
    digest = hashlib.sha256(f"{experiment.name}:{project_name}:{short_index}".encode()).digest()
    slot = int.from_bytes(digest[:8], "big") % sum(a.weight for a in experiment.arms)
    for arm in experiment.arms:
        if slot < arm.weight:
            return arm.id
        slot -= arm.weight
    raise AssertionError("unreachable: slot is below the total weight")


def assign_arms(
    experiments: Iterable[Experiment], project_name: str, short_index: int, today: date
) -> dict:
    """``{experiment name: arm id}`` for every experiment active on ``today``."""
    return {
        e.name: assign_arm(e, project_name, short_index) for e in experiments if is_active(e, today)
    }


def hashtag_mode_for(
    experiments: Iterable[Experiment], arms: Mapping[str, str], platform: str
) -> Optional[str]:
    """The hashtag mode the short's arm sets on ``platform``, or ``None`` (legacy hashtags)."""
    for experiment in experiments:
        arm_id = arms.get(experiment.name)
        if arm_id is None or platform not in experiment.platforms:
            continue
        return next(a.hashtag_mode for a in experiment.arms if a.id == arm_id)
    return None


def arm_results(
    experiment: Experiment,
    observations: Iterable[Tuple[str, Optional[str], Optional[int]]],
    today: date,
) -> List[ArmResult]:
    """Per-arm n and median initial plays from ``(arm, posted_at, plays)`` observations.

    A row counts toward ``n`` only with measured plays and a settled day-7
    window. Medians stay unknown until EVERY arm holds ``min_n``, so no
    half-filled comparison is ever reported.
    """
    # Deferred: ``tracking`` imports the pipeline, which loads brand_policy.
    from ..tracking.day7 import posted_date, window_settled

    plays_by_arm = {a.id: [] for a in experiment.arms}
    for arm, posted_at, plays in observations:
        posted = posted_date(posted_at)
        if arm in plays_by_arm and plays is not None and posted and window_settled(posted, today):
            plays_by_arm[arm].append(plays)
    ready = all(len(v) >= experiment.min_n for v in plays_by_arm.values())
    return [
        ArmResult(
            experiment.name,
            experiment.platforms[0],
            arm.id,
            len(plays_by_arm[arm.id]),
            statistics.median(plays_by_arm[arm.id]) if ready else None,
        )
        for arm in experiment.arms
    ]


def arm_observations(
    experiment: Experiment,
    records: Iterable[Any],
    arms_for_project,
) -> List[Tuple[str, Optional[str], Optional[int]]]:
    """``(arm, posted_at, plays)`` for every tracked short tagged into ``experiment``.

    ``arms_for_project(project)`` returns ``{short_index: {experiment: arm}}``
    (the sidecar's tags) or ``None`` when unreadable; plays are the short's
    Facebook ``PlatformMetrics.reel_initial_plays`` (``None`` when never measured).
    """
    rows = []
    for record in records:
        tagged = arms_for_project(record.project) or {}
        for short in record.shorts:
            arm = (tagged.get(short.index) or {}).get(experiment.name)
            if arm is None:
                continue
            metrics = short.metrics.get(experiment.platforms[0])
            rows.append((arm, short.posted_at, getattr(metrics, "reel_initial_plays", None)))
    return rows


def arm_report_lines(results: Sequence[ArmResult]) -> List[str]:
    """One ``EXPERIMENT ...`` line per arm for the nightly output."""
    lines = []
    for r in results:
        median = "unknown" if r.median_plays is None else f"{r.median_plays:g}"
        lines.append(
            f"EXPERIMENT {r.experiment} platform={r.platform} arm={r.arm} n={r.n} "
            f"median_plays={median}"
        )
    return lines


# --- YouTube Shorts arms: title body and hashtag set -------------------------

#: The two factors a Short's sidecar ``experiment_arms`` may tag with an arm id.
SHORTS_FACTORS = ("yt_title_body", "yt_hashtag_set")

#: ``yt_title_body`` arms: arm id -> the stamped-title body templates the arm may
#: render (``"{stamp}: {body}"``). Every entry is one of the generator's own
#: ``STAMPED_BODY_TEMPLATES`` (a test holds that), so each already passes the
#: title review and anti-spam gates; the arm only narrows which bodies a Short
#: may take. ``place_led`` names the place (optionally with its weekday);
#: ``clip_led`` leads with the clip's length.
SHORTS_TITLE_BODY_ARMS: Dict[str, Tuple[str, ...]] = {
    "place_led": ("{location}", "{location}, on a {weekday}"),
    "clip_led": ("{secs} seconds, {location}",),
}

#: ``yt_hashtag_set`` arms: arm id -> the optional hashtags a non-interactive
#: Short's YouTube description carries after ``#Shorts`` (the terrain hashtag,
#: place and region tags follow as before). ``pov`` is the legacy set. Every
#: entry is in the generator's ``GENERIC_HASHTAGS``, which the title review backs.
SHORTS_HASHTAG_SET_ARMS: Dict[str, Tuple[str, ...]] = {
    "pov": ("#POVDriving",),
    "road": ("#RoadTrip", "#OpenRoad"),
}

#: One source of truth: factor -> its arm ids, in a fixed order.
SHORTS_ARMS: Dict[str, Tuple[str, ...]] = {
    "yt_title_body": tuple(SHORTS_TITLE_BODY_ARMS),
    "yt_hashtag_set": tuple(SHORTS_HASHTAG_SET_ARMS),
}

#: The outcome is read at 72 h after posting, within 12 h either way
#: (the day-7 snapshot rule in ``tracking.evidence_export``, at a 72 h target).
OUTCOME_AGE = timedelta(hours=72)
OUTCOME_TOLERANCE = timedelta(hours=12)

#: Named defaults for :class:`ArmThresholds` (JP-approved conservative values,
#: decision D2: act only on a clear, repeated gap).
DEFAULT_MIN_N = 36  #: settled Shorts an arm needs before it can be judged
DEFAULT_MIN_GAP = 3.0  #: a reference arm's median ratio must be >= 3x this arm's
DEFAULT_PAUSE_GAP = 6.0  #: ...and >= 6x (with no better retention) to pause
PAUSE_RETEST_DAYS = 28  #: a paused arm is re-tested once every 28 days

HOLD = "hold"
DEMOTE = "demote"
PAUSE = "pause"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

#: An arm's gap must hold in each of this many chronological slices of its
#: Shorts (halves): one good or bad stretch is not a "repeated" gap.
REPEAT_SLICES = 2


@dataclass(frozen=True)
class ArmThresholds:
    """Tracked config for the arm decision; ``policy.arm_thresholds`` overrides."""

    min_n: int = DEFAULT_MIN_N
    min_gap: float = DEFAULT_MIN_GAP
    pause_gap: float = DEFAULT_PAUSE_GAP
    retest_days: int = PAUSE_RETEST_DAYS


_THRESHOLD_KEYS = frozenset({"min_n", "min_gap", "pause_gap", "retest_days"})


def parse_arm_thresholds(block: Any) -> ArmThresholds:
    """Validate ``policy.arm_thresholds`` (``None`` = the named defaults)."""
    if block is None:
        return ArmThresholds()
    if not isinstance(block, dict):
        raise ValueError(f"policy.arm_thresholds must be a mapping (or omitted), got {block!r}")
    unknown = set(block) - _THRESHOLD_KEYS
    if unknown:
        raise ValueError(f"policy.arm_thresholds.{sorted(unknown)[0]} is not a recognized key")
    values: Dict[str, Any] = {}
    for key in ("min_n", "retest_days"):
        if key in block:
            v = block[key]
            floor = 2 * REPEAT_SLICES if key == "min_n" else 1
            if isinstance(v, bool) or not isinstance(v, int) or v < floor:
                raise ValueError(
                    f"policy.arm_thresholds.{key} must be an int >= {floor}, got {v!r}"
                )
            values[key] = v
    for key in ("min_gap", "pause_gap"):
        if key in block:
            v = block[key]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 1
            ):
                raise ValueError(
                    f"policy.arm_thresholds.{key} must be a finite number > 1, got {v!r}"
                )
            values[key] = float(v)
    out = ArmThresholds(**values)
    if out.pause_gap < out.min_gap:
        raise ValueError("policy.arm_thresholds.pause_gap must be >= min_gap")
    return out


def assign_stratified_arm(
    factor: str,
    arm_ids: Sequence[str],
    stratum: Tuple[str, str],
    short_key: Any,
    assigned: Sequence[str],
) -> str:
    """The arm for a new Short in ``stratum`` (``(project, slot)``).

    ``assigned`` holds the arms already given to that stratum's Shorts. The
    new Short takes an arm with the fewest so far, so arm counts never differ
    by more than one within the stratum (the ordinal confound: an arm no
    longer lands on the early or on the strong slot by luck). Ties break on a
    stable sha256 of ``factor:project:slot:short_key:arm`` -- never random.
    """
    counts = Counter(assigned)
    fewest = min(counts.get(a, 0) for a in arm_ids)
    tied = [a for a in arm_ids if counts.get(a, 0) == fewest]

    def rank(arm: str) -> bytes:
        key = f"{factor}:{stratum[0]}:{stratum[1]}:{short_key}:{arm}"
        return hashlib.sha256(key.encode()).digest()

    return min(tied, key=rank)


def assign_shorts_arms(
    stratum: Tuple[str, str],
    short_key: Any,
    assigned: Dict[Tuple[str, str, str], List[str]],
    factors: Iterable[str],
) -> Dict[str, str]:
    """``{factor: arm}`` for one new Short, each through :func:`assign_stratified_arm`.

    ``assigned`` is the caller's running record, ``(factor, project, slot) ->
    arms given so far``; it is updated in place, so calling this for a batch in
    a fixed order gives the same arms every time. Only ``factors`` the Short's
    text can actually carry are assigned (an arm that changes nothing would only
    blur the outcome).
    """
    out: Dict[str, str] = {}
    for factor in factors:
        history = assigned.setdefault((factor, stratum[0], stratum[1]), [])
        arm = assign_stratified_arm(factor, SHORTS_ARMS[factor], stratum, short_key, history)
        history.append(arm)
        out[factor] = arm
    return out


def _utc(ts: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts)) if ts else None
    except ValueError:
        return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) if dt else None


def youtube_outcome(
    short: Any, youtube_at: Optional[str], today: date
) -> Optional[Tuple[int, Optional[float]]]:
    """``(views_72h, avg_view_percentage)`` for a settled YouTube Short, or ``None``.

    ``youtube_at`` is YouTube's OWN publish time (the delivery queue's
    ``published_at``), never ``short.posted_at``: that field holds whichever
    platform registered last, hours after YouTube for Facebook/Instagram.
    No YouTube time means no outcome. Read from the measured observation of
    the Short's current YouTube id (``short.platform_ids["youtube"]``) nearest
    72 h after that time (within 12 h) in ``short.observation_log["youtube"]``.
    Unsettled means ``published + 72 h > today - SETTLE_LAG_DAYS`` (the
    Analytics lag in ``tracking.day7``). Unreported retention is ``None``,
    never 0; a snapshot with no views is not an outcome.
    """
    from ..tracking.day7 import SETTLE_LAG_DAYS, posted_date

    published, video_id = (
        _utc(youtube_at),
        (getattr(short, "platform_ids", None) or {}).get("youtube"),
    )
    day = posted_date(youtube_at)
    if published is None or day is None or not video_id:
        return None
    if day + OUTCOME_AGE > today - timedelta(days=SETTLE_LAG_DAYS):
        return None
    best = None
    for entry in (getattr(short, "observation_log", None) or {}).get("youtube") or []:
        if entry.get("status") != "measured" or entry.get("post_id") != video_id:
            continue
        cumulative = entry.get("cumulative") or {}
        views, fetched = cumulative.get("views"), _utc(entry.get("fetched_at"))
        if views is None or fetched is None:
            continue
        miss = abs(fetched - published - OUTCOME_AGE)
        if miss <= OUTCOME_TOLERANCE and (best is None or miss < best[0]):
            best = (miss, (int(views), cumulative.get("avg_view_percentage")))
    return best[1] if best else None


@dataclass(frozen=True)
class ShortOutcome:
    project: str
    arm: str
    posted: date
    views_72h: int
    avg_view_pct: Optional[float]


def tagged_arms(factor: str, records: Iterable[Any], arms_for_project) -> List[str]:
    """Every arm id of ``factor`` that any tracked Short is tagged with."""
    seen = set()
    for record in records:
        tagged = arms_for_project(record.project) or {}
        for short in record.shorts:
            arm = (tagged.get(short.index) or {}).get(factor)
            if arm:
                seen.add(arm)
    return sorted(seen)


def shorts_arm_outcomes(
    factor: str,
    records: Iterable[Any],
    arms_for_project,
    youtube_times: Mapping[Tuple[str, int], str],
    today: date,
) -> List[ShortOutcome]:
    """One :class:`ShortOutcome` per tracked Short tagged into ``factor`` with a
    settled 72 h outcome. ``arms_for_project`` is as for :func:`arm_observations`;
    ``youtube_times`` maps ``(project, short_index)`` to YouTube's publish time."""
    from ..tracking.day7 import posted_date

    rows = []
    for record in records:
        tagged = arms_for_project(record.project) or {}
        for short in record.shorts:
            arm = (tagged.get(short.index) or {}).get(factor)
            youtube_at = youtube_times.get((record.project, short.index))
            outcome = youtube_outcome(short, youtube_at, today) if arm else None
            if outcome is not None:
                rows.append(
                    ShortOutcome(
                        record.project, arm, posted_date(youtube_at), outcome[0], outcome[1]
                    )
                )
    return rows


@dataclass(frozen=True)
class ArmDecision:
    factor: str
    arm: str
    decision: str
    n: int
    #: Median of the arm's views / its project's median (``None`` = unknown).
    score: Optional[float]
    #: Median score of the other arms that hold ``min_n`` (``None`` = unknown).
    reference: Optional[float]
    #: ``reference / score`` (``inf`` when the arm's median is 0).
    gap: Optional[float]
    reason: str


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _gap(reference: float, score: float) -> float:
    """``reference / score``: ``inf`` only for a dead arm beside a live
    reference; a reference of 0 or less is no gap at all."""
    if reference <= 0:
        return 0.0
    return math.inf if score <= 0 else reference / score


def decide_shorts_arms(
    factor: str,
    outcomes: Sequence[ShortOutcome],
    thresholds: ArmThresholds,
    seen_arms: Sequence[str] = (),
) -> List[ArmDecision]:
    """One decision per arm, from settled outcomes and ``thresholds`` alone.

    Each Short's views are divided by the median views of its project's
    Shorts (a project counts only if it holds two or more arms and a non-zero
    median), so an arm is never judged by the strength of the projects it
    happened to land in. An arm's score is the median of those ratios; its
    reference is the median score of the OTHER arms holding ``min_n``
    (``caption_ranking``'s pool-median-reference policy, with a ratio gap
    instead of a lift band). An arm below ``min_n`` or with no reference is
    ``INSUFFICIENT_EVIDENCE``. Otherwise a gap of at least ``min_gap`` that
    holds in each chronological half of the arm's Shorts is ``demote``
    (``pause`` at ``pause_gap`` when its median retention is also known and no
    better than the others'); everything else, winners included, is ``hold``.
    An arm in ``seen_arms`` (tagged on a Short) with no countable Short is
    ``INSUFFICIENT_EVIDENCE`` at ``n=0``.
    """
    by_project: Dict[str, List[ShortOutcome]] = {}
    for o in outcomes:
        by_project.setdefault(o.project, []).append(o)
    ratios: Dict[str, List[Tuple[date, str, float]]] = {}
    pcts: Dict[str, List[float]] = {}
    for project, rows in sorted(by_project.items()):
        if len({r.arm for r in rows}) < 2:
            continue
        base = statistics.median(r.views_72h for r in rows)
        if base <= 0:
            continue
        for r in rows:
            ratios.setdefault(r.arm, []).append((r.posted, project, r.views_72h / base))
            if r.avg_view_pct is not None:
                pcts.setdefault(r.arm, []).append(r.avg_view_pct)
    if not ratios and not seen_arms:
        return []
    scores = {a: _median_or_none([x[2] for x in v]) for a, v in ratios.items()}
    judged = {a for a, v in ratios.items() if len(v) >= thresholds.min_n}
    out = []
    for arm in sorted(set(ratios) | set(seen_arms)):
        if arm not in ratios:
            out.append(
                ArmDecision(factor, arm, INSUFFICIENT_EVIDENCE, 0, None, None, None, "no_countable")
            )
            continue
        n, score = len(ratios[arm]), scores[arm]
        others = sorted(a for a in judged if a != arm)
        reference = _median_or_none([scores[a] for a in others])
        if n < thresholds.min_n:
            out.append(
                ArmDecision(factor, arm, INSUFFICIENT_EVIDENCE, n, score, None, None, "n_below_min")
            )
            continue
        if reference is None:
            out.append(
                ArmDecision(
                    factor, arm, INSUFFICIENT_EVIDENCE, n, score, None, None, "no_reference"
                )
            )
            continue
        gap = _gap(reference, score)
        if gap < thresholds.min_gap:
            out.append(ArmDecision(factor, arm, HOLD, n, score, reference, gap, "within_gap"))
            continue
        ordered = sorted(ratios[arm], key=lambda x: (x[0], x[1], x[2]))
        size = len(ordered) // REPEAT_SLICES
        slices = [
            ordered[i * size : (i + 1) * size if i < REPEAT_SLICES - 1 else None]
            for i in range(REPEAT_SLICES)
        ]
        repeated = all(
            _gap(reference, statistics.median(x[2] for x in chunk)) >= thresholds.min_gap
            for chunk in slices
        )
        if not repeated:
            out.append(ArmDecision(factor, arm, HOLD, n, score, reference, gap, "gap_not_repeated"))
            continue
        mine = _median_or_none(pcts.get(arm, []))
        theirs = _median_or_none([p for a in others for p in pcts.get(a, [])])
        no_better = mine is not None and theirs is not None and mine <= theirs
        if gap >= thresholds.pause_gap and no_better:
            out.append(
                ArmDecision(factor, arm, PAUSE, n, score, reference, gap, "clear_gap_repeated")
            )
        else:
            out.append(
                ArmDecision(factor, arm, DEMOTE, n, score, reference, gap, "clear_gap_repeated")
            )
    return out


def pause_retest_due(
    paused_on: date, today: date, thresholds: ArmThresholds = ArmThresholds()
) -> bool:
    """True once a paused arm's fixed re-test schedule has come round
    (``retest_days`` after ``paused_on``)."""
    return today >= paused_on + timedelta(days=thresholds.retest_days)


def _num(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return "inf" if math.isinf(value) else f"{value:.2f}"


def decision_lines(
    factor: str, decisions: Sequence[ArmDecision], thresholds: ArmThresholds = ArmThresholds()
) -> List[str]:
    """One ``ARM_DECISION ...`` line per arm for the nightly output."""
    if not decisions:
        return [
            f"ARM_DECISION factor={factor} arm=- decision={INSUFFICIENT_EVIDENCE} n=0 reason=no_outcomes"
        ]
    lines = []
    for d in decisions:
        line = (
            f"ARM_DECISION factor={d.factor} arm={d.arm} decision={d.decision} n={d.n} "
            f"min_n={thresholds.min_n} score={_num(d.score)} reference={_num(d.reference)} "
            f"gap={_num(d.gap)} reason={d.reason}"
        )
        if d.decision == PAUSE:
            line += f" retest_days={thresholds.retest_days}"
        lines.append(line)
    return lines
