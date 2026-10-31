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

The outcome is Facebook Reel reach (``PlatformMetrics.reach``). Reach not yet
measured is unknown, never 0, and a short counts toward an arm's ``n`` only
once its day-7 window is settled (``tracking.day7.window_settled``).
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

HASHTAG_MODES = frozenset({"none", "location", "june"})
#: Facebook reach is the only outcome measured today, so only it can be tested.
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
    median_reach: Optional[float]


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
    """Per-arm n and median reach from ``(arm, posted_at, reach)`` observations.

    A row counts toward ``n`` only with a measured reach and a settled day-7
    window. Medians stay unknown until EVERY arm holds ``min_n``, so no
    half-filled comparison is ever reported.
    """
    # Deferred: ``tracking`` imports the pipeline, which loads brand_policy.
    from ..tracking.day7 import posted_date, window_settled

    reach_by_arm = {a.id: [] for a in experiment.arms}
    for arm, posted_at, reach in observations:
        posted = posted_date(posted_at)
        if arm in reach_by_arm and reach is not None and posted and window_settled(posted, today):
            reach_by_arm[arm].append(reach)
    ready = all(len(v) >= experiment.min_n for v in reach_by_arm.values())
    return [
        ArmResult(
            experiment.name,
            experiment.platforms[0],
            arm.id,
            len(reach_by_arm[arm.id]),
            statistics.median(reach_by_arm[arm.id]) if ready else None,
        )
        for arm in experiment.arms
    ]


def arm_observations(
    experiment: Experiment,
    records: Iterable[Any],
    arms_for_project,
) -> List[Tuple[str, Optional[str], Optional[int]]]:
    """``(arm, posted_at, reach)`` for every tracked short tagged into ``experiment``.

    ``arms_for_project(project)`` returns ``{short_index: {experiment: arm}}``
    (the sidecar's tags) or ``None`` when unreadable; reach is the short's
    Facebook ``PlatformMetrics.reach`` (``None`` when never measured).
    """
    rows = []
    for record in records:
        tagged = arms_for_project(record.project) or {}
        for short in record.shorts:
            arm = (tagged.get(short.index) or {}).get(experiment.name)
            if arm is None:
                continue
            metrics = short.metrics.get(experiment.platforms[0])
            rows.append((arm, short.posted_at, getattr(metrics, "reach", None)))
    return rows


def arm_report_lines(results: Sequence[ArmResult]) -> List[str]:
    """One ``EXPERIMENT ...`` line per arm for the nightly output."""
    lines = []
    for r in results:
        median = "unknown" if r.median_reach is None else f"{r.median_reach:g}"
        lines.append(
            f"EXPERIMENT {r.experiment} platform={r.platform} arm={r.arm} n={r.n} "
            f"median_reach={median}"
        )
    return lines
