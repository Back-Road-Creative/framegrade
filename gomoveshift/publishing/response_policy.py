"""Pure state -> bounded-response policy (shadow-first).

Per-platform daily-cap and auto-pause math for the automatic, bounded
response to platform suppression. Deliberately pure -- no DB, no clock --
so the ladder/streak math is testable without a queue fixture. Callers
(``dispatch.apply_suppression_response``) own persistence and the actual
pause/cap side effects.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Mapping, Optional, Sequence, Tuple

Status = Literal["HEALTHY", "SUPPRESSED", "RECOVERING"]

#: Fallback ladder when a brand has no ``policy.daily_ladder``
#: (``publishing.brand_policy.BrandPolicy.daily_ladder`` is ``None``).
DEFAULT_LADDER: Tuple[int, ...] = (1, 2, 3)


def daily_cap_for(state: Status, ladder: Optional[Sequence[int]], healthy_days: int) -> int:
    """The daily cap for a platform currently in ``state``.

    ``ladder`` is the brand's ``policy.daily_ladder`` (``None`` falls back
    to :data:`DEFAULT_LADDER`, matching ``brand_policy``'s own "missing =
    no override" contract). ``healthy_days`` is the count of consecutive
    non-SUPPRESSED days immediately before today (0 on the day suppression
    is confirmed lifted).

    ``SUPPRESSED`` always caps at 1/day regardless of ladder position or
    ``healthy_days`` -- the safety floor a brand cannot configure away.
    ``RECOVERING`` climbs the ladder one rung per 7 healthy days, starting
    at rung 0 (the floor) and never exceeding the brand's own ceiling
    (``ladder[-1]``). ``HEALTHY`` is uncapped by this policy -- the
    ladder's ceiling.
    """
    rungs = tuple(ladder) if ladder else DEFAULT_LADDER
    if state == "SUPPRESSED":
        return 1
    if state == "RECOVERING":
        step = min(len(rungs) - 1, healthy_days // 7)
        return rungs[step]
    return rungs[-1]


def should_auto_pause(states_by_day: Mapping[str, Status], today: str, days: int = 7) -> bool:
    """True iff ``today`` and the ``days - 1`` days before it are ALL
    recorded SUPPRESSED -- a consecutive run ending today, not merely
    ``days`` SUPPRESSED days somewhere in a longer window. A day missing
    from ``states_by_day`` breaks the streak, same as a HEALTHY/RECOVERING
    day would: an ungraded day is not evidence of suppression.
    """
    end = date.fromisoformat(today)
    for offset in range(days):
        day = (end - timedelta(days=offset)).isoformat()
        if states_by_day.get(day) != "SUPPRESSED":
            return False
    return True


def consecutive_healthy_days(states_by_day: Mapping[str, Status], today: str) -> int:
    """Count consecutive non-SUPPRESSED days immediately before ``today``
    (``today`` itself excluded -- it's the day whose cap is being decided,
    not a day already counted toward recovery). Stops at the first
    SUPPRESSED or missing day, same "no evidence, no credit" rule as
    :func:`should_auto_pause`.
    """
    count = 0
    day = date.fromisoformat(today) - timedelta(days=1)
    while True:
        state = states_by_day.get(day.isoformat())
        if state is None or state == "SUPPRESSED":
            break
        count += 1
        day -= timedelta(days=1)
    return count
