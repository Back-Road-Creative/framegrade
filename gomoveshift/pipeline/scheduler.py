"""
Pipeline Scheduler — calculates shorts posting windows from publish calendar.

Generates a release schedule that slots shorts into the gaps between
long form publish dates, using the channel's posting cadence.

Publishing rules:
  - Long form: Wednesday at 11:00 AM EST
  - Shorts: 12:00 PM, 4:00 PM daily
  - Shorts start AFTER their long form publishes, stop BEFORE next long form

Usage:
    from gomoveshift.pipeline.scheduler import PipelineScheduler

    scheduler = PipelineScheduler()
    schedule = scheduler.generate_schedule(
        publish_date="2024-01-03",  # the Wednesday long form goes live
        publish_day="wed",
        num_shorts=9,
    )
    schedule.save("/path/to/release_schedule.yaml")
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from ._atomic_write import atomic_write_yaml
from .schedule_constants import (
    EST,
    LF_PUBLISH_INTERVAL_DAYS,
    LONG_FORM_DAYS,
    LONG_FORM_TIME,
    SHORTS_TIMES,
)

logger = logging.getLogger(__name__)

# Day names
DAY_NAMES = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
DAY_LABELS = {v: k.capitalize() for k, v in DAY_NAMES.items()}


def _spread_bucket_across_window(eligible_counts: List[int], bucket: int) -> List[int]:
    """Distribute ``bucket`` slots across a window's dates evenly, honoring
    each date's eligible-slot cap.

    Round-robins one slot per date (in date order) across successive
    passes, so no date receives a second slot before every other date
    (with remaining capacity) has received a first. Front-loading — filling
    the earliest days to capacity before touching later ones — is exactly
    what this avoids (it left trailing window days completely empty).

    Returns a list parallel to ``eligible_counts``; each entry never exceeds
    its cap, and the sum is exactly ``min(bucket, sum(eligible_counts))``.
    """
    allowance = [0] * len(eligible_counts)
    remaining = min(bucket, sum(eligible_counts))
    while remaining > 0:
        placed_this_pass = False
        for i, cap in enumerate(eligible_counts):
            if remaining <= 0:
                break
            if allowance[i] < cap:
                allowance[i] += 1
                remaining -= 1
                placed_this_pass = True
        if not placed_this_pass:
            break
    return allowance


def _to_utc(iso_dt: str) -> datetime:
    """Parse an ISO-8601 datetime string (with TZ suffix) into a UTC-aware
    ``datetime``. Used for cross-DST slot/LF window comparison.

    ISO-string lexicographic comparison breaks across DST transitions
    because two instants with the same wall-clock representation can
    carry different offsets (``-04:00`` vs ``-05:00``) — string order
    no longer matches UTC instant order. Always compare in UTC.
    """
    return datetime.fromisoformat(iso_dt).astimezone(timezone.utc)


@dataclass
class ScheduledSlot:
    """A single scheduled posting slot."""

    short_index: int
    date: str  # YYYY-MM-DD
    time: str  # HH:MM:SS
    datetime_est: str  # Full ISO datetime
    day_of_week: str

    def to_dict(self) -> dict:
        return {
            "short_index": self.short_index,
            "date": self.date,
            "time": self.time,
            "datetime_est": self.datetime_est,
            "day_of_week": self.day_of_week,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledSlot":
        return cls(
            short_index=d["short_index"],
            date=d["date"],
            time=d["time"],
            datetime_est=d["datetime_est"],
            day_of_week=d["day_of_week"],
        )


@dataclass
class ReleaseSchedule:
    """Complete release schedule for a project's shorts."""

    project_name: str
    location: str
    long_form_publish: str  # ISO datetime of long form publish
    window_start: str  # ISO datetime of first shorts slot
    window_end: str  # ISO datetime of last shorts slot
    num_shorts: int
    num_slots: int
    slots: List[ScheduledSlot]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "location": self.location,
            "long_form_publish": self.long_form_publish,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "num_shorts": self.num_shorts,
            "num_slots": self.num_slots,
            "notes": self.notes,
            "schedule": [s.to_dict() for s in self.slots],
        }

    def save(self, path: Path) -> None:
        """Save schedule to YAML for human review."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        # Add review instructions at top
        data["_review_instructions"] = (
            "Review this schedule and edit dates/times if needed. "
            "The CSV generator will use these values. "
            "Delete any slots you don't want to publish."
        )

        # Atomic write (temp file + os.replace) — a crash/full-disk mid-write
        # must never leave a truncated release_schedule.yaml. A truncated
        # file often still parses, so the corruption would otherwise be
        # silent (it just has fewer slots than it should).
        atomic_write_yaml(path, data, default_flow_style=False, sort_keys=False)

        logger.info(f"Schedule saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ReleaseSchedule":
        """Load schedule from YAML (possibly user-edited)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Schedule not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Remove review instructions if present
        data.pop("_review_instructions", None)

        slots = [ScheduledSlot(**slot_data) for slot_data in data.get("schedule", [])]

        return cls(
            project_name=data.get("project_name", ""),
            location=data.get("location", ""),
            long_form_publish=data.get("long_form_publish", ""),
            window_start=data.get("window_start", ""),
            window_end=data.get("window_end", ""),
            num_shorts=data.get("num_shorts", len(slots)),
            num_slots=data.get("num_slots", len(slots)),
            slots=slots,
            notes=data.get("notes", ""),
        )


class PipelineScheduler:
    """
    Calculates shorts posting windows based on the GoMoveShift calendar.

    Rules:
      - Long form publishes on Wed at 11am EST
      - Shorts publish at 12pm and 4pm daily
      - Shorts window starts after the associated long form publishes
      - Shorts window ends before the NEXT long form publishes
      - This gives 2 shorts per day across the window (14 per LF weekly)
    """

    def __init__(
        self,
        long_form_time: dt_time | None = None,
        shorts_times: Optional[List[dt_time]] = None,
        long_form_days: Optional[set] = None,
    ):
        """
        Args:
            long_form_time: Time of day long form publishes (EST)
            shorts_times: Times of day shorts publish (EST)
            long_form_days: Days of week for long form (0=Mon, 6=Sun)
        """
        self.long_form_time = long_form_time if long_form_time is not None else LONG_FORM_TIME
        self.shorts_times = shorts_times or SHORTS_TIMES
        self.long_form_days = frozenset(long_form_days) if long_form_days else LONG_FORM_DAYS

    @classmethod
    def from_config(cls, config) -> "PipelineScheduler":
        """Build a ``PipelineScheduler`` using cadence from ``config.drive``.

        Takes the long-form days and time from Waypost (the run's
        ``--lf-day`` / ``--lf-time`` transport wins); falls back to the module-level
        defaults for ``shorts_times``. See ``schedule_constants`` for
        the accessor contract.
        """
        from .schedule_constants import long_form_days as _ld
        from .schedule_constants import long_form_time as _lt
        from .schedule_constants import shorts_times as _st

        return cls(
            long_form_time=_lt(config),
            shorts_times=_st(config),
            long_form_days=_ld(config),
        )

    def _get_publish_datetime(self, date_str: str, publish_day: str) -> datetime:
        """
        Resolve the long form publish datetime.

        Args:
            date_str: Date string YYYY-MM-DD or "next_wed" / "next_sat"
            publish_day: Day of week (e.g., "wed")

        Returns:
            datetime in EST
        """
        if date_str.startswith("next_"):
            # The next occurrence of publish_day is seo_manager's search (never today).
            target_dow = DAY_NAMES.get(publish_day)
            if target_dow is None:
                raise ValueError(f"Invalid publish day: {publish_day}")
            from .seo_slots import next_cadence_day

            return next_cadence_day({target_dow}, self.long_form_time, datetime.now(EST))
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return datetime.combine(target_date, self.long_form_time, tzinfo=EST)

    def _get_next_long_form_publish(self, after: datetime) -> datetime:
        """The next long form publish datetime after a given datetime (seo_manager's search)."""
        from .seo_slots import next_cadence_day

        return next_cadence_day(self.long_form_days, self.long_form_time, after)

    def _calculate_window(
        self,
        long_form_dt: datetime,
    ) -> Tuple[datetime, datetime]:
        """
        Calculate the shorts posting window.

        Window starts: SAME DAY as long form, at first shorts slot AFTER
        the long form publish time (e.g. LF at 11am → first short at 12pm).
        If no same-day slots remain, starts next day at 8am.

        Window ends: morning of next long form day, last slot before LF time.

        Returns:
            (window_start, window_end) as datetimes
        """
        publish_date = long_form_dt.date()
        available_times = [t for t in self.shorts_times if t > self.long_form_time]

        if available_times:
            window_start = datetime.combine(publish_date, available_times[0], tzinfo=EST)
        else:
            # No slots left today (e.g. LF at 4pm) — start next day
            window_start = datetime.combine(
                publish_date + timedelta(days=1),
                self.shorts_times[0],
                tzinfo=EST,
            )

        # Window ends before next long form
        next_lf = self._get_next_long_form_publish(long_form_dt)
        last_day = next_lf.date()
        early_slots = [t for t in self.shorts_times if t < self.long_form_time]

        if early_slots:
            # Can use morning slots on the day of next long form
            window_end = datetime.combine(last_day, early_slots[-1], tzinfo=EST)
        else:
            # Use previous day's last slot
            window_end = datetime.combine(
                last_day - timedelta(days=1),
                self.shorts_times[-1],
                tzinfo=EST,
            )

        return window_start, window_end

    def _generate_slots(
        self,
        window_start: datetime,
        window_end: datetime,
        num_shorts: int,
    ) -> List[ScheduledSlot]:
        """
        Generate posting slots within the window.

        Creates 3 slots per day (8am, 12pm, 4pm) and assigns shorts
        sequentially. Always reserves the last slot in the window
        (morning before next long form) to bridge the content gap.
        """
        # First, collect ALL available slots in the window
        all_slots = []
        current_date = window_start.date()
        end_date = window_end.date()

        while current_date <= end_date:
            for slot_time in self.shorts_times:
                slot_dt = datetime.combine(current_date, slot_time, tzinfo=EST)
                if window_start <= slot_dt <= window_end:
                    all_slots.append(slot_dt)
            current_date += timedelta(days=1)

        if not all_slots or num_shorts <= 0:
            return []

        # Always reserve the last slot (morning before next long form)
        # This ensures content bridges to the next long form publish
        last_slot_dt = all_slots[-1]
        remaining_slots = all_slots[:-1]

        if num_shorts == 1:
            # Only one short — put it in the last slot
            selected = [last_slot_dt]
        elif num_shorts >= len(all_slots):
            # More shorts than slots — fill all slots, cap at window size
            selected = all_slots
        else:
            # Fill from the start, but always include the last slot
            selected = remaining_slots[: num_shorts - 1] + [last_slot_dt]

        # Build ScheduledSlot objects
        slots = []
        for idx, slot_dt in enumerate(selected, 1):
            dow_name = slot_dt.strftime("%a")
            slots.append(
                ScheduledSlot(
                    short_index=idx,
                    date=slot_dt.strftime("%Y-%m-%d"),
                    time=slot_dt.strftime("%H:%M:%S"),
                    datetime_est=slot_dt.isoformat(),
                    day_of_week=dow_name,
                )
            )

        return slots

    def generate_schedule(
        self,
        publish_date: str,
        publish_day: str,
        num_shorts: int,
        project_name: str = "",
        location: str = "",
    ) -> ReleaseSchedule:
        """
        Generate a complete release schedule for shorts.

        Args:
            publish_date: Long form publish date (YYYY-MM-DD or "next_wed")
            publish_day: Day of week (e.g., "wed", "sat")
            num_shorts: Number of shorts to schedule
            project_name: Project name for the schedule
            location: Location for reference

        Returns:
            ReleaseSchedule with all slots filled
        """
        publish_day = publish_day.lower()[:3]
        if publish_day not in DAY_NAMES:
            raise ValueError(
                f"Invalid publish day: '{publish_day}'. Valid: {', '.join(DAY_NAMES.keys())}"
            )

        # Resolve long form publish datetime
        lf_dt = self._get_publish_datetime(publish_date, publish_day)
        logger.info(f"Long form publishes: {lf_dt.strftime('%A %B %d at %I:%M %p EST')}")

        # Calculate posting window
        window_start, window_end = self._calculate_window(lf_dt)
        logger.info(
            f"Shorts window: {window_start.strftime('%a %m/%d %I%p')} → "
            f"{window_end.strftime('%a %m/%d %I%p')}"
        )

        # Calculate available slot count
        next_lf = self._get_next_long_form_publish(lf_dt)
        total_days = (window_end.date() - window_start.date()).days + 1
        logger.info(
            f"Window spans {total_days} day(s), next long form: {next_lf.strftime('%A %B %d')}"
        )

        # Generate slots (capped at window size — shorts must not bleed past next longform)
        slots = self._generate_slots(window_start, window_end, num_shorts)

        if len(slots) < num_shorts:
            logger.warning(
                f"Only {len(slots)} slots available in window for "
                f"{num_shorts} shorts. Some shorts won't be scheduled."
            )

        schedule = ReleaseSchedule(
            project_name=project_name,
            location=location,
            long_form_publish=lf_dt.isoformat(),
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            num_shorts=num_shorts,
            num_slots=len(slots),
            slots=slots,
            notes=(
                f"Long form on {lf_dt.strftime('%A %B %d at %I:%M %p')} EST. "
                f"Shorts fill {window_start.strftime('%a')}-{window_end.strftime('%a')} "
                f"window ({len(slots)} slots for {num_shorts} shorts)."
            ),
        )

        logger.info(f"Generated schedule: {len(slots)} slots for {num_shorts} shorts")
        for slot in slots:
            logger.debug(f"  Short #{slot.short_index}: {slot.day_of_week} {slot.date} {slot.time}")

        # Validate — hard fail if schedule violates invariants
        validation_errors = self.validate_schedule(schedule)
        if validation_errors:
            for err in validation_errors:
                logger.error(f"  SCHEDULE INVALID: {err}")
            raise ValueError(
                f"Generated schedule has {len(validation_errors)} validation error(s): "
                f"{validation_errors[0]}"
            )

        return schedule

    def validate_schedule(self, schedule: "ReleaseSchedule") -> List[str]:
        """
        Validate a schedule against hard invariants. Returns list of errors.

        Checks:
          1. All slots are strictly after the longform publish time
          2. All slots are strictly before the NEXT longform publish time
          3. No two slots share the same datetime
          4. Slots are in chronological order
          5. Publish date matches the declared publish day
        """
        errors = []
        if not schedule.slots:
            return errors

        lf_dt = datetime.fromisoformat(schedule.long_form_publish)
        next_lf = self._get_next_long_form_publish(lf_dt)

        # Check publish date matches day of week
        from .seo_slots import anchor_errors

        if anchor_errors([lf_dt.strftime("%Y-%m-%d")], self.long_form_days):
            errors.append(
                f"Longform date {lf_dt.date()} is {lf_dt.strftime('%a')} "
                f"but longform days are {[DAY_LABELS[d] for d in sorted(self.long_form_days)]}"
            )

        prev_dt = None
        seen = set()
        for slot in schedule.slots:
            slot_dt = datetime.fromisoformat(slot.datetime_est)

            # Must be after longform
            if slot_dt <= lf_dt:
                errors.append(
                    f"Short #{slot.short_index} at {slot.datetime_est} "
                    f"is not after longform at {lf_dt.isoformat()}"
                )

            # Must be before next longform
            if slot_dt >= next_lf:
                errors.append(
                    f"Short #{slot.short_index} at {slot.datetime_est} "
                    f"bleeds past next longform at {next_lf.isoformat()}"
                )

            # Valid slot times only — derive the valid set from SHORTS_TIMES
            # so this validator stays in lockstep with the constants.
            valid_slot_strs = {t.strftime("%H:%M:%S") for t in self.shorts_times}
            if slot.time not in valid_slot_strs:
                errors.append(
                    f"Short #{slot.short_index} time {slot.time} "
                    f"is not a valid slot time ({sorted(valid_slot_strs)})"
                )

            # No duplicate datetimes
            if slot.datetime_est in seen:
                errors.append(f"Duplicate slot datetime: {slot.datetime_est}")
            seen.add(slot.datetime_est)

            # Chronological order
            if prev_dt and slot_dt <= prev_dt:
                errors.append(
                    f"Short #{slot.short_index} at {slot.datetime_est} is not after previous slot"
                )
            prev_dt = slot_dt

        return errors

    def get_available_slot_count(
        self,
        publish_date: str,
        publish_day: str,
    ) -> int:
        """
        Calculate how many shorts slots are available in a window.

        Useful for knowing how many shorts to extract.
        """
        lf_dt = self._get_publish_datetime(publish_date, publish_day)
        window_start, window_end = self._calculate_window(lf_dt)

        # Count all valid slots in window
        count = 0
        current_date = window_start.date()
        end_date = window_end.date()

        while current_date <= end_date:
            for slot_time in self.shorts_times:
                slot_dt = datetime.combine(current_date, slot_time, tzinfo=EST)
                if window_start <= slot_dt <= window_end:
                    count += 1
            current_date += timedelta(days=1)

        return count

    # ── Drive Pipeline Scheduling ────────────────────────────────────

    @staticmethod
    def _select_run_identity_times(base_date, n_lfs, defaults, pool, per_day, taken):
        """Pick the run-wide consistent posting times (see ``generate_drive_schedule``).

        Orders candidates defaults-first then ``pool``, keeps each whose share of
        booked days across the run window is below ``CONTESTED`` (so a slot another
        drive owns for most of the run is dropped run-wide, not per-day), and fills
        any shortfall with the least-contested remaining candidates. Returns
        ``per_day`` ``datetime.time``s. Empty ``taken`` ⇒ the first ``per_day``
        defaults, i.e. legacy behavior.
        """
        contested = 0.5
        window = [
            base_date + timedelta(days=k)
            for k in range(max(n_lfs, 1) * LF_PUBLISH_INTERVAL_DAYS + 1)
        ]
        span = len(window) or 1

        def contested_share(t: dt_time) -> float:
            ts = t.strftime("%H:%M:%S")
            booked = sum(1 for d in window if (d.strftime("%Y-%m-%d"), ts) in taken)
            return booked / span

        ordered = list(defaults) + [t for t in pool if t not in defaults]
        identity: List[dt_time] = []
        for t in ordered:
            if len(identity) >= per_day:
                break
            if t not in identity and contested_share(t) < contested:
                identity.append(t)
        if len(identity) < per_day:
            for t in sorted((x for x in ordered if x not in identity), key=contested_share):
                if len(identity) >= per_day:
                    break
                identity.append(t)
        return identity

    def generate_drive_schedule(
        self,
        regular_lfs: list,
        region: str,
        start_date: Optional[str] = None,
        shorts_per_lf: Optional[dict] = None,
        publish_day: Optional[str] = None,
        occupied_slots: Optional[set] = None,
        slot_pool: Optional[List[dt_time]] = None,
        shorts_per_day: Optional[int] = None,
        shorts_counts: Optional[List[int]] = None,
        locked_identity_times: Optional[List[dt_time]] = None,
        lf_dates: Optional[List[date]] = None,
    ) -> "DriveReleaseSchedule":
        """Generate a schedule for a drive pipeline output set.

        Rules:
          - Publish day resolved in priority order: explicit ``publish_day``
            arg (from CLI / state) > regional default. Regional defaults:
            Ireland → Wed, USA → Wed (was Sat; the Wed-only cadence
            applies to both).
          - Shorts fill the configured slots (``SHORTS_TIMES``) between
            LF publishes.
          - When shorts_per_lf is provided, best-ranked shorts get
            earliest slots.

        Args:
            regular_lfs: List of RegularLFPlan objects.
            region: Content region ("ireland" or "usa").
            start_date: First LF publish date (YYYY-MM-DD).
                        If None, uses next ``publish_day`` of week.
            shorts_per_lf: Optional mapping of LF index to list of
                          RenderedShort objects.
            publish_day: Day-of-week name ("mon".."sun"). When provided,
                         overrides the regional default. Must correspond
                         to a day in ``self.long_form_days``.
            shorts_counts: Optional list parallel to ``regular_lfs`` giving
                          the ACTUAL number of shorts (bucket size) each LF
                          produced. When provided, each LF's window is
                          numbered from that LF's cumulative post offset and
                          capped at its bucket size, so slot ``short_index``
                          values align with the per-LF post numbering
                          (``_build_social_posts_per_lf``) and an oversized
                          LF's shorts never spill into the next LF's window.
                          ``None`` preserves the legacy fixed per-window
                          capacity with continuous cross-window numbering.
            locked_identity_times: Optional pre-resolved run-wide posting-time
                          pair (``datetime.time`` list). When set, it is used
                          as the identity verbatim — no ledger re-resolution —
                          and identity-time slots are excluded from the
                          collision set so a resumed run keeps its own booked
                          times. ``None`` resolves the identity fresh.
            lf_dates: Optional publish date per LF, parallel to ``regular_lfs``,
                          for a run whose LFs no longer sit one series apart
                          (a live LF plus slots re-resolved behind it). Each
                          LF's shorts window still ends one series step after
                          its own date. ``None`` = the series from ``start_date``.

        Returns:
            DriveReleaseSchedule with all slots. ``identity_times`` echoes the
            resolved run-wide posting-time pair for read-back on resume.
        """
        # Determine publish day. Priority: explicit arg > regional
        # default. The regional-default branch is now defensive — both
        # regions use Wednesday under the current cadence.
        # If an explicit publish_day is given that doesn't match
        # self.long_form_days, log a warning and honor the explicit
        # value anyway — tests/dev runs may deliberately schedule
        # off-cadence.
        if publish_day:
            publish_dow = DAY_NAMES.get(publish_day.lower()[:3])
            if publish_dow is None:
                raise ValueError(
                    f"Invalid publish_day {publish_day!r}; expected one of {list(DAY_NAMES.keys())}"
                )
            if publish_dow not in self.long_form_days:
                logger.warning(
                    "generate_drive_schedule: publish_day=%s (dow=%d) "
                    "is not in long_form_days=%s — scheduling off-cadence",
                    publish_day,
                    publish_dow,
                    sorted(self.long_form_days),
                )
        elif region == "usa":
            # The earlier cadence was Sat (dow=5); the current one is Wed (dow=2).
            # Honor self.long_form_days as the source of truth — pick
            # the smallest dow in the set (Wed under new cadence).
            publish_dow = min(self.long_form_days) if self.long_form_days else 2
        else:
            publish_dow = min(self.long_form_days) if self.long_form_days else 2

        # Resolve start date
        if start_date:
            base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            # The base LF date is a slot decision: seo_manager makes it (Phase D asks
            # next_free_slot) and hands it in. No ledger-blind date of our own.
            raise ValueError(
                "generate_drive_schedule: start_date is required; the base LF date comes "
                "from seo_manager (next_free_slot), never from this scheduler"
            )

        # Slot allocation. The drive picks ONE consistent set of times for its
        # whole run (its "identity") and posts at those every day — it does NOT
        # re-optimize per date, so a single drive never drifts across times. The
        # identity is the drive's resolved DEFAULT times when they're broadly free
        # across the run window; any default another drive owns on MOST of the
        # window is replaced — once, run-wide — by the next priority ``slot_pool``
        # time that is broadly free (so a second concurrent USA drive locks to
        # 10:00/16:00 for its whole run, not just the days the other drive is present). A rare
        # single-day collision is still substituted per-day from the pool, but the
        # identity stays fixed. Empty pool/occupied ⇒ identity == defaults ==
        # exact legacy behavior.
        defaults = self.shorts_times[:shorts_per_day] if shorts_per_day else self.shorts_times
        pool = slot_pool or self.shorts_times
        per_day = shorts_per_day if shorts_per_day is not None else len(defaults)
        taken = set(occupied_slots or ())
        # A persisted run-wide lock wins. A resumed drive reuses the SAME
        # posting-time pair it locked on its first Phase-D build rather than
        # re-resolving against a ledger that now holds its OWN booked shorts
        # (which would contest its identity times and flip it mid-run). Only
        # resolve from the ledger when no lock exists yet.
        if locked_identity_times:
            identity = list(locked_identity_times)
            # The drive OWNS its locked times for the whole run, so a booking
            # already sitting at one of them is THIS drive's own short (a
            # resumed run re-seeing its earlier bookings) — not a rival to
            # dodge. Drop identity-time slots from the collision set so per-day
            # placement returns the locked pair verbatim instead of
            # substituting the drive's own slots away.
            _locked = {t.strftime("%H:%M:%S") for t in identity}
            taken = {(d, t) for (d, t) in taken if t not in _locked}
        else:
            identity = self._select_run_identity_times(
                base_date, len(regular_lfs), defaults, pool, per_day, taken
            )

        # Schedule each LF one cadence interval apart, as seo_manager's series rule
        # says. required_shorts_per_lf derives the per-LF shorts target from the same
        # interval constant (re-exported from seo_manager), so the slot supply and the
        # required count cannot drift.
        lf_slots: List[ScheduledSlot] = []
        shorts_slots: List[ScheduledSlot] = []
        short_idx = 1
        # Cumulative per-LF post offset. When actual bucket sizes are
        # supplied (``shorts_counts``), each LF's window is numbered from its
        # own offset and capped at its bucket size, so slot ``short_index``
        # values track the per-LF post numbering
        # (``_build_social_posts_per_lf``'s ``global_index_offset``) and an
        # LF's shorts never spill into the next LF's window. ``None`` keeps
        # the legacy continuous cross-window numbering.
        post_offset = 0

        # seo_manager's series rule gives every LF date plus the one after the last
        # (the final shorts window's end); GMS does no date arithmetic of its own.
        from .seo_slots import lf_series

        if lf_dates is not None and len(lf_dates) != len(regular_lfs):
            raise ValueError(
                f"generate_drive_schedule: lf_dates has {len(lf_dates)} dates for "
                f"{len(regular_lfs)} long-forms"
            )
        series = lf_series(base_date, len(regular_lfs) + 1)

        for i, lf in enumerate(regular_lfs):
            lf_date = lf_dates[i] if lf_dates is not None else series[i]
            lf_dt = datetime.combine(lf_date, self.long_form_time, tzinfo=EST)

            lf_slots.append(
                ScheduledSlot(
                    short_index=0,  # 0 = long form marker
                    date=lf_date.strftime("%Y-%m-%d"),
                    time=self.long_form_time.strftime("%H:%M:%S"),
                    datetime_est=lf_dt.isoformat(),
                    day_of_week=lf_date.strftime("%a"),
                )
            )

            # Generate shorts slots between this LF and next LF
            next_lf_date = lf_series(lf_date, 2)[1] if lf_dates is not None else series[i + 1]
            next_lf_dt = datetime.combine(next_lf_date, self.long_form_time, tzinfo=EST)

            # Shorts start same day after LF or next day at 8am
            shorts_start = lf_dt
            current_date = lf_date

            # With actual bucket sizes, rebase numbering to this LF's post
            # offset and cap the window at its bucket size; ``None`` = legacy.
            window_cap = shorts_counts[i] if shorts_counts is not None else None
            if window_cap is not None:
                short_idx = post_offset + 1
            slots_this_window = 0

            # Spread the bucket across the whole window instead of
            # front-loading it — a bucket smaller than the window's full
            # per-day capacity previously filled the earliest days to
            # capacity, then left the tail of the window empty (e.g. a
            # 12-short bucket in a 7-day/2-per-day window left the last
            # eligible day with zero shorts). Compute each date's ELIGIBLE
            # identity-slot count (respecting the same boundary rule used
            # below) and spread the bucket across those caps. ``None``
            # (legacy path) skips this entirely — day_allowance stays
            # None and every day gets its full identity allocation, byte
            # for byte the pre-allowance behavior.
            day_allowance = None
            if window_cap is not None:
                window_dates = []
                _d = lf_date
                while _d <= next_lf_date:
                    window_dates.append(_d)
                    _d += timedelta(days=1)
                eligible_counts = []
                for _wd in window_dates:
                    _cnt = 0
                    for _t in identity:
                        _cand = datetime.combine(_wd, _t, tzinfo=EST)
                        if shorts_start < _cand < next_lf_dt:
                            _cnt += 1
                    eligible_counts.append(_cnt)
                _allowances = _spread_bucket_across_window(eligible_counts, window_cap)
                day_allowance = {
                    _wd.strftime("%Y-%m-%d"): _a for _wd, _a in zip(window_dates, _allowances)
                }

            while current_date <= next_lf_date:
                if window_cap is not None and slots_this_window >= window_cap:
                    break
                date_str = current_date.strftime("%Y-%m-%d")
                used_today: set = set()
                day_cap = day_allowance.get(date_str) if day_allowance is not None else None
                for default_time in identity:
                    if window_cap is not None and slots_this_window >= window_cap:
                        break
                    if day_cap is not None and len(used_today) >= day_cap:
                        break
                    default_dt = datetime.combine(current_date, default_time, tzinfo=EST)
                    # Boundary days exclude out-of-window times (e.g. the next LF's
                    # drop morning); skip the default, as the legacy path did.
                    if not (shorts_start < default_dt < next_lf_dt):
                        continue
                    default_str = default_time.strftime("%H:%M:%S")
                    chosen_str, chosen_dt = None, None
                    if (date_str, default_str) not in taken and default_str not in used_today:
                        chosen_str, chosen_dt = default_str, default_dt
                    else:
                        # Default slot taken — substitute the next free pool time.
                        for sub_time in pool:
                            sub_dt = datetime.combine(current_date, sub_time, tzinfo=EST)
                            if not (shorts_start < sub_dt < next_lf_dt):
                                continue
                            sub_str = sub_time.strftime("%H:%M:%S")
                            if (date_str, sub_str) in taken or sub_str in used_today:
                                continue
                            chosen_str, chosen_dt = sub_str, sub_dt
                            break
                    if chosen_str is None:
                        continue  # fully booked window — leave this slot empty
                    shorts_slots.append(
                        ScheduledSlot(
                            short_index=short_idx,
                            date=date_str,
                            time=chosen_str,
                            datetime_est=chosen_dt.isoformat(),
                            day_of_week=current_date.strftime("%a"),
                        )
                    )
                    taken.add((date_str, chosen_str))  # reserve within this run too
                    used_today.add(chosen_str)
                    short_idx += 1
                    slots_this_window += 1
                current_date += timedelta(days=1)

            # Advance the post offset by the FULL bucket size (not the
            # number of slots physically placed) so the next LF's numbering
            # stays aligned with its posts even if this window ran short.
            if window_cap is not None:
                post_offset += window_cap

        # Rank-aware slot assignment: when shorts_per_lf is provided,
        # re-number slots so best-ranked shorts get the earliest times.
        if shorts_per_lf:
            shorts_slots = self._apply_rank_ordering(
                shorts_slots,
                lf_slots,
                regular_lfs,
                shorts_per_lf,
            )

        return DriveReleaseSchedule(
            region=region,
            lf_slots=lf_slots,
            shorts_slots=shorts_slots,
            identity_times=[t.strftime("%H:%M:%S") for t in identity],
        )

    @staticmethod
    def _apply_rank_ordering(
        shorts_slots: List[ScheduledSlot],
        lf_slots: List[ScheduledSlot],
        regular_lfs: list,
        shorts_per_lf: dict,
    ) -> List[ScheduledSlot]:
        """Re-number shorts slots so best-ranked shorts get earliest times.

        For each LF's shorts window, sorts the RenderedShort objects by rank
        (ascending — rank 1 is best) and assigns them to time slots in
        chronological order. Slots beyond the number of shorts keep their
        original sequential numbering.

        Args:
            shorts_slots: All shorts slots (chronological, sequentially numbered).
            lf_slots: LF slots for boundary detection.
            regular_lfs: RegularLFPlan list (for LF index mapping).
            shorts_per_lf: Mapping of LF index to list of RenderedShort objects.

        Returns:
            Updated shorts_slots with rank-aware short_index values.
        """
        if not shorts_slots or not shorts_per_lf:
            return shorts_slots

        # Partition shorts_slots by LF window (between consecutive LF dates)
        result = []
        for i, lf in enumerate(regular_lfs):
            lf_dt = _to_utc(lf_slots[i].datetime_est) if i < len(lf_slots) else None
            next_dt = _to_utc(lf_slots[i + 1].datetime_est) if i + 1 < len(lf_slots) else None

            # Collect slots in this LF's window. Compare in UTC, not ISO
            # strings — see ``_to_utc`` docstring for the DST rationale.
            window_slots = []
            for slot in shorts_slots:
                slot_dt = _to_utc(slot.datetime_est)
                if lf_dt is not None and slot_dt <= lf_dt:
                    continue
                if next_dt is not None and slot_dt >= next_dt:
                    continue
                window_slots.append(slot)

            # Get shorts for this LF, sorted by rank (best first)
            lf_shorts = shorts_per_lf.get(lf.index, [])
            ranked = sorted(lf_shorts, key=lambda s: (s.rank, s.index))

            # Assign rank-ordered shorts to chronological slots.
            # Best-ranked short (lowest rank) gets the earliest time slot.
            # Use the short's .index (original extraction order), not .rank.
            for j, slot in enumerate(window_slots):
                if j < len(ranked):
                    slot.short_index = ranked[j].index
                # Slots beyond available shorts keep original numbering

            result.extend(window_slots)

        return result


@dataclass
class DriveReleaseSchedule:
    """Complete release schedule for a drive pipeline output set."""

    region: str
    lf_slots: List[ScheduledSlot]
    shorts_slots: List[ScheduledSlot]
    # wall-clock ISO-8601 stamp captured at schedule build time
    # (``_build_and_persist_drive_schedule`` in ``phase_d.py``). Drives
    # the rebase-on-stale-resume staleness guard
    # (``_maybe_rebase_drive_schedule``). Optional for legacy schedules
    # written before the stamp existed — a missing stamp is treated as
    # "infinitely stale → eligible to recompute on resume" so older
    # in-flight runs self-heal on the first Phase-D re-entry.
    computed_at: Optional[str] = None
    # The run-wide locked shorts posting-time pair (``HH:MM:SS`` strings),
    # resolved ONCE on the drive's first Phase-D build and read back verbatim
    # on every resume so the drive never re-contends against its OWN booked
    # shorts (which would flip its identity mid-run). ``None`` on a legacy
    # schedule written before this field existed → the next build resolves fresh.
    identity_times: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "lf_slots": [s.to_dict() for s in self.lf_slots],
            "shorts_slots": [s.to_dict() for s in self.shorts_slots],
            "computed_at": self.computed_at,
            "identity_times": self.identity_times,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriveReleaseSchedule":
        """Re-hydrate from a JSON-loaded dict.

        Closes the state-contract round-trip gap that a
        ``_drive_schedule_field()`` shim would only paper over — the
        contract path enforces the dataclass shape at load time so
        downstream consumers can rely on attribute access
        (``.lf_slots``, ``.shorts_slots``) regardless of resume vs. fresh.
        """
        return cls(
            region=d.get("region", ""),
            lf_slots=[ScheduledSlot.from_dict(s) for s in d.get("lf_slots", []) or []],
            shorts_slots=[ScheduledSlot.from_dict(s) for s in d.get("shorts_slots", []) or []],
            computed_at=d.get("computed_at"),
            identity_times=d.get("identity_times"),
        )
