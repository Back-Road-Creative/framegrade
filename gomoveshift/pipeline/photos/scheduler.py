"""Daily photo post scheduler.

Schedules 1 photo post per day at the canonical photo time (EST),
avoiding conflicts with the configured shorts and long-form windows.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from ..content_ledger import ContentLedger
from ..schedule_constants import EST, PHOTO_POST_TIME

logger = logging.getLogger(__name__)


@dataclass
class PhotoSlot:
    """A scheduled photo posting slot."""

    date: str  # YYYY-MM-DD
    time: str  # HH:MM:SS
    day_of_week: str


class PhotoScheduler:
    """Schedules daily photo posts, checking ledger for conflicts."""

    def __init__(self, ledger: Optional[ContentLedger] = None):
        # No-arg load resolves the canonical SEO-storage ledger
        # (``<config.seo.storage_path>/.content_ledger.json``) and pins it.
        self.ledger = ledger or ContentLedger.load()

    def schedule_batch(
        self,
        count: int = 7,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[PhotoSlot]:
        """Schedule a batch of photo posts.

        Args:
            count: Number of slots to schedule
            start_date: Start date (YYYY-MM-DD), defaults to tomorrow. A start
                      in the past is clamped forward to today (never raises) —
                      ``publish_batch`` passes the *generate* day, so an
                      overnight review legitimately arrives with a past start.
            end_date: End date (YYYY-MM-DD), inclusive. When set, slots
                      will not extend past this date — fewer than `count`
                      slots may be returned.

        Returns:
            List of PhotoSlot objects
        """
        now = datetime.now(EST)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if start_date:
            current = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=EST)
            if current.date() < today.date():
                logger.warning(
                    f"start_date {start_date} is in the past "
                    f"(today: {today.date()}) — clamping to {today.date()}"
                )
                current = today
        else:
            current = today + timedelta(days=1)

        end_dt = None
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=EST)

        # Re-read the shared ledger: another process may have booked slots
        # since this scheduler was constructed.
        self.ledger.reload()
        occupied = self.ledger.get_occupied_slots()
        slots = []
        # Tighter safety cap: count + 14 when no end_date (enough for conflicts)
        max_days = count + 14 if end_dt is None else (end_dt - current).days + 1
        days_checked = 0

        while len(slots) < count and days_checked < max_days:
            # Hard boundary: stop if past end_date
            if end_dt is not None and current.date() > end_dt.date():
                break

            candidate = datetime.combine(current.date(), PHOTO_POST_TIME, tzinfo=EST)
            slot_key = (candidate.strftime("%Y-%m-%d"), candidate.strftime("%H:%M:%S"))

            # Never book a slot that has already passed — Metricool rejects
            # a past scheduled time and the ledger booking would be wrong
            # forever. Mirrors ContentLedger's photo branch.
            if candidate <= now:
                current += timedelta(days=1)
                days_checked += 1
                continue

            if slot_key not in occupied:
                slots.append(
                    PhotoSlot(
                        date=slot_key[0],
                        time=slot_key[1],
                        day_of_week=candidate.strftime("%A"),
                    )
                )

            current += timedelta(days=1)
            days_checked += 1

        if slots:
            logger.info(f"Scheduled {len(slots)} photo slots starting {slots[0].date}")
        else:
            logger.info("Scheduled 0 photo slots (all days occupied or range empty)")

        if len(slots) < count:
            logger.warning(
                f"Only {len(slots)}/{count} slots available "
                f"(some days had conflicts or range too short)"
            )

        # Defensive validation
        self._validate_range(slots, start_date, end_date)

        return slots

    def _validate_range(
        self,
        slots: List[PhotoSlot],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> None:
        """Verify all slots fall within [start_date, end_date]."""
        for slot in slots:
            if start_date and slot.date < start_date:
                raise ValueError(f"Slot {slot.date} is before start_date {start_date}")
            if end_date and slot.date > end_date:
                raise ValueError(f"Slot {slot.date} is after end_date {end_date}")
