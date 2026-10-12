"""Own-channel video ids that no uploads listing returns.

A channel can hold private and unlisted videos that the uploads playlist and
its format-split siblings (``UUSH``, ``UULF``, ``UULV``) omit; only
``videos.list`` by id sees them. A deployment that has such videos supplies
their ids itself. ``enumerate_uploads`` keeps one only if ``videos.list``
places it on the caller's channel, so a deleted or foreign id costs nothing
but its slot in a 50-id batch.

The default is empty: this package ships no channel's inventory.
"""

from __future__ import annotations

FROZEN_KNOWN_IDS: tuple[str, ...] = ()
