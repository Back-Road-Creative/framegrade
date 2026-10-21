"""Own-channel video ids that no uploads listing returns.

A channel can hold private and unlisted videos that the uploads playlist and
its format-split siblings (``UUSH``, ``UULF``, ``UULV``) omit; only
``videos.list`` by id sees them. A deployment that has such videos lists their
ids in ``<GMS_INPUTS_DIR>/known_ids.txt``, one per line (``#`` starts a
comment). ``enumerate_uploads`` keeps one only if ``videos.list`` places it on
the caller's channel, so a deleted or foreign id costs nothing but its slot in
a 50-id batch.

The default is empty: this package ships no channel's inventory.
"""

from __future__ import annotations

from ...paths import gms_inputs_dir

FROZEN_KNOWN_IDS: tuple[str, ...] = ()

KNOWN_IDS_FILENAME = "known_ids.txt"


def load_known_ids() -> tuple[str, ...]:
    """The operator's ids from ``<GMS_INPUTS_DIR>/known_ids.txt`` in file order,
    repeats dropped; ``FROZEN_KNOWN_IDS`` when the file is absent."""
    path = gms_inputs_dir() / KNOWN_IDS_FILENAME
    if not path.is_file():
        return FROZEN_KNOWN_IDS
    ids: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        video_id = line.split("#", 1)[0].strip()
        if video_id:
            ids.setdefault(video_id)
    return tuple(ids)
