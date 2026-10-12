"""A tracked short's posting time says which moment it is.

The publishing side already refuses to let a late human click masquerade as a
platform publication: `deliveries.published_at_source` records which of four
promises a timestamp makes, and `queue.transition` enforces the difference
(`publishing/queue.py`). That contract stopped at the queue. The one bridge
from the queue into tracking (`cli/handlers/metrics.py`) carried platform ids
and nothing else, and `ShortRecord.posted_at` was a bare ISO string -- so
every downstream consumer of a posting time read an operator's Tuesday-evening
catch-up click and a platform's own confirmed publish second as the same fact.

`tracking/decline.py` and `tracking/collector.py` anchor age-based judgements
to that string. An `operator_confirmed` time can be hours late, which makes a
short look younger than it is and its early decline look steeper -- the exact
class of error the publishing-side provenance was introduced to prevent.

So the source travels with the time, and the invalid state -- a posting time
whose provenance is unknown because nobody recorded it -- is made unreachable
for new writes rather than validated after the fact.
"""

from __future__ import annotations

import pytest

from gomoveshift.publishing import queue as pubqueue
from gomoveshift.tracking import schema
from gomoveshift.tracking.schema import ShortRecord
from gomoveshift.tracking.store import TrackingStore


def test_a_posting_time_carries_its_source():
    short = ShortRecord(
        index=1,
        posted_at="2026-09-12T10:00:00",
        posted_at_source=pubqueue.PUBLISHED_AT_PLATFORM_CONFIRMED,
    )

    assert short.posted_at_source == "platform_confirmed"


def test_the_source_survives_a_round_trip():
    short = ShortRecord(
        index=1,
        posted_at="2026-09-12T10:00:00",
        posted_at_source=pubqueue.PUBLISHED_AT_OPERATOR_CONFIRMED,
    )

    assert ShortRecord.from_dict(short.to_dict()).posted_at_source == "operator_confirmed"


def test_a_legacy_record_reads_back_with_no_source_rather_than_a_guess():
    """Records written before this existed have a posting time and no
    provenance. That is an honest unknown and it is never backfilled -- the
    same rule `deliveries` applies to its own pre-existing rows. Inventing
    `operator_confirmed` for them would assert evidence nobody has."""
    legacy = {"index": 1, "posted_at": "2026-09-01T08:00:00", "platform_ids": {}}

    restored = ShortRecord.from_dict(legacy)

    assert restored.posted_at == "2026-09-01T08:00:00"
    assert restored.posted_at_source is None


def test_a_new_posting_time_may_not_be_recorded_without_its_source():
    """Robust by construction: the gap this module guards cannot reopen by someone
    later setting `posted_at` and forgetting the source. A time with no
    provenance is refused at construction rather than caught downstream."""
    with pytest.raises(ValueError) as excinfo:
        ShortRecord(index=1, posted_at="2026-09-12T10:00:00")

    assert "posted_at_source" in str(excinfo.value)


def test_a_source_without_a_time_is_also_refused():
    """A provenance label describing no timestamp is a claim about nothing."""
    with pytest.raises(ValueError):
        ShortRecord(index=1, posted_at_source=pubqueue.PUBLISHED_AT_PLATFORM_CONFIRMED)


def test_only_the_queue_s_own_sources_are_accepted():
    """A record claiming a source the queue never issues cannot be
    interpreted by anything downstream, so it is refused rather than
    stored."""
    with pytest.raises(ValueError) as excinfo:
        ShortRecord(
            index=1,
            posted_at="2026-09-12T10:00:00",
            posted_at_source="probably_fine",
        )

    assert "probably_fine" in str(excinfo.value)


def test_a_record_with_neither_is_fine():
    """A short that has not been posted yet is the normal starting state."""
    short = ShortRecord(index=1)

    assert short.posted_at is None
    assert short.posted_at_source is None


def test_the_two_packages_agree_on_the_vocabulary_in_both_directions():
    """`tracking` deliberately holds its own literal tuple rather than
    importing the queue's -- `publishing` -> `tracking` is the only import edge
    between these packages, and importing back would make them mutually
    dependent. The agreement is pinned HERE, where a test may import both:
    a source added on either side and not the other fails loudly, instead of
    a queue row being refused at ingest or a record claiming a source nothing
    issues."""
    assert set(schema._PUBLISHED_AT_SOURCES) == set(pubqueue._PUBLISHED_AT_SOURCES)


def test_every_source_the_queue_can_issue_is_accepted_here():
    for source in pubqueue._PUBLISHED_AT_SOURCES:
        short = ShortRecord(index=1, posted_at="2026-09-12T10:00:00", posted_at_source=source)
        assert short.posted_at_source == source


# ── The bridges: queue -> tracking ───────────────────────────────────────
#
# There are exactly two, and before this both carried the platform id alone:
# `queue._register_platform_id_best_effort` on every live publish, and
# `metrics backfill-ids` for rows that published before that hook existed.
# A fix to one and not the other leaves the gap open on the path that
# actually runs, so both are pinned here.


def test_a_time_with_no_source_hands_over_neither():
    """A pre-provenance row has a `published_at` and no source. Passing the
    bare time on would turn an unknown into a measurement, so the pair is
    dropped and only the id registers."""
    assert pubqueue.recorded_publish_time("2026-09-01T08:00:00", None) == (None, None)


def test_a_source_with_no_time_hands_over_neither():
    assert pubqueue.recorded_publish_time(None, "platform_confirmed") == (None, None)


def test_a_complete_pair_crosses_intact():
    assert pubqueue.recorded_publish_time("2026-09-12T10:00:00", "platform_confirmed") == (
        "2026-09-12T10:00:00",
        "platform_confirmed",
    )


def test_the_live_publish_hook_carries_the_provenance(tmp_path, monkeypatch):
    """`transition` derives `published_at` itself and refuses a publish with
    no `published_at_source`, so every row it writes has the complete pair --
    and the hook it fires must land both on the tracked short."""
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    store = TrackingStore(root=tmp_path)
    store.register_short("proj", 1)

    conn = pubqueue.open_queue_db(tmp_path)
    delivery = pubqueue.enqueue(conn, "proj", 1, "instagram", "hash-a")
    pubqueue.transition(
        conn,
        delivery,
        "queued",
        "published",
        platform_post_id="ig_1",
        published_at_source=pubqueue.PUBLISHED_AT_OPERATOR_CONFIRMED,
    )
    conn.close()

    short = TrackingStore(root=tmp_path).load("proj").shorts[0]
    assert short.platform_ids["instagram"] == "ig_1"
    assert short.posted_at_source == "operator_confirmed"
    assert short.posted_at


def test_registering_an_id_may_not_record_a_time_with_no_provenance(tmp_path):
    """The store refuses the half-pair rather than storing an unqualified
    string, which would be indistinguishable from a legacy honest unknown."""
    store = TrackingStore(root=tmp_path)
    store.register_short("proj", 1)

    with pytest.raises(ValueError) as excinfo:
        store.update_platform_id("proj", 1, "instagram", "ig_1", posted_at="2026-09-12T10:00:00")

    assert "provenance" in str(excinfo.value)
