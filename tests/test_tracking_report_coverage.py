"""A report must state the population it covered.

``TrackingRecord`` (schema.py) enumerates only ``shorts`` — there is no
representation anywhere in ``tracking/`` for long-form videos, stills,
carousels, or a post a human published directly in an app rather than
through this pipeline. ``PerformanceReporter`` folds every figure ("Total
views", "Average views/short", best/worst rankings) over that shorts-only
population and never said so: a reader could take a shorts-only total for
a channel-wide one. This pins a coverage disclosure onto both report
surfaces (per-project and cross-project) naming what was counted and what
was never enumerated at all.
"""

from __future__ import annotations

from gomoveshift.tracking.report import PerformanceReporter
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.store import TrackingStore


def _short(index: int, **platforms: PlatformMetrics) -> ShortRecord:
    short = ShortRecord(index=index, file=f"short{index}.mp4", metrics=platforms)
    short.collected_at = "2026-01-01T00:00:00"
    return short


def test_project_report_header_states_shorts_only_coverage():
    """The per-project report must say the numbers cover shorts only, and
    name the kinds that are never enumerated."""
    record = TrackingRecord(project="demo", location="Ireland")
    record.shorts.append(_short(1, youtube=PlatformMetrics(views=100)))

    reporter = PerformanceReporter(store=TrackingStore())
    header = reporter._header(record)

    assert "long-form" in header.lower()
    assert "not enumerated" in header.lower() or "not collected" in header.lower()


def test_project_report_with_no_metrics_still_states_coverage(tmp_path):
    """Even the "no metrics yet" early-return must say what population it is
    "no metrics" over — a reader must not take an empty shorts report as an
    empty channel.

    Asserts the coverage disclosure itself, NOT the "registered
    shorts" wording: that string is present regardless of the disclosure,
    so a test pinned to it would pass whether or not the disclosure is there.
    """
    store = TrackingStore(root=tmp_path)
    record = TrackingRecord(project="demo", location="Ireland")
    record.shorts.append(ShortRecord(index=1, file="short1.mp4"))
    store.save(record)

    reporter = PerformanceReporter(store=store)
    report = reporter.generate_project_report("demo")

    assert "long-form" in report.lower()
    assert "not enumerated" in report.lower()


def test_cross_project_report_with_no_metrics_states_coverage(tmp_path):
    """The cross-project early return is the widest-sounding surface in the
    module ("across any project yet") and so needs the disclosure most."""
    reporter = PerformanceReporter(store=TrackingStore(root=tmp_path))
    report = reporter.generate_cross_project_report()

    assert "long-form" in report.lower()
    assert "not enumerated" in report.lower()


def test_cross_project_report_states_shorts_only_coverage(tmp_path):
    store = TrackingStore(root=tmp_path)
    record = TrackingRecord(project="demo", location="Ireland")
    record.shorts.append(_short(1, youtube=PlatformMetrics(views=100)))
    store.save(record)

    reporter = PerformanceReporter(store=store)
    report = reporter.generate_cross_project_report()

    assert "long-form" in report.lower()
    assert "not enumerated" in report.lower() or "not collected" in report.lower()
