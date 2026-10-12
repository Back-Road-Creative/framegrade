"""Channel→journey reconciler for ``gomoveshift sync-site --journeys``.

The per-drive auto-publish hook only knows the videos its own run produced, so a
long-form published outside that run (a separate leg of a multi-state drive, a
re-upload, back-catalog) never reaches a journey page. This is the
channel-reconciliation backstop: for each public ``real_lf`` that
``data/video_routing.yaml`` assigns to a journey, embed it into the routed day
via :func:`journal_publisher.apply_merge` (so dedup-by-id and ``video_count``
maintenance match the per-drive flow). A held id is not ``real_lf`` in the
manifest, so it auto-embeds on the first sync after it goes public — the
periodic sync *is* the deferral mechanism (no ``pending_journal`` sidecar).
"""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import yaml
from ruamel.yaml import YAMLError as RuamelYAMLError

from .._atomic_write import atomic_write_text
from ..journal_publisher import (
    MAX_TITLE_LEN,
    MergePlan,
    VideoEntry,
    apply_merge,
    format_duration,
    journey_lock,
    load_journey,
    serialize_journey,
)
from ..region_resolver import derive_region
from .videos_yaml import VideoLibrary

ROUTING_FILENAME = "video_routing.yaml"


class JourneyRegionMismatchError(ValueError):
    """A finished LF's ``ResolvedLocation`` region disagrees with the run's
    ``--region``. Raised (loud-refuse) before any journey merge / routing write:
    the routing written from a finished drive is authoritative forever, so a
    wrong/stale ``--region`` would durably file (e.g.) a USA LF under the
    Ireland journey. Better to refuse the merge than route silently."""


def assert_region_consistency(state: dict, region: str | None) -> None:
    """Loud-refuse the journal merge when any LF disagrees with the run region.

    Derive each LF's region from its canonical ``ResolvedLocation``
    (``state["per_lf_resolved_location"]``) via the single ``derive_region``
    decider — never the display label — and raise :class:`JourneyRegionMismatchError`
    on the FIRST positive disagreement with ``region`` (the ``--region`` that
    selected the journey). All-or-nothing: one wrong-region LF refuses the whole
    run's merge, because the operator must fix ``--region`` (or split the drive)
    before ANY of these ids is routed.

    No-op when ``region`` is unset (nothing to cross-check), when there is no
    ``per_lf_resolved_location`` map, or when an LF's model is indeterminate
    (GPS-less / legacy → ``derive_region`` returns ``""``). This mirrors the
    pipeline's other per-LF region gates, which are no-ops on legacy state.
    """
    if not region:
        return
    resolved = state.get("per_lf_resolved_location")
    if not isinstance(resolved, dict):
        return
    run_region = str(region).strip().lower()
    mismatches: list[tuple[str, str]] = []
    for key, model in resolved.items():
        obj = model if not isinstance(model, dict) else SimpleNamespace(**model)
        lf_region = derive_region(None, resolved=obj)
        if lf_region and lf_region != run_region:
            mismatches.append((str(key), lf_region))
    if mismatches:
        detail = ", ".join(f"LF {k} -> {r}" for k, r in sorted(mismatches))
        raise JourneyRegionMismatchError(
            f"REGION-CROSSCHECK: run --region {run_region!r} but {len(mismatches)} "
            f"LF(s) resolve elsewhere ({detail}); refusing to route them into the "
            "wrong journey (routing is authoritative forever). Re-run with the "
            "correct --region, or split the drive by region."
        )


@dataclass
class ReconcileResult:
    """Per-run summary. ``warnings`` are non-fatal (routing points at a missing
    journey/day) — never raised, so a recoverable reconcile can't break the sync."""

    embedded: list[tuple[str, str]] = field(default_factory=list)  # (journey, youtube_id)
    skipped_present: list[tuple[str, str]] = field(default_factory=list)
    deferred_held: list[str] = field(default_factory=list)  # routed but not yet public
    warnings: list[str] = field(default_factory=list)


def load_routing(path: Path) -> dict[str, dict]:
    """Read ``video_routing.yaml`` → ``{id: {"journey": slug, "date": "YYYY-MM-DD"}}``.

    Missing file → ``{}`` (a no-op, not an error). The top-level ``routes:``
    mapping leaves room for other collection kinds later; malformed rows (no
    journey/date) are skipped. A document that parses but is not a routing
    mapping at all (a YAML list, a bare scalar) is likewise ``{}`` -- it
    records no routes. It used to raise ``AttributeError`` from ``.get``/
    ``.items``, which is neither of this function's documented outcomes and
    crashed both callers: ``reconcile_journeys`` promises it never raises on a
    routing/data inconsistency, and the catalog reader lost every photo entry
    it had already gathered.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    document = yaml.safe_load(p.read_text())
    routes = document.get("routes") if isinstance(document, dict) else None
    if not isinstance(routes, dict):
        return {}
    return {
        str(vid): {"journey": str(r["journey"]), "date": str(r["date"])}
        for vid, r in routes.items()
        if isinstance(r, dict) and r.get("journey") and r.get("date")
    }


def routing_from_state(state: dict, journey: str) -> dict[str, dict]:
    """Build ``{video_id: {journey, date}}`` from a finished drive's pipeline state.

    Pairs each LF's uploaded ``video_id`` with its recording date (the same day
    key ``journal_publisher`` embeds under) so a later ``sync-site --journeys``
    can re-embed/heal these ids. Both source dicts are keyed by the same LF
    index in the on-disk state, so we match by raw key; LFs missing either an
    id or a date are skipped.
    """
    uploads = state.get("youtube_uploads")
    rec_dates = state.get("per_lf_recording_date")
    if not isinstance(uploads, dict) or not isinstance(rec_dates, dict):
        return {}
    out: dict[str, dict] = {}
    for key, record in uploads.items():
        if not isinstance(record, dict):
            continue
        vid = record.get("video_id")
        date = rec_dates.get(key)
        if vid and date:
            out[str(vid)] = {"journey": journey, "date": str(date)}
    return out


def merge_routing(path: Path, entries: dict[str, dict]) -> None:
    """Idempotent upsert of ``{id: {journey, date}}`` into ``video_routing.yaml``.

    Preserves existing routes (and any other top-level keys); a re-published id
    overwrites its own route. The file is machine-maintained data, so a plain
    sorted YAML dump (deterministic diffs) is fine. Atomic write — a torn file
    would strand the routing table. No-op on empty ``entries``.
    """
    if not entries:
        return
    p = Path(path)
    doc = (yaml.safe_load(p.read_text()) or {}) if p.is_file() else {}
    routes = doc.get("routes")
    if not isinstance(routes, dict):
        routes = {}
    for vid, route in entries.items():
        routes[str(vid)] = {"journey": route["journey"], "date": route["date"]}
    doc["routes"] = routes
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        yaml.safe_dump(doc, sort_keys=True, default_flow_style=False, allow_unicode=True)
    )
    os.replace(tmp, p)


def _existing_ids(doc: dict) -> set[str]:
    """Every ``youtube_id`` already embedded anywhere in the journey doc."""
    return {
        str(v["youtube_id"])
        for day in doc.get("days", []) or []
        for v in day.get("videos", []) or []
        if isinstance(v, dict) and v.get("youtube_id")
    }


#: Every way a source file can be unusable without this process being at fault:
#: the read (``OSError``), then the parse. The two loaders use DIFFERENT yaml
#: libraries and neither ``YAMLError`` subclasses the other, so a single merged
#: tuple would silently stop catching half of what it names. ``load_journey``
#: also raises ``ValueError`` for a missing/malformed frontmatter delimiter.
_ROUTING_UNREADABLE = (OSError, yaml.YAMLError)
_JOURNEY_UNREADABLE = (OSError, ValueError, RuamelYAMLError)


def reconcile_journeys(manifest: dict, site_path: Path, *, apply: bool) -> ReconcileResult:
    """Embed routed public ``real_lf`` videos into their journeys.

    Reads the routing file + each journey markdown; mutates files only when
    ``apply`` is true. Returns a :class:`ReconcileResult`; never raises on a
    routing/data inconsistency, nor on a journey another writer is mid-write on
    (both become ``warnings``).
    """
    site_path = Path(site_path)
    result = ReconcileResult()
    routing_path = site_path / "data" / ROUTING_FILENAME
    try:
        routing = load_routing(routing_path)
    except _ROUTING_UNREADABLE as exc:
        # The docstring above promises this pass never raises on a routing/data
        # inconsistency -- but load_routing READS and PARSES a file, and both of
        # those fail. A truncated routing file used to end sync-site in a
        # traceback rather than the warning the contract names.
        result.warnings.append(f"routing file {routing_path} could not be read: {exc}")
        return result
    if not routing:
        return result

    real_lf = {
        v["id"]: v
        for v in manifest.get("videos") or []
        if v.get("category") == "real_lf" and v.get("id")
    }

    # Group routed ids by journey so each file is read/written exactly once.
    by_journey: dict[str, list[str]] = defaultdict(list)
    for vid, route in routing.items():
        if vid not in real_lf:
            result.deferred_held.append(vid)  # routed but not (yet) a public LF
        else:
            by_journey[route["journey"]].append(vid)

    # Hold each target journey's write lock for this whole pass, BEFORE the
    # first load: every journey writer takes it across its own load -> mutate ->
    # write window, so skipping it would serialize a doc that never saw theirs.
    # A journey someone else holds is dropped here rather than clobbered — its
    # ids embed on the next periodic sync. Non-blocking: N at once can't deadlock.
    paths = {slug: site_path / "content" / "journeys" / f"{slug}.md" for slug in by_journey}
    locks = ExitStack()
    for slug, path in sorted(paths.items()):
        if path.is_file() and not locks.enter_context(journey_lock(path, skip_if_busy=True)):
            result.warnings.append(
                f"{slug}: another writer holds this journey's lock; skipped "
                f"{len(by_journey[slug])} routed id(s) — the next sync embeds them."
            )
            by_journey.pop(slug)

    for slug in sorted(by_journey):
        journey_path = paths[slug]
        if not journey_path.is_file():
            result.warnings.append(f"routing references missing journey {slug!r} ({journey_path})")
            continue
        try:
            doc, body = load_journey(journey_path)
        except _JOURNEY_UNREADABLE as exc:
            # Degrade per file, the way the catalog reader does. One unreadable
            # journey used to abort the pass part-way through: journeys sorted
            # before it were already written, journeys after it were never
            # reached, and the caller got an exception instead of a result
            # saying which.
            result.warnings.append(
                f"{slug}: journey could not be read ({exc}); skipped "
                f"{len(by_journey[slug])} routed id(s)"
            )
            continue
        present = _existing_ids(doc)
        day_dates = {str(d.get("date")) for d in doc.get("days", []) or []}

        inserts: list[VideoEntry] = []
        for vid in by_journey[slug]:
            if vid in present:
                result.skipped_present.append((slug, vid))
                continue
            route_date = routing[vid]["date"]
            if route_date not in day_dates:
                result.warnings.append(
                    f"{slug}: routed id {vid} -> day {route_date} which has no matching "
                    "day section; author the day stub, then re-sync."
                )
                continue
            video = real_lf[vid]
            inserts.append(
                VideoEntry(
                    lf_index=0,  # manifest-sourced; apply_merge ignores it
                    youtube_id=vid,
                    title=str(video.get("title", ""))[:MAX_TITLE_LEN],
                    duration=format_duration(video.get("duration_seconds") or 0),
                    day_date=route_date,
                    publish_at=None,  # public-only — no future schedule to record
                )
            )

        if not inserts:
            continue
        apply_merge(
            doc,
            MergePlan(inserts=inserts, skips=[], errors=[]),
            # Dry run writes neither the journey nor the library.
            library=VideoLibrary.for_site(site_path, writable=apply),
        )
        if apply:
            # Atomic: a crash mid-write must not truncate a published page.
            atomic_write_text(journey_path, serialize_journey(doc, body))
        result.embedded.extend((slug, e.youtube_id) for e in inserts)

    locks.close()  # every write has settled
    return result
