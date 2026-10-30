"""Handler for ``gomoveshift publish-journal``.

Merges a completed drive run's YouTube uploads into the site checkout's
``content/journeys/<slug>.md`` ``days[].videos[]`` array. Idempotent by
``youtube_id``. Dry-run by default; ``--apply`` writes.

Two operating modes:
- Default — merge from a pipeline_state.json (Phase D produced it).
- ``--replay-pending`` — flush any sidecar entries whose ``publish_at``
  has now passed; intended for cron.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ...pipeline._atomic_write import atomic_write_text
from ...pipeline.journal_git_ops import (
    GitOpError,
    commit_paths,
    git_toplevel,
    is_tracked_dirty,
)
from ...pipeline.journal_publisher import (
    VideoEntry,
    _default_youtube_duration_lookup,
    _default_youtube_live_lookup,
    apply_merge,
    ffprobe_duration_seconds,
    format_duration,
    journey_lock,
    load_journey,
    plan_merge,
    plan_superlong_merge,
    serialize_journey,
)
from ...pipeline.pending_journal import (
    PENDING_FILENAME,
    PendingEntry,
    load_pending,
    merge_pending,
    pending_lock,
    save_pending,
    split_by_readiness,
)
from ...pipeline.site_sync.videos_yaml import VideoLibrary
from ...pipeline.youtube_sidecar import load_superlong_sidecar

logger = logging.getLogger(__name__)


def _resolve_journey_path(site_path: Path, slug: str) -> Path:
    return site_path / "content" / "journeys" / f"{slug}.md"


def _deferrals_to_pending(deferrals: list[VideoEntry], journey_path: Path) -> list[PendingEntry]:
    return [
        PendingEntry(
            youtube_id=e.youtube_id,
            publish_at=e.publish_at or "",
            title=e.title,
            duration=e.duration,
            day_date=e.day_date,
            journey_path=str(journey_path),
            lf_index=e.lf_index,
        )
        for e in deferrals
    ]


def _record_routes(site_path: Path, slug: str, inserts: list[VideoEntry]) -> Optional[Path]:
    """Upsert ``id -> {journey, date}`` into the site's ``video_routing.yaml``.

    An embed must never outlive the route that reproduces it. ``sync-site
    --journeys`` re-derives every journey page from that routing table, so an
    id embedded here without a route is invisible to the reconciliation
    backstop: each later regeneration drops it and the journal cron's superset
    guard refuses the push forever, because nothing can re-derive an embed
    whose route was never written (one unrouted replay embed can
    deadlock the hourly sync).

    Returns the routing file's path when it was written (so an ``--auto-commit``
    caller can commit it alongside the journey; leaving a tracked write
    uncommitted would trip the next run's ``is_tracked_dirty`` pre-check and
    stall every journey in the repo), else ``None``.

    Call this only for inserts actually written — never for skips, deferrals
    or a dry run. Like the per-drive call in ``handlers/drive.py``, routing is
    a side-quest: the journey YAML is already durable by the time we get here,
    so a failure (read-only site checkout, torn table) is logged and swallowed
    rather than reported as a failed publish.
    """
    routes = {
        e.youtube_id: {"journey": slug, "date": e.day_date}
        for e in inserts
        if e.youtube_id and e.day_date
    }
    if not routes:
        return None
    try:
        from ...pipeline.site_sync import journey_reconcile

        path = Path(site_path).resolve() / "data" / journey_reconcile.ROUTING_FILENAME
        journey_reconcile.merge_routing(path, routes)
        logger.info("auto-route: recorded %d route(s) -> journey %s", len(routes), slug)
        return path
    except Exception:  # never fail a journey write on a recoverable side-quest
        logger.exception("auto-route: routing upsert failed for journey %s", slug)
        return None


def run_publish_journal(
    *,
    state_path: Path,
    site_path: Path,
    journey_slug: str,
    apply: bool,
    allow_day_create: bool,
    out=None,
    err=None,
    ffprobe_duration: Callable[[Path], float] = ffprobe_duration_seconds,
    now: Optional[datetime] = None,
    pending_dir: Optional[Path] = None,
) -> int:
    """Returns the CLI exit code. Pulled out of cmd_publish_journal so the
    drive integration hook can call it directly with a parsed state dict.

    ``out``/``err`` default to None and resolve to ``sys.stdout``/``sys.stderr``
    at call time so pytest's capsys (which swaps the streams per-test) sees the
    output. Capturing them at function-def time would bypass capsys.

    ``pending_dir`` is where the deferral sidecar is written. Defaults to the
    state file's directory (standalone CLI). The drive auto-hook passes the
    durable drive output dir so the deferral lands where the replay cron's
    ``--scan-root`` looks — the state dir is on ephemeral ext4 staging, which
    the cron does not scan. When given, it also wins as the base for reading
    ``superlong_uploads.json``; absent it, that read prefers
    ``state["output_base"]`` over the deferral sidecar's own fallback, since
    the superlong sidecar always lives at the run's durable output root, not
    next to the state file (see ``DriveOrchestrator._resolve_paths``).
    """
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    pending_base = pending_dir if pending_dir is not None else state_path.parent
    # 1. State file
    if not state_path.is_file():
        print(f"Error: state file not found: {state_path}", file=err)
        return 2
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"Error: state file is not valid JSON: {exc}", file=err)
        return 2

    # The superlong sidecar (``superlong_uploads.json``) is written next to
    # the drive's durable OUTPUT ROOT (``state["output_base"]``), never at
    # the state file's own directory — ``pipeline_state.json`` lives under
    # ``<output_base>/_stages/`` and ``_stages`` can even be a wholly
    # separate ext4-backed mount (``DriveOrchestrator._resolve_paths`` /
    # ``GMS_INTERMEDIATE_DIR``), so ``state_path.parent`` is the wrong base
    # and ``state_path.parent.parent`` isn't guaranteed to line up either.
    # An explicit ``pending_dir`` (the drive auto-hook's seam) still wins;
    # absent that, prefer the state-recorded output_base over the pending
    # sidecar's own directory-of-state-file fallback.
    superlong_base = (
        pending_dir
        if pending_dir is not None
        else (Path(state["output_base"]) if state.get("output_base") else pending_base)
    )

    # 2. Something to merge (a superlong-only day has no regular LF upload).
    uploads = state.get("youtube_uploads") or {}
    if not uploads and not state.get("superlong_parts"):
        print(
            "Error: state['youtube_uploads'] is empty; did Phase D run?",
            file=err,
        )
        return 1

    # 3. Journey file must exist
    journey_path = _resolve_journey_path(site_path, journey_slug)
    if not journey_path.is_file():
        print(
            f"Error: journey file not found: {journey_path}\n"
            "(--allow-day-create stubs missing days within an existing trip, "
            "but you must author the trip frontmatter first.)",
            file=err,
        )
        return 1

    # Hold the journey lock across the whole load -> plan -> write window: any
    # other writer (photo ingest, replay cron, reconciler) starting inside it
    # would otherwise serialize a doc that never saw these inserts. Busy is a
    # recoverable refusal, NOT an exception — the uploads already happened.
    with journey_lock(journey_path, skip_if_busy=True) as held:
        if not held:
            message = (
                f"Error: another writer holds the lock on {journey_path}; nothing "
                "was written. Re-run once it finishes (the uploads are unaffected)."
            )
            print(message, file=err)
            logger.warning("%s", message)
            return 1
        doc, body = load_journey(journey_path)
        plan = plan_merge(
            state,
            doc,
            ffprobe_duration=ffprobe_duration,
            youtube_duration_lookup=_default_youtube_duration_lookup,
            allow_day_create=allow_day_create,
            now=now,
        )
        superlong_plan = plan_superlong_merge(
            state, load_superlong_sidecar(superlong_base), doc, now=now
        )
        plan.inserts.extend(superlong_plan.inserts)
        plan.deferrals.extend(superlong_plan.deferrals)
        plan.errors.extend(superlong_plan.errors)

        if plan.has_errors:
            print(plan.preview_text(), file=err)
            # Even on error, persist any *valid* deferrals so a future-dated video
            # isn't lost just because an unrelated LF failed (e.g. one LF's ffprobe
            # errored). The errored LFs are reported above; the operator resolves
            # them and re-runs for the rest. The journey YAML is left untouched.
            if apply and plan.deferrals:
                pending_path = pending_base / PENDING_FILENAME
                merge_pending(pending_path, _deferrals_to_pending(plan.deferrals, journey_path))
                print(
                    f"Persisted {len(plan.deferrals)} valid deferral(s) to {pending_path} "
                    "despite the errors above; resolve them and re-run for the rest.",
                    file=err,
                )
            return 1

        print(plan.preview_text(), file=out)

        if not apply:
            print("\n(dry-run; pass --apply to write)", file=out)
            return 0

        if not plan.inserts and not plan.days_to_create and not plan.deferrals:
            print("\nNo changes to write.", file=out)
            return 0

        # Write the journey YAML first (only the immediate inserts). Atomic: a
        # crash mid-write must leave the published page intact, never truncated.
        if plan.inserts or plan.days_to_create:
            apply_merge(doc, plan, library=VideoLibrary.for_site(site_path, writable=True))
            atomic_write_text(journey_path, serialize_journey(doc, body))
            print(f"\nWrote {journey_path}", file=out)
            _record_routes(site_path, journey_slug, plan.inserts)

    # Then persist deferrals to the pending sidecar (pending_base = the drive
    # output dir when the auto-hook routes it there, else next to state.json).
    if plan.deferrals:
        pending_path = pending_base / PENDING_FILENAME
        merge_pending(pending_path, _deferrals_to_pending(plan.deferrals, journey_path))
        print(
            f"Deferred {len(plan.deferrals)} entr"
            f"{'y' if len(plan.deferrals) == 1 else 'ies'} to {pending_path} "
            "(replay later with `gomoveshift publish-journal --replay-pending`)",
            file=out,
        )

    return 0


def run_replay_pending(
    *,
    pending_files: list[Path],
    apply: bool,
    auto_commit: bool = False,
    auto_push: bool = False,
    out=None,
    err=None,
    now: Optional[datetime] = None,
    live_lookup: Callable[[list[str]], dict[str, dict]] = lambda ids: {},
) -> int:
    """Flush any deferred entries whose ``publish_at`` has now passed.

    ``pending_files`` is the explicit list of sidecars to scan; the parser
    builds it either from ``--pending-file`` (one) or ``--scan-root``
    (recursive glob for ``pending_journal.json``). Empty list → exit 0: a
    scan root with nothing pending is the normal steady state, not a cron
    failure. A ``--scan-root`` that doesn't exist/isn't a dir is rejected
    upstream in ``cmd_publish_journal`` before this function ever runs, so
    that genuine operator error still exits non-zero.

    When ``auto_commit`` is set, journey files modified during the sweep
    are committed to their enclosing git repo (one commit per repo, not
    per journey). When ``auto_push`` is additionally set, the commit is
    pushed to ``origin``. A dirty tracked working tree skips that repo's
    writes entirely (skip-and-log) — never auto-stash, per design.

    ``live_lookup`` (``{video_id: {"title", "seconds"}}``; the CLI injects
    :func:`_default_youtube_live_lookup`) supplies each ready entry's title and
    duration as YouTube shows them now. The sidecar's values were stored at
    drive time: the video may have been retitled live since, or reported a
    zero duration while still processing. A title or duration the lookup
    cannot supply keeps the stored one.
    """
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    if auto_push and not auto_commit:
        print(
            "Error: --auto-push requires --auto-commit (cannot push without committing first)",
            file=err,
        )
        return 2
    if not pending_files:
        print(
            "No pending sidecars found (nothing due for replay). Pass "
            "--pending-file or --scan-root pointing at a dir containing "
            "pending_journal.json if you expected entries here.",
            file=out,
        )
        return 0

    if now is None:
        now = datetime.now(tz=timezone.utc)

    total_ready = 0
    total_still = 0
    failures: list[str] = []

    # Group by journey so each .md file is loaded/written exactly once even
    # if multiple sidecars point at the same journey.
    pending_by_journey: dict[Path, list[tuple[Path, PendingEntry]]] = {}
    for pending_path in pending_files:
        try:
            entries = load_pending(pending_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{pending_path}: failed to load ({exc})")
            continue
        ready, still = split_by_readiness(entries, now)
        total_still += len(still)
        for entry in ready:
            pending_by_journey.setdefault(Path(entry.journey_path), []).append(
                (pending_path, entry)
            )

    # Dirty-WT pre-check when auto-commit is on: any journey whose enclosing
    # git repo is dirty is dropped from this run BEFORE its sidecar entries
    # get consumed. Honors "skip and log, never auto-stash" — see
    # journal_git_ops module docstring.
    dirty_repos: set[Path] = set()
    if auto_commit:
        repo_by_journey: dict[Path, Optional[Path]] = {
            jp: git_toplevel(jp) for jp in pending_by_journey
        }
        for journey_path, repo_root in repo_by_journey.items():
            if repo_root is None:
                failures.append(
                    f"{journey_path}: not in a git working tree; "
                    "cannot --auto-commit. Drop the flag or fix the path."
                )
                pending_by_journey.pop(journey_path, None)
                continue
            if repo_root in dirty_repos:
                pending_by_journey.pop(journey_path, None)
                continue
            if is_tracked_dirty(repo_root):
                dirty_repos.add(repo_root)
                failures.append(
                    f"{repo_root}: tracked working tree is dirty; skipping all "
                    "journey writes in this repo (commit/stash your work, then "
                    "re-run)."
                )
                pending_by_journey.pop(journey_path, None)

    # Hold each remaining journey's write lock for the whole sweep, BEFORE the
    # first load: every journey writer takes it across its own load -> mutate ->
    # write window, so skipping it would serialize a doc that never saw theirs.
    # Same skip-and-log shape as the dirty-WT pre-check — dropped BEFORE its
    # entries can be consumed, so they stay pending. Non-blocking: no deadlock.
    locks = ExitStack()
    for journey_path in list(pending_by_journey):
        if journey_path.is_file() and not locks.enter_context(
            journey_lock(journey_path, skip_if_busy=True)
        ):
            failures.append(
                f"{journey_path}: another writer holds this journey's lock; "
                f"{len(pending_by_journey[journey_path])} entries kept pending"
            )
            pending_by_journey.pop(journey_path)

    # Per-repo commit groups: each writable journey file under its repo root.
    writes_by_repo: dict[Path, list[Path]] = {}
    flushed_count_by_repo: dict[Path, int] = {}
    # (journey_path_str, youtube_id) pairs durably written to (or already
    # present in) a journey file — and therefore safe to drop from the sidecar.
    # We consume the sidecar AFTER the writes settle, never before: a
    # missing-file, plan-error, or crashing journey keeps its entries so a
    # later clean run can still flush them (no consume-before-confirm loss).
    flushed_keys: set[tuple[str, str]] = set()

    ready_ids = sorted({e.youtube_id for pairs in pending_by_journey.values() for _p, e in pairs})
    live = live_lookup(ready_ids) if ready_ids else {}
    for pairs in pending_by_journey.values():
        for i, (pending_path, entry) in enumerate(pairs):
            meta = live.get(entry.youtube_id) or {}
            title = str(meta.get("title") or "") or entry.title
            seconds = float(meta.get("seconds") or 0)
            duration = format_duration(seconds) if seconds > 0 else entry.duration
            pairs[i] = (pending_path, replace(entry, title=title, duration=duration))

    for journey_path, entry_pairs in pending_by_journey.items():
        if not journey_path.is_file():
            failures.append(
                f"{journey_path}: journey file missing; cannot flush "
                f"{len(entry_pairs)} pending entries (kept pending)"
            )
            continue
        try:
            # Synthesize a state-shaped dict for plan_merge so we reuse all
            # the same cross-day collision + day-existence logic.
            synthetic_state = _entries_to_state(entry_pairs)
            doc, body = load_journey(journey_path)
            plan = plan_merge(
                synthetic_state,
                doc,
                ffprobe_duration=_inject_duration_passthrough(entry_pairs),
                allow_day_create=False,
                now=now,
            )
        except Exception as exc:  # malformed duration, unreadable journey, etc.
            failures.append(
                f"{journey_path}: replay processing failed ({exc}); "
                f"{len(entry_pairs)} entries kept pending"
            )
            continue

        if plan.has_errors:
            failures.append(
                f"{journey_path}: plan errors during replay:\n  " + "\n  ".join(plan.errors)
            )
            continue

        if apply and (plan.inserts or plan.days_to_create):
            apply_merge(doc, plan, library=VideoLibrary.for_journey(journey_path, writable=True))
            atomic_write_text(journey_path, serialize_journey(doc, body))
            print(
                f"Wrote {journey_path}: {len(plan.inserts)} entr"
                f"{'y' if len(plan.inserts) == 1 else 'ies'} flushed",
                file=out,
            )
            # journey_path is ``<site>/content/journeys/<slug>.md``.
            routing_path = _record_routes(journey_path.parents[2], journey_path.stem, plan.inserts)
            if auto_commit:
                repo_root = git_toplevel(journey_path)
                if repo_root is not None:
                    writes = writes_by_repo.setdefault(repo_root, [])
                    writes.append(journey_path)
                    # video_routing.yaml is tracked: commit it with the embed it
                    # reproduces, or the next sweep sees a dirty tree and skips.
                    if (
                        routing_path is not None
                        and routing_path not in writes
                        and routing_path.is_relative_to(repo_root)
                    ):
                        writes.append(routing_path)
                    flushed_count_by_repo[repo_root] = flushed_count_by_repo.get(
                        repo_root, 0
                    ) + len(plan.inserts)

        if apply:
            # This journey processed cleanly: every ready entry is now either
            # written (insert) or already present (skip). Safe to consume.
            for _src, entry in entry_pairs:
                flushed_keys.add((str(journey_path), entry.youtube_id))

        total_ready += len(plan.inserts)

    # Consume the sidecars AFTER writes settle: drop only the entries that were
    # durably flushed, preserving future entries and any kept pending above.
    if apply:
        for pending_path in pending_files:
            # Re-load AND save under the sidecar's own lock: a concurrent
            # writer (e.g. a live drive run's merge_pending) could otherwise
            # insert a new deferral between our earlier scan and this
            # consume, and our stale `keep` would drop it on save.
            with pending_lock(pending_path):
                try:
                    all_entries = load_pending(pending_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                keep = [
                    e
                    for e in all_entries
                    if (str(Path(e.journey_path)), e.youtube_id) not in flushed_keys
                ]
                save_pending(pending_path, keep)

    locks.close()  # every journey write and its sidecar consume has settled

    if not apply:
        print(
            f"(dry-run) would flush {total_ready} entries, "
            f"{total_still} still pending. Pass --apply to write.",
            file=out,
        )

    # Commit + push pass — runs ONCE per repo after all writes settle.
    if apply and auto_commit:
        for repo_root, journey_files in writes_by_repo.items():
            flushed = flushed_count_by_repo[repo_root]
            rel_files = ", ".join(str(p.relative_to(repo_root)) for p in journey_files)
            message = (
                f"chore(journeys): publish-journal flush ({flushed} "
                f"entr{'y' if flushed == 1 else 'ies'})\n\n"
                f"Files: {rel_files}\n"
                f"Triggered by: gomoveshift publish-journal --replay-pending "
                f"--apply --auto-commit"
                f"{' --auto-push' if auto_push else ''}\n"
                f"Sweep time (UTC): {now.isoformat()}"
            )
            try:
                result = commit_paths(repo_root, journey_files, message, push=auto_push)
            except GitOpError as exc:
                failures.append(f"{repo_root}: {exc}")
                continue
            if result.commit_sha:
                push_note = " + pushed" if result.pushed else ""
                print(
                    f"Committed {result.commit_sha[:7]} in {repo_root}{push_note}",
                    file=out,
                )
            else:
                print(
                    f"{repo_root}: nothing to commit ({result.skipped_reason})",
                    file=out,
                )

    if failures:
        for msg in failures:
            logger.warning("%s", msg)
        # auto-commit failures are operationally significant — cron alerting
        # should fire. Plan-error and dirty-WT warnings only exit non-zero
        # when auto_commit was requested, otherwise informational.
        if auto_commit:
            return 3

    return 0


def _entries_to_state(
    entry_pairs: list[tuple[Path, PendingEntry]],
) -> dict:
    """Build a ``youtube_uploads`` + ``per_lf_recording_date`` shape that
    ``plan_merge`` accepts.

    The sidecar already carries duration + title (computed once by the
    original pipeline run), so we bypass ffprobe via
    ``_inject_duration_passthrough``.
    """
    uploads: dict[str, dict] = {}
    rec_dates: dict[str, str] = {}
    titles: dict[str, str] = {}
    for i, (_path, entry) in enumerate(entry_pairs, start=1):
        # lf_index from the sidecar would collide across multiple sidecars;
        # re-index sequentially so plan_merge sees one consistent map.
        uploads[str(i)] = {
            "video_id": entry.youtube_id,
            "title": entry.title,
            # publish_at omitted — already past, do not re-defer.
        }
        rec_dates[str(i)] = entry.day_date
        titles[str(i)] = entry.title
    return {
        "youtube_uploads": uploads,
        "per_lf_recording_date": rec_dates,
        "per_lf_titles": titles,
        # plan_merge ffprobes ``regular_lf_paths`` for duration; we substitute
        # a passthrough callable, but the dict must still be non-empty.
        "regular_lf_paths": [f"__pending_lf_{i}__" for i in range(len(entry_pairs))],
    }


def _inject_duration_passthrough(
    entry_pairs: list[tuple[Path, PendingEntry]],
) -> Callable[[Path], float]:
    """Map the synthetic ``regular_lf_paths`` back to each entry's stored
    duration string so plan_merge reformats it identically."""
    from ...pipeline.journal_publisher import format_duration  # local: avoid cycle

    duration_by_synth_name = {
        f"__pending_lf_{i}__": _duration_to_seconds(entry.duration)
        for i, (_p, entry) in enumerate(entry_pairs)
    }

    def _probe(path: Path) -> float:
        # plan_merge calls Path(str(lf_path)); the synthetic strings round-trip.
        return duration_by_synth_name[str(path)]

    # format_duration is re-applied inside plan_merge from the float we return.
    _ = format_duration  # silence "unused import" linters
    return _probe


def _duration_to_seconds(formatted: str) -> float:
    """Inverse of :func:`format_duration` for the two shapes we emit:
    ``MM:SS`` and ``H:MM:SS``."""
    parts = formatted.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"unparseable duration: {formatted!r}")


def cmd_publish_journal(args, config):  # noqa: ARG001 -- config unused, signature parity
    """Argparse entry point for ``gomoveshift publish-journal``."""
    # Wire centralized logging so unattended
    # cron replay runs don't drop INFO/WARN output. Without this, the handler
    # falls through to module-level logger with no handler attached → silent.
    from ..logging_setup import configure_logging

    configure_logging(log_file=getattr(args, "log_file", None))

    if getattr(args, "replay_pending", False):
        scan_root = getattr(args, "scan_root", None)
        if scan_root and not Path(scan_root).resolve().is_dir():
            print(
                f"Error: --scan-root {scan_root!r} does not exist or is not a directory.",
                file=sys.stderr,
            )
            return 2
        pending_files = _gather_pending_files(args)
        return run_replay_pending(
            pending_files=pending_files,
            apply=bool(getattr(args, "apply", False)),
            auto_commit=bool(getattr(args, "auto_commit", False)),
            auto_push=bool(getattr(args, "auto_push", False)),
            live_lookup=_default_youtube_live_lookup,
        )
    missing: list[str] = []
    if not getattr(args, "state_path", None):
        missing.append("state_path")
    if not getattr(args, "journey", None):
        missing.append("--journey")
    if not getattr(args, "site_path", None):
        missing.append("--site-path")
    if missing:
        print(
            "Error: missing required argument(s) for merge mode: "
            + ", ".join(missing)
            + " (or pass --replay-pending for the sidecar-flush mode)",
            file=sys.stderr,
        )
        return 2
    return run_publish_journal(
        state_path=Path(args.state_path).resolve(),
        site_path=Path(args.site_path).resolve(),
        journey_slug=args.journey,
        apply=bool(getattr(args, "apply", False)),
        allow_day_create=bool(getattr(args, "allow_day_create", False)),
    )


def _gather_pending_files(args) -> list[Path]:
    explicit = getattr(args, "pending_file", None)
    scan_root = getattr(args, "scan_root", None)
    files: list[Path] = []
    if explicit:
        files.append(Path(explicit).resolve())
    if scan_root:
        root = Path(scan_root).resolve()
        if root.is_dir():
            files.extend(sorted(root.rglob(PENDING_FILENAME)))
    return files
