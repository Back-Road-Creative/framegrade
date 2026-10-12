"""Handler for ``gomoveshift reconcile-playlists``.

Runs the daily playlist reconciler: desired membership = f(channel
classification), inserting anything missing into the target playlists, capped
per run by ``playlist_reconciler.max_inserts_per_run``. Dry-run by default;
``--apply`` executes the inserts. Without ``--prune`` that is ALL it ever does
(the nightly cron), and a re-run is idempotent, so the cron caller treats a
mid-run failure as "resume next fire".

``--prune`` also plans removals (dry-run unless ``--apply``, at most
``--max-deletes`` filled in class order deleted, unsorted, misfiled, nonpublic, each
ledgered first; private/unlisted entries go last, governed by config
``prune_include_nonpublic`` (``--[no-]include-nonpublic`` overrides) and a scheduled
upload (private, future ``publishAt``) is never pruned, skipped ``scheduled``; the one
exception to "last": a private/unlisted entry in ``Unsorted (fallback)`` is removed as
``unsorted`` (first class wins); the last output line is always ``Prune: planned/applied/skipped/
deferred/errors`` so the cron's ``tail -1`` logs the counts); ``--undo LEDGER`` re-inserts a ledger's
entries; ``--audit`` prints ``RECONCILE_AUDIT key=count`` lines. The three are
mutually exclusive; ``--undo``/``--audit`` ignore ``--apply``.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from ...paths import gms_output_dir


def default_ledger_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return gms_output_dir() / "reconcile-playlists" / f"{stamp}-prune-ledger.jsonl"


def run_reconcile_playlists(
    *,
    apply: bool,
    reconciler_config,
    credentials_dir: Path | None = None,
    service=None,
    now: datetime | None = None,
    out=None,
    err=None,
    prune: bool = False,
    audit: bool = False,
    undo: Path | None = None,
    max_deletes: int = 150,
    include_nonpublic: bool | None = None,
    ledger: Path | None = None,
) -> int:
    """Returns the CLI exit code. ``service`` and ``now`` are injectable so the
    happy path is unit-testable without live credentials."""
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    if service is None:
        # Wrap the import + build so a missing creds/libs surface is a clean
        # message, not a traceback. Kept out of the unit test (which injects
        # ``service``), so live credentials are never touched in CI. Same
        # SOPS creds path as the uploader — the stored grant carries the
        # write scope the inserts need.
        try:
            from googleapiclient.discovery import build

            from ...pipeline._yt_creds import load_youtube_credentials
            from ...pipeline.drive_phases.youtube_uploader import _build_credentials_obj

            creds = _build_credentials_obj(load_youtube_credentials(credentials_dir))
            service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        except Exception as exc:
            print(
                f"Error: could not build the YouTube client ({exc}). "
                "Check SOPS credentials / google-api libs.",
                file=err,
            )
            return 2

    from ...pipeline import playlist_reconciler

    if undo:
        result = playlist_reconciler.undo_deletes(service, Path(undo))
        counts = " ".join(f"{k}={result[k]}" for k in ("restored", "skipped", "errors"))
        print(f"Undo: {counts} ledger={undo}", file=out)
        return 1 if result["errors"] else 0

    extra = {}
    if prune or audit:
        extra = dict(
            prune=prune, audit=audit, max_deletes=max_deletes, include_nonpublic=include_nonpublic
        )
        if prune and apply:
            extra["ledger"] = ledger or default_ledger_path()
    summary = playlist_reconciler.reconcile(
        service, reconciler_config, apply=apply, now=now, **extra
    )
    if audit:
        for key, count in summary["audit"].items():
            print(f"RECONCILE_AUDIT {key}={count}", file=out)
        return 0
    unrouted = summary.get("unrouted") or []
    named = ""
    if unrouted:
        cap = playlist_reconciler.UNROUTED_LOG_CAP
        ids = ",".join(row["video_id"] for row in unrouted[:cap])
        more = f",\u2026 and {len(unrouted) - cap} more" if len(unrouted) > cap else ""
        named = f" [{ids}{more}]"
    print(
        f"Reconcile: planned={summary['planned']} applied={summary['applied']} "
        f"skipped={summary['skipped']} unroutable={summary['unroutable']}{named}",
        file=out,
    )
    if prune:
        pruned = summary["prune"]
        for row in pruned["rows"]:
            print(
                f"RECONCILE_PRUNE {row['reason']} playlist={row['playlist_title']!r} "
                f"video={row['video_id']} item={row['item_id']} position={row['position']}",
                file=out,
            )
        counts = " ".join(
            f"{k}={pruned[k]}" for k in ("planned", "applied", "skipped", "deferred", "errors")
        )
        # Notes first: the cron logs ``tail -1``, which must be the counts line.
        if apply:
            print(f"(ledger {extra['ledger']}; restore with --undo)", file=out)
        else:
            print("(dry-run; pass --apply to execute the planned inserts and deletes)", file=out)
        print(f"Prune: {counts}", file=out)
    elif not apply:
        print("(dry-run; pass --apply to execute the planned inserts)", file=out)
    return 1 if prune and summary["prune"]["errors"] else 0


def cmd_reconcile_playlists(args, config):
    """Argparse entry point for ``gomoveshift reconcile-playlists``."""
    return run_reconcile_playlists(
        apply=bool(getattr(args, "apply", False)),
        reconciler_config=config.playlist_reconciler,
        credentials_dir=getattr(args, "credentials_dir", None),
        prune=bool(getattr(args, "prune", False)),
        audit=bool(getattr(args, "audit", False)),
        undo=getattr(args, "undo", None),
        max_deletes=int(getattr(args, "max_deletes", 150)),
        include_nonpublic=getattr(args, "include_nonpublic", None),
        ledger=getattr(args, "ledger", None),
    )
