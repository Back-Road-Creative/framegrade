"""Handler for ``gomoveshift sync-site``.

Makes the Hugo site's ``/videos/`` library a generated mirror of the channel's
real long-forms so no published long-form can go missing: enumerate the channel
into a manifest, UPSERT ``data/videos.yaml`` (preserving human-authored fields),
and write ``data/channel_manifest.json``. Dry-run by default; ``--apply`` writes.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from ...pipeline.journal_publisher import journey_lock
from ...pipeline.site_sync.channel_manifest import build_channel_manifest, write_manifest
from ...pipeline.site_sync.journey_reconcile import reconcile_journeys
from ...pipeline.site_sync.videos_yaml import (
    dump_videos_yaml,
    load_videos_yaml,
    upsert_videos_yaml,
)


def run_sync_site(
    *,
    site_path: Path,
    apply: bool,
    journeys: bool = False,
    credentials_dir: Path | None = None,
    service=None,
    now: datetime | None = None,
    out=None,
    err=None,
) -> int:
    """Returns the CLI exit code. ``service`` and ``now`` are injectable so the
    happy path is unit-testable without live credentials."""
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    site_path = Path(site_path)
    if not site_path.is_dir():
        print(f"Error: site path is not a directory: {site_path}", file=err)
        return 2
    if now is None:
        now = datetime.now(tz=timezone.utc)

    if service is None:
        # Wrap the import + build so a missing creds/libs surface is a clean
        # message, not a traceback. Kept out of the unit test (which injects
        # ``service``), so live credentials are never touched in CI.
        try:
            from ...pipeline.drive_phases.phase_d_scheduling import (
                _build_readonly_youtube_service,
            )

            service = _build_readonly_youtube_service(credentials_dir)
        except Exception as exc:
            print(
                f"Error: could not build the read-only YouTube client ({exc}). "
                "Check SOPS credentials / google-api libs.",
                file=err,
            )
            return 2

    manifest = build_channel_manifest(service, now=now)
    counts = manifest.get("counts", {})

    videos_path = site_path / "data" / "videos.yaml"
    manifest_path = site_path / "data" / "channel_manifest.json"

    # Hold the lock across the whole load -> upsert -> write window: a cron
    # site-sync and a live drive run's own sync can otherwise overlap, and
    # whichever serializes second writes back a snapshot that never saw the
    # other's new row — a silent lost update. Busy is a recoverable refusal,
    # not an exception (same convention as the journal merge's journey_lock).
    with journey_lock(videos_path, skip_if_busy=True) as held:
        if not held:
            message = (
                f"Error: another writer holds the lock on {videos_path}; nothing "
                "was written. Re-run once it finishes."
            )
            print(message, file=err)
            return 1

        existing = load_videos_yaml(videos_path)
        existing_ids = {v.get("id") for v in (existing or {}).get("videos") or [] if v.get("id")}
        library = upsert_videos_yaml(manifest, existing)
        real_lf_ids = [
            v["id"] for v in manifest.get("videos") or [] if v.get("category") == "real_lf"
        ]
        known_ids = sorted({v["id"] for v in manifest.get("videos") or [] if v.get("id")})
        added = [vid for vid in real_lf_ids if vid not in existing_ids]

        print(
            f"Channel: {counts.get('total', 0)} videos "
            f"({counts.get('real_lf', 0)} real_lf, {counts.get('short', 0)} short, "
            f"{counts.get('stream', 0)} stream, {counts.get('held', 0)} held).",
            file=out,
        )
        print(
            f"Library: {len(library['videos'])} entries "
            f"({len(added)} added, {len(existing_ids)} preserved).",
            file=out,
        )

        if journeys:
            verb = "embedded" if apply else "would embed"
            rec = reconcile_journeys(manifest, site_path, apply=apply)
            for slug, vid in rec.embedded:
                print(f"  + {slug}: {verb} {vid}", file=out)
            print(
                f"Journeys: {len(rec.embedded)} {verb}, {len(rec.skipped_present)} present, "
                f"{len(rec.deferred_held)} deferred (held).",
                file=out,
            )
            for w in rec.warnings:
                print(f"  ! {w}", file=err)

        if not apply:
            print(
                "\n(dry-run; pass --apply to write data/videos.yaml + data/channel_manifest.json)",
                file=out,
            )
            return 0

        # Commit a SLIM manifest — the full uploads list is hundreds of entries. Shape:
        # ``{generated_at, source, counts, real_lf_ids, known_ids}``.
        #   * ``real_lf_ids`` — the public long-forms videos.yaml must cover (hard gate).
        #   * ``known_ids`` — EVERY id the channel accounts for, whatever its category,
        #     sorted. The library is long-form-only, so without it the site's coverage
        #     gate cannot tell a legitimately embedded Short/stream/held clip from an id
        #     that exists nowhere (a typo'd or deleted embed), and has to warn on both.
        #     Category-agnostic on purpose: a new category needs no new key. Sorted (not
        #     publish order) because it is a membership set and a stable order keeps the
        #     daily sync from producing a spurious diff.
        slim_manifest = {
            "generated_at": manifest["generated_at"],
            "source": manifest.get("source", ""),
            "counts": counts,
            "real_lf_ids": real_lf_ids,
            "known_ids": known_ids,
        }
        # write_manifest is a pure overwrite (no read-modify merge, so no lost-
        # update to lose), but it stays inside the same critical section as the
        # videos.yaml write so the two on-disk artifacts one sync run produces
        # never straddle a concurrent writer's own window.
        write_manifest(slim_manifest, manifest_path)
        dump_videos_yaml(library, videos_path)
        print(f"\nWrote {videos_path}\nWrote {manifest_path}", file=out)
        return 0


def cmd_sync_site(args, config):  # noqa: ARG001 -- config unused, signature parity
    """Argparse entry point for ``gomoveshift sync-site``."""
    return run_sync_site(
        site_path=Path(args.site_path),
        apply=bool(getattr(args, "apply", False)),
        journeys=bool(getattr(args, "journeys", False)),
        credentials_dir=getattr(args, "credentials_dir", None),
    )
