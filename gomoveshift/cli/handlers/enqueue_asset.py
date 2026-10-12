"""Handler for ``gomoveshift enqueue-asset``.

The standalone-asset door onto the delivery queue: a route-story film,
a photo export, or any other Waypost product with no drive run behind it
goes through here rather than ``enqueue-plan``. See
``docs/social-publishing-queue.md`` ("Standalone assets").

Dry-run by default, matching ``enqueue-plan``/``dispatch-queue``: without
``--apply`` nothing is written. ``--apply`` additionally requires
``--confirm <hash>`` to equal the review hash computed from the CURRENT
title/description/tags/file; a mismatch refuses rather than spooling stale
content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...publishing.asset_plan import build_asset_sidecar, write_asset_bundle
from ...publishing.enqueue_from_plan import enqueue_from_plan_sidecar
from ...publishing.media_fingerprint import MediaVerificationError, verify_media
from ...publishing.plan_io import load_publish_plan
from ...publishing.queue import open_queue_db


def _load_metadata(args) -> dict:
    """Metadata from ``--metadata <json>`` (expected shape:
    ``title, description, tags[], recording_date``) or from the individual
    ``--title``/``--description``/``--tag`` flags. ``recording_date``
    defaults to the date portion of ``--at`` when not supplied."""
    if args.metadata:
        try:
            raw = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        except OSError as exc:
            raise SystemExit(f"Error: --metadata unreadable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Error: --metadata is not valid JSON: {exc}") from exc
        title = raw.get("title")
        description = raw.get("description", "")
        tags = raw.get("tags") or []
        recording_date = raw.get("recording_date") or args.at[:10]
    else:
        title = args.title
        description = args.description or ""
        tags = args.tag or []
        recording_date = args.at[:10]
    if not title:
        raise SystemExit(
            "Error: a title is required -- pass --title or --metadata with a 'title' key."
        )
    return {
        "title": title,
        "description": description,
        "tags": list(tags),
        "recording_date": recording_date,
    }


def _load_privacy_attestation(receipt_path) -> dict | None:
    if not receipt_path:
        return None
    try:
        raw = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Error: --receipt unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: --receipt is not valid JSON: {exc}") from exc
    return raw.get("privacy")


def cmd_enqueue_asset(args, config) -> int:
    video = Path(args.video).resolve()
    metadata = _load_metadata(args)
    privacy_attestation = _load_privacy_attestation(args.receipt)

    try:
        doc = build_asset_sidecar(
            video,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            recording_date=metadata["recording_date"],
            platforms=args.platform,
            scheduled_at_utc=args.at,
            display_timezone=args.display_timezone,
            privacy_attestation=privacy_attestation,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    record = doc["shorts"][0]
    review = record["reviewed_plan_hash"]
    media = record["media"]

    print(f"Asset: {video}")
    print(
        f"  media: {media['width']}x{media['height']} {media['codec']} {media['duration_seconds']:.1f}s"
    )
    for platform in args.platform:
        payload = record["payloads"][platform]
        print(f"  {platform}: {payload}")
    print(f"review hash: {review}")

    if not args.apply:
        print("\nDRY RUN - nothing was written. Re-run with --apply --confirm <hash>.")
        return 0

    if args.confirm != review:
        print(
            "Error: content changed since review -- the current review hash does not match "
            "--confirm. Re-run without --apply to see the current hash, then re-apply with it."
        )
        return 2

    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else video.parent
    sidecar_path, _attestation_path = write_asset_bundle(bundle_dir, doc, privacy_attestation)

    try:
        verify_media(load_publish_plan(sidecar_path).shorts[0])
    except MediaVerificationError as exc:
        print(f"Error: {exc}")
        return 1

    conn = open_queue_db(config.seo.storage_path)
    try:
        result = enqueue_from_plan_sidecar(conn, sidecar_path, shadow=bool(args.shadow))
    finally:
        conn.close()

    suffix = " (SHADOW - can never publish)" if args.shadow else ""
    print(
        f"\nenqueued={result.enqueued_count} already_present={result.already_present_count}{suffix}"
    )
    for project_name, short_index, platform, delivery_id in result.enqueued:
        print(f"  id={delivery_id} platform={platform}")
    return 0


def build_enqueue_asset_argv(
    video: str,
    *,
    platforms,
    at: str,
    title: str,
    description: str,
    confirm: str,
    tags=None,
    display_timezone: str = "",
    receipt: str = "",
    bundle_dir: str = "",
    shadow: bool = False,
) -> "list[str]":
    """Argv for ``gomoveshift enqueue-asset <video> --apply [...]``, matching
    ``cli/parser.py::_add_enqueue_asset_parser`` -- the console's form always
    applies with an operator-supplied confirm hash."""
    argv = [sys.executable, "-m", "gomoveshift", "enqueue-asset", str(video)]
    argv += ["--at", at, "--title", title, "--description", description]
    argv += ["--confirm", confirm, "--apply"]
    for platform in platforms:
        argv += ["--platform", platform]
    for tag in tags or []:
        argv += ["--tag", tag]
    if display_timezone:
        argv += ["--display-timezone", display_timezone]
    if receipt:
        argv += ["--receipt", str(receipt)]
    if bundle_dir:
        argv += ["--bundle-dir", str(bundle_dir)]
    if shadow:
        argv.append("--shadow")
    return argv
