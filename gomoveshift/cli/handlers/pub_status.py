"""Handler for ``gomoveshift publishing-status``.

Ground-truth status of the first-party social publishing stack: per
platform, whether its credential file is PRESENT on disk (existence only,
NEVER contents — no decrypt, no read) and the delivery queue's row counts by
state. See ``docs/social-publishing-status.md`` for the full evidence-cited
table this command's presence check summarizes.

``--validate`` adds the other half: after restoring a backup into an
isolated instance, cross-check what the queue rows DEPEND on rather than only
counting them. Read-only by construction — it opens no adapter, imports
nothing from ``dispatch``, and issues no write — so it is safe to run on a
restore before any dispatcher is allowed near it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ...publishing.plan_io import PublishPlanFormatError, load_publish_plan
from ...publishing.queue import TERMINAL_STATUSES, open_queue_db

#: platform -> credential filename, resolved against
#: ``Path(seo.credentials_dir).parent`` — the same convention as
#: ``_yt_creds.load_youtube_credentials``, ``meta_auth.load_meta_credentials``,
#: and ``tiktok_auth.load_tiktok_credentials``. Instagram and Facebook share
#: one Meta credential file.
_CREDENTIAL_FILENAMES = {
    "youtube": "youtube_upload_token.json.enc",
    "instagram": "meta_publishing.json.enc",
    "facebook": "meta_publishing.json.enc",
    "tiktok": "tiktok_publishing.json.enc",
}

_PLATFORM_ORDER = ("youtube", "instagram", "facebook", "tiktok")


def credential_presence(credentials_dir) -> Dict[str, bool]:
    """``{platform: file exists}`` for every known platform. Checks existence
    only — never opens, decrypts, or reads the file."""
    base = Path(credentials_dir).parent
    return {platform: (base / name).exists() for platform, name in _CREDENTIAL_FILENAMES.items()}


def queue_state_counts(conn) -> Dict[str, int]:
    """``{status: row count}`` across the whole delivery queue (every
    platform, shadow rows included)."""
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM deliveries GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


@dataclass(frozen=True)
class RestoreProblem:
    """One reason a restored delivery cannot do what its row claims it will.

    ``kind`` is a stable machine-readable token so a restore script can grep
    for a class of failure; ``detail`` is the human sentence.
    """

    delivery_id: int
    platform: str
    kind: str
    detail: str


def _sidecar_problem(row, plan_cache: Dict[str, object]) -> "Optional[RestoreProblem]":
    """The queue DB and the publish-plan sidecars are separate files that a
    restore can easily bring back one of. A row whose sidecar is missing, or
    is a different run's, cannot publish -- and nothing says so today until a
    drain pass fails on it, long after the restore was called good."""
    sidecar_path = row["sidecar_path"]
    if not sidecar_path:
        return RestoreProblem(
            row["id"],
            row["platform"],
            "sidecar_path_unset",
            "no publish-plan sidecar recorded, so the delivery cannot be read back",
        )
    if sidecar_path not in plan_cache:
        try:
            plan_cache[sidecar_path] = load_publish_plan(Path(sidecar_path))
        except PublishPlanFormatError as exc:
            plan_cache[sidecar_path] = exc
    plan = plan_cache[sidecar_path]
    if isinstance(plan, PublishPlanFormatError):
        return RestoreProblem(row["id"], row["platform"], "sidecar_unreadable", str(plan))
    match = any(
        short.project_name == row["project_name"] and short.short_index == row["short_index"]
        for short in plan.shorts
    )
    if not match:
        return RestoreProblem(
            row["id"],
            row["platform"],
            "short_not_in_sidecar",
            f"{sidecar_path} carries no short {row['project_name']}#{row['short_index']}",
        )
    return None


def validate_restored_state(conn, credentials_dir, *, now: "Optional[str]" = None) -> List:
    """Everything a NON-TERMINAL delivery depends on, checked against the
    files this instance actually has. Returns the problems, oldest delivery
    first; an empty list means the restore is coherent.

    Terminal rows are skipped deliberately: a published or cancelled delivery
    owes no further work, so a sidecar it no longer needs is not a restore
    defect, and listing it would bury the rows that DO still need one.

    Never writes. That is the "with dispatch disabled" half of the acceptance
    criterion -- a validator that could mutate the queue would be one more
    thing to trust rather than the thing that establishes trust.
    """
    now = now or datetime.now(timezone.utc).isoformat()
    presence = credential_presence(credentials_dir)
    placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
    rows = conn.execute(
        f"SELECT * FROM deliveries WHERE status NOT IN ({placeholders}) ORDER BY id",  # noqa: S608
        TERMINAL_STATUSES,
    ).fetchall()

    problems: List[RestoreProblem] = []
    plan_cache: Dict[str, object] = {}
    for row in rows:
        sidecar_problem = _sidecar_problem(row, plan_cache)
        if sidecar_problem is not None:
            problems.append(sidecar_problem)
        if not presence.get(row["platform"], False):
            problems.append(
                RestoreProblem(
                    row["id"],
                    row["platform"],
                    "credential_absent",
                    f"no credential file for {row['platform']}; the first drain would fail",
                )
            )
        if row["lease_owner"] and (row["lease_expires_at"] or "") > now:
            # The worker that took this lease died with the instance. Until it
            # expires, queue._has_live_foreign_lease refuses the operator's
            # cancel and reschedule, and the row reads as busy work in flight.
            problems.append(
                RestoreProblem(
                    row["id"],
                    row["platform"],
                    "stale_lease",
                    f"held by {row['lease_owner']} until {row['lease_expires_at']}; "
                    "no such worker is running after a restore",
                )
            )
    return problems


def cmd_publishing_status(args, config) -> int:
    presence = credential_presence(config.seo.credentials_dir)
    print("credential files present (existence only, never contents):")
    for platform in _PLATFORM_ORDER:
        print(f"  {platform}: {'present' if presence[platform] else 'ABSENT'}")

    conn = open_queue_db(config.seo.storage_path)
    try:
        counts = queue_state_counts(conn)
    finally:
        conn.close()

    print("\ndelivery queue row counts by state:")
    if counts:
        for status in sorted(counts):
            print(f"  {status}: {counts[status]}")
    else:
        print("  (empty)")

    if not getattr(args, "validate", False):
        # Without the flag this stays a report. A broken restore must not
        # start failing a status check that never claimed to validate.
        return 0

    conn = open_queue_db(config.seo.storage_path)
    try:
        problems = validate_restored_state(conn, config.seo.credentials_dir)
    finally:
        conn.close()

    print("\nrestore validation (read-only; nothing is dispatched):")
    if not problems:
        print("  no problems found")
        return 0
    for problem in problems:
        print(
            f"  delivery {problem.delivery_id} ({problem.platform}) {problem.kind}: {problem.detail}"
        )
    return 1
