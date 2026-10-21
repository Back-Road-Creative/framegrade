#!/usr/bin/env python3
"""Apply approved retitles from a ``retitle_worklist.py`` CSV to live YouTube
videos. Contract: docs/pipeline-internals.md.

Every live change is snapshotted to the audit log first, read back after the
write (only a verified row is APPLIED), recorded in a reversal CSV
(``<worklist>.revert.csv``) and safe to re-run (ALREADY_APPLIED). ``--revert
PATH`` restores the prior text from such a reversal CSV; it runs NO place or
evidence gates, because it puts back the exact text that was live before."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gomoveshift.pipeline.location_evidence import (  # noqa: E402
    LocationEvidence,
    title_claim_issues,
)
from gomoveshift.pipeline.text_format import hashtags_in  # noqa: E402
from gomoveshift.pipeline.validation.content_gates import find_grammar_issues  # noqa: E402
from gomoveshift.pipeline.validation.place_corrections import (  # noqa: E402
    PlaceCorrections,
    find_denylisted,
    load_place_corrections,
)

MAX_TITLE_LEN = 100
MAX_DESCRIPTION_BYTES = 5000  # YouTube snippet.description limit
REFUSED, APPROVED, APPLIED, SKIPPED_STALE, ERROR = (
    "REFUSED",
    "APPROVED",
    "APPLIED",
    "SKIPPED_STALE",
    "ERROR",
)
# ALREADY_APPLIED = live text already the target (no call); VERIFY_FAILED =
# the read-back after a write did not match; SNAPSHOT = pre-write live text.
ALREADY_APPLIED, VERIFY_FAILED, SNAPSHOT = "ALREADY_APPLIED", "VERIFY_FAILED", "SNAPSHOT"


def read_worklist(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def evaluate_row(row: dict[str, str], corrections: PlaceCorrections | None = None) -> list[str]:
    """Gate-failure reasons for ``row``'s suggested title (empty = clean)."""
    title = (row.get("suggested_title") or "").strip()
    if not title or title == "needs-human":
        return ["no usable suggested title"]
    issues = find_grammar_issues(title)
    if len(title) > MAX_TITLE_LEN:
        issues.append(f"title exceeds {MAX_TITLE_LEN} chars ({len(title)})")
    corrections = corrections if corrections is not None else load_place_corrections()
    issues += [f"denylisted place name: '{n}'" for n in find_denylisted(title, corrections)]
    return issues + _description_issues(row, corrections)


def _description_issues(row: dict[str, str], corrections: PlaceCorrections) -> list[str]:
    """Gate-failure reasons for ``row``'s optional suggested description. A row
    without one keeps its live description untouched."""
    desc = row.get("suggested_description") or ""
    if not desc.strip():
        return []
    if not (row.get("current_description") or "").strip():
        return ["suggested_description needs current_description (the drift check)"]
    issues = find_grammar_issues(desc)
    size = len(desc.encode("utf-8"))
    if size > MAX_DESCRIPTION_BYTES:
        issues.append(f"description exceeds {MAX_DESCRIPTION_BYTES} bytes ({size})")
    if "<" in desc or ">" in desc:
        issues.append("description contains an angle bracket (YouTube rejects < and >)")
    issues += [
        f"denylisted place name in description: '{n}'" for n in find_denylisted(desc, corrections)
    ]
    return issues


def evidence_issues(row: dict[str, str]) -> list[str]:
    """Why the row's suggested title claims places its ``location_evidence``
    record (the segment's GPS, as JSON) does not support. Fails closed: a row
    with no record is refused, so an operator title cannot ship on its own say."""
    raw = (row.get("location_evidence") or "").strip()
    try:
        record = LocationEvidence.from_dict(json.loads(raw)) if raw else None
    except (ValueError, KeyError, TypeError):
        return ["location_evidence is not a valid evidence record"]
    return title_claim_issues((row.get("suggested_title") or "").strip(), record)


def hashtag_issues(row: dict[str, str], live_description: str | None = None) -> list[str]:
    """Why the suggested title carries a hashtag its video's description will
    not: the same rule as the publish-plan gate (``plan_io._assert_consistent``),
    applied before the write instead of when the plan sidecar is re-parsed.

    The description checked is the one the video will have afterwards: the
    row's ``suggested_description`` when present, else ``live_description``
    (fetched at apply), else -- dry run only -- the row's ``current_description``.
    With none of them there is nothing to check yet; apply checks it."""
    title = (row.get("suggested_title") or "").strip()
    desc: str | None = row.get("suggested_description") or ""
    if not (desc or "").strip():
        desc = live_description if live_description is not None else row.get("current_description")
    if not (desc or "").strip() and live_description is None:
        return []  # dry run with no description to read: apply checks the live one
    described = {t.lower() for t in hashtags_in(desc or "")}
    missing = [t for t in hashtags_in(title) if t.lower() not in described]
    if not missing:
        return []
    return [f"title hashtag(s) {missing} absent from the description the video will have"]


def dry_run(rows, corrections: PlaceCorrections | None = None, out=None) -> list[dict]:
    out = out if out is not None else sys.stdout
    corrections = corrections if corrections is not None else load_place_corrections()
    results = []
    for row in rows:
        # Evidence is checked here, not in evaluate_row: retitle_worklist runs
        # evaluate_row on candidates it composes before any record exists.
        issues = evaluate_row(row, corrections) + evidence_issues(row) + hashtag_issues(row)
        verdict = REFUSED if issues else APPROVED
        old, new = row.get("current_title", ""), row.get("suggested_title", "")
        print(f"{row.get('video_id')}: {old!r} -> {new!r} [{verdict}]", file=out)
        if (row.get("suggested_description") or "").strip():
            print(f"  description -> {row['suggested_description']!r}", file=out)
        for issue in issues:
            print(f"  - {issue}", file=out)
        results.append({"row": row, "verdict": verdict})
    return results


def _audit_row(video_id: str, old: str, new: str, result: str, **descriptions: str) -> dict:
    return {
        "video_id": video_id,
        "old": old,
        "new": new,
        **descriptions,
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


READBACK_ATTEMPTS = 3
READBACK_WAIT_S = 2.0
REVERT_FIELDS = (
    "video_id",
    "current_title",
    "restore_title",
    "current_description",
    "restore_description",
)


def _live_snippet(service, vid: str) -> dict | None:
    items = service.videos().list(part="snippet", id=vid).execute().get("items") or []
    return items[0]["snippet"] if items else None


def _push_verified(
    service, vid: str, snippet: dict, changes: dict[str, str], audit_path: Path | None, sleep
) -> tuple[str, dict[str, str]]:
    """Snapshot the live text to the audit log, send ``changes`` (one retry),
    then read the snippet back (up to READBACK_ATTEMPTS reads, ``sleep`` between
    them, since a read right after a write can lag). Returns (status, readback)."""
    if audit_path is not None:  # file only: the returned records stay outcome-only
        write_audit_log(
            [
                _audit_row(
                    vid,
                    snippet.get("title") or "",
                    changes.get("title", ""),
                    SNAPSHOT,
                    snapshot_title=snippet.get("title") or "",
                    snapshot_description=snippet.get("description") or "",
                )
            ],
            audit_path,
        )
    request = service.videos().update(
        part="snippet", body={"id": vid, "snippet": {**snippet, **changes}}
    )
    try:
        try:
            request.execute()
        except Exception:
            request.execute()  # one retry
    except Exception as exc:
        return ERROR, {"error": str(exc)}
    readback: dict[str, str] = {}
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            sleep(READBACK_WAIT_S)
        live = _live_snippet(service, vid) or {}
        readback = {
            "readback_title": (live.get("title") or "").strip(),
            **(
                {"readback_description": (live.get("description") or "").strip()}
                if "description" in changes
                else {}
            ),
        }
        if all(readback[f"readback_{k}"] == v.strip() for k, v in changes.items()):
            return APPLIED, readback
    return VERIFY_FAILED, readback


def _recorder(records: list[dict], audit_path: Path | None):
    def audit(record: dict) -> None:
        records.append(record)
        if audit_path is not None:
            write_audit_log([record], audit_path)

    return audit


def _append_revert(path: Path, row: dict[str, str]) -> None:
    """Append one reversal row (header on first write), flushed to disk."""
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVERT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _sync_plan(plan_sync, vid: str, title: str, out, err) -> None:
    """Bring the publish-plan sidecar(s) of ``vid`` to the title now live, in
    the same step that recorded it. A failure never undoes the live write; it
    is loud and names the command that finishes the job."""
    if plan_sync is None:
        return
    try:
        for change in plan_sync(vid, title):
            print(f"{vid}: plan {change.sidecar_path} title -> {title!r}", file=out)
    except Exception as exc:
        print(
            f"{vid}: WARNING plan title NOT synced ({exc}) -- run `gomoveshift queue "
            "sync-plan-titles --audit <this audit log> --apply`",
            file=err,
        )


def apply_updates(
    service,
    results: list[dict],
    out=None,
    err=None,
    *,
    audit_path: Path | None = None,
    revert_path: Path | None = None,
    sleep=time.sleep,
    plan_sync=None,
) -> list[dict]:
    """Push each APPROVED row's title -- and its description, when the row
    carries one -- live in one update. One retry per call; stops on the first
    unrecovered API error or failed read-back.

    A row whose live text already equals the suggestion is ALREADY_APPLIED (no
    call). Before each write the live title/description are appended to
    ``audit_path`` as a SNAPSHOT record (not in the returned list); after it the snippet is read back and
    only a verified row is APPLIED (and appended to ``revert_path``)."""
    out, err = out or sys.stdout, err or sys.stderr
    records: list[dict] = []
    audit = _recorder(records, audit_path)
    for result in results:
        if result["verdict"] != APPROVED:
            continue
        row = result["row"]
        vid = row.get("video_id", "")
        old_title, new_title = row.get("current_title", ""), row.get("suggested_title", "")
        new_desc = row.get("suggested_description") or ""
        descs = (
            {"old_description": row.get("current_description", ""), "new_description": new_desc}
            if new_desc.strip()
            else {}
        )
        gps_issues = evidence_issues(row)
        if gps_issues:  # re-checked here: never push a title the GPS no longer backs
            print(f"{vid}: REFUSED at apply -- {'; '.join(gps_issues)}", file=err)
            audit(_audit_row(vid, old_title, new_title, REFUSED, **descs))
            continue
        snippet = _live_snippet(service, vid)
        if snippet is None:
            print(f"{vid}: not found on YouTube -- skipping", file=err)
            continue
        live_title = (snippet.get("title") or "").strip()
        live_desc = (snippet.get("description") or "").strip()
        if live_title == new_title.strip() and (not descs or live_desc == new_desc.strip()):
            print(f"{vid}: already applied -- skipping", file=out)
            audit(_audit_row(vid, old_title, new_title, ALREADY_APPLIED, **descs))
            _sync_plan(plan_sync, vid, new_title, out, err)
            continue
        if live_title != old_title.strip():
            print(f"{vid}: STALE ROW -- live title differs -- skipping", file=err)
            audit(_audit_row(vid, old_title, new_title, SKIPPED_STALE, **descs))
            continue
        if descs and live_desc != descs["old_description"].strip():
            print(f"{vid}: STALE ROW -- live description differs -- skipping", file=err)
            audit(_audit_row(vid, old_title, new_title, SKIPPED_STALE, **descs))
            continue
        tag_issues = hashtag_issues(row, live_desc)
        if tag_issues:  # the description the video will have, known only now
            print(f"{vid}: REFUSED at apply -- {'; '.join(tag_issues)}", file=err)
            audit(_audit_row(vid, old_title, new_title, REFUSED, **descs))
            continue
        changes = {"title": new_title, **({"description": new_desc} if descs else {})}
        status, extra = _push_verified(service, vid, snippet, changes, audit_path, sleep)
        audit(_audit_row(vid, old_title, new_title, status, **descs, **extra))
        if status == ERROR:
            print(f"{vid}: API error ({extra['error']}) -- stopping", file=err)
            break
        if status == VERIFY_FAILED:
            print(f"{vid}: VERIFY FAILED -- live text differs after write -- stopping", file=err)
            break
        print(f"{vid}: applied {old_title!r} -> {new_title!r}", file=out)
        _sync_plan(plan_sync, vid, new_title, out, err)
        if revert_path is not None:
            _append_revert(
                revert_path,
                {
                    "video_id": vid,
                    "current_title": new_title,
                    "restore_title": snippet.get("title") or "",
                    "current_description": new_desc if descs else "",
                    "restore_description": (snippet.get("description") or "") if descs else "",
                },
            )
    return records


def read_revert_file(path: Path) -> list[dict[str, str]] | None:
    """Rows of a reversal CSV, or None when ``path`` is not in that format."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != REVERT_FIELDS:
            return None
        return list(reader)


def revert_updates(
    service,
    rows: list[dict[str, str]],
    out=None,
    err=None,
    *,
    audit_path: Path | None = None,
    sleep=time.sleep,
    plan_sync=None,
) -> list[dict]:
    """Restore the prior live text from a reversal CSV. It deliberately runs
    NO place/evidence/grammar gates: ``restore_*`` is the exact text that was
    live before the apply, taken from the snapshot. A row whose live value
    equals ``current_*`` is pushed back (snapshot + read-back as in a forward
    apply); one already at ``restore_*`` is ALREADY_APPLIED; anything else was
    edited since and is SKIPPED_STALE. Stops on the first error."""
    out, err = out or sys.stdout, err or sys.stderr
    records: list[dict] = []
    audit = _recorder(records, audit_path)
    for row in rows:
        vid = row["video_id"]
        fields = ["title", *(["description"] if row["current_description"].strip() else [])]
        snippet = _live_snippet(service, vid)
        if snippet is None:
            print(f"{vid}: not found on YouTube -- skipping", file=err)
            continue
        states = []
        for f in fields:
            live = (snippet.get(f) or "").strip()
            cur, rest = row[f"current_{f}"].strip(), row[f"restore_{f}"].strip()
            states.append("restored" if live == rest else "current" if live == cur else "stale")
        descs = {"old_description": row["current_description"]} if len(fields) > 1 else {}
        cur_t, rest_t = row["current_title"], row["restore_title"]
        if "stale" in states:
            print(f"{vid}: STALE ROW -- live text differs -- skipping", file=err)
            audit(_audit_row(vid, cur_t, rest_t, SKIPPED_STALE, **descs))
            continue
        if all(s == "restored" for s in states):
            print(f"{vid}: already restored -- skipping", file=out)
            audit(_audit_row(vid, cur_t, rest_t, ALREADY_APPLIED, **descs))
            continue
        changes = {f: row[f"restore_{f}"] for f in fields}
        status, extra = _push_verified(service, vid, snippet, changes, audit_path, sleep)
        audit(_audit_row(vid, cur_t, rest_t, status, **descs, **extra))
        if status != APPLIED:
            print(f"{vid}: {status} -- stopping", file=err)
            break
        print(f"{vid}: restored {row['current_title']!r} -> {row['restore_title']!r}", file=out)
        _sync_plan(plan_sync, vid, row["restore_title"], out, err)
    return records


def write_audit_log(records: list[dict], path: Path) -> None:
    """Append ``records`` to ``path`` and fsync, so a crash keeps them."""
    with Path(path).open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
        f.flush()
        os.fsync(f.fileno())


def main(
    argv: list[str] | None = None, service: Any | None = None, out=None, err=None, sleep=time.sleep
) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path)
    parser.add_argument("--revert", type=Path, help="reversal CSV to restore from")
    parser.add_argument("--revert-out", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes-live", action="store_true")
    parser.add_argument("--credentials-dir", type=Path, default=None)
    parser.add_argument("--audit-log", type=Path, default=None)
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help="Directory holding social_queue.sqlite3 (default: config seo.storage_path). Each "
        "applied title is written into the short's publish-plan sidecar in the same step.",
    )
    args = parser.parse_args(argv)

    if args.apply != args.yes_live:
        print("Error: --apply and --yes-live must be passed together.", file=err)
        return 2

    if bool(args.worklist) == bool(args.revert):
        print("Error: pass exactly one of --worklist or --revert.", file=err)
        return 2

    revert_rows, results = None, None
    if args.revert:
        revert_rows = read_revert_file(args.revert)
        if revert_rows is None:
            print(f"Error: {args.revert} is not a reversal worklist.", file=err)
            return 2
        for row in revert_rows:
            print(
                f"{row['video_id']}: {row['current_title']!r} -> {row['restore_title']!r}", file=out
            )
    else:
        results = dry_run(read_worklist(args.worklist), load_place_corrections(), out=out)
    if not args.apply:
        print("DRY RUN -- no API calls made. Re-run with --apply --yes-live to push.", file=out)
        return 0

    queue_dir = args.queue_dir
    if queue_dir is None and service is None:
        try:
            from gomoveshift import config as gms_config

            queue_dir = Path(gms_config.get_config().seo.storage_path)
        except Exception as exc:
            print(
                f"Error: no delivery queue to sync publish plans against ({exc}); pass "
                "--queue-dir. Refusing before any live write.",
                file=err,
            )
            return 2
    conn = None
    plan_sync = None
    if queue_dir is not None:
        from gomoveshift.publishing.plan_title_sync import sync_plan_title
        from gomoveshift.publishing.queue import open_queue_db

        conn = open_queue_db(queue_dir)

        def plan_sync(vid, title):
            return sync_plan_title(conn, vid, title, apply=True)

    try:
        return _run_live(args, service, results, revert_rows, out, err, sleep, plan_sync)
    finally:
        if conn is not None:
            conn.close()


def _run_live(args, service, results, revert_rows, out, err, sleep, plan_sync) -> int:
    if service is None:
        try:
            from googleapiclient.discovery import build

            from gomoveshift.pipeline._yt_creds import load_youtube_credentials
            from gomoveshift.pipeline.drive_phases.youtube_uploader import _build_credentials_obj

            creds = _build_credentials_obj(load_youtube_credentials(args.credentials_dir))
            service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        except Exception as exc:
            print(f"Error: could not build the YouTube client ({exc}).", file=err)
            return 2

    source = args.revert or args.worklist
    audit_path = args.audit_log or source.with_suffix(".audit.jsonl")
    if revert_rows is not None:
        revert_updates(
            service, revert_rows, out, err, audit_path=audit_path, sleep=sleep, plan_sync=plan_sync
        )
        return 0
    revert_path = args.revert_out or args.worklist.with_suffix(".revert.csv")
    apply_updates(
        service,
        results,
        out,
        err,
        audit_path=audit_path,
        revert_path=revert_path,
        sleep=sleep,
        plan_sync=plan_sync,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
