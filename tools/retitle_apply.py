#!/usr/bin/env python3
"""Apply approved retitles from a ``retitle_worklist.py`` CSV to live YouTube
videos. Contract: docs/pipeline-internals.md."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

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


def dry_run(rows, corrections: PlaceCorrections | None = None, out=None) -> list[dict]:
    out = out if out is not None else sys.stdout
    corrections = corrections if corrections is not None else load_place_corrections()
    results = []
    for row in rows:
        issues = evaluate_row(row, corrections)
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


def apply_updates(service, results: list[dict], out=None, err=None) -> list[dict]:
    """Push each APPROVED row's title -- and its description, when the row
    carries one -- live in one update. One retry per call; stops on the first
    unrecovered API error."""
    out, err = out or sys.stdout, err or sys.stderr
    records: list[dict] = []
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
        items = service.videos().list(part="snippet", id=vid).execute().get("items") or []
        if not items:
            print(f"{vid}: not found on YouTube -- skipping", file=err)
            continue
        snippet = items[0]["snippet"]
        if (snippet.get("title") or "").strip() != old_title.strip():
            print(f"{vid}: STALE ROW -- live title differs -- skipping", file=err)
            records.append(_audit_row(vid, old_title, new_title, SKIPPED_STALE, **descs))
            continue
        live_desc = (snippet.get("description") or "").strip()
        if descs and live_desc != descs["old_description"].strip():
            print(f"{vid}: STALE ROW -- live description differs -- skipping", file=err)
            records.append(_audit_row(vid, old_title, new_title, SKIPPED_STALE, **descs))
            continue
        changes = {"title": new_title, **({"description": new_desc} if descs else {})}
        body = {"id": vid, "snippet": {**snippet, **changes}}
        request = service.videos().update(part="snippet", body=body)
        try:
            try:
                request.execute()
            except Exception:
                request.execute()  # one retry
        except Exception as exc:
            print(f"{vid}: API error ({exc}) -- stopping", file=err)
            records.append(_audit_row(vid, old_title, new_title, ERROR, **descs))
            break
        print(f"{vid}: applied {old_title!r} -> {new_title!r}", file=out)
        records.append(_audit_row(vid, old_title, new_title, APPLIED, **descs))
    return records


def write_audit_log(records: list[dict], path: Path) -> None:
    with Path(path).open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def main(argv: list[str] | None = None, service: Any | None = None, out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes-live", action="store_true")
    parser.add_argument("--credentials-dir", type=Path, default=None)
    parser.add_argument("--audit-log", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.apply != args.yes_live:
        print("Error: --apply and --yes-live must be passed together.", file=err)
        return 2

    results = dry_run(read_worklist(args.worklist), load_place_corrections(), out=out)
    if not args.apply:
        print("DRY RUN -- no API calls made. Re-run with --apply --yes-live to push.", file=out)
        return 0

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

    records = apply_updates(service, results, out=out, err=err)
    write_audit_log(records, args.audit_log or args.worklist.with_suffix(".audit.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
