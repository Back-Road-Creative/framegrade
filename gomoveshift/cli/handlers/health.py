"""``gomoveshift health check`` -- detect stuck or failed pipeline states (detection only)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from ...health import launch, run_check
from ...health.heal import default_drive_running, register_recipe, run_heal
from ...health.store import (
    code_sha,
    default_data_root,
    default_incidents_dir,
    load_all,
    utc_iso,
    write_row,
)


def cmd_health(args, config) -> int:
    root = Path(args.data_root) if args.data_root else default_data_root()
    incidents_dir = Path(args.incidents_dir) if args.incidents_dir else default_incidents_dir(root)
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else root / "logs" / "jobs"
    if args.health_command == "heal":
        return _cmd_heal(args, config, root, jobs_dir, incidents_dir)
    if args.health_command == "backlog":
        return _cmd_backlog(args, config, root, jobs_dir)
    result = run_check(
        config,
        incidents_dir=incidents_dir,
        jobs_dir=jobs_dir,
        code_root=Path(args.code_root) if args.code_root else root,
        staging_sha=args.staging_sha,
        ledger_path=getattr(args, "ledger_path", None),
    )
    if args.json:
        print(json.dumps(result["incidents"], indent=2, sort_keys=True))
    else:
        for row in result["incidents"]:
            print(f"[{row['kind']}] {row['key']}")
        print(
            f"{len(result['incidents'])} open incident(s); {len(result['new'])} new, "
            f"{len(result['pushed'])} pushed, {len(result['resolved'])} resolved, "
            f"{result['unreadable']} unreadable"
        )
        for note in result["notes"]:
            print(f"note: {note}")
    return 1 if result["incidents"] else 0


def _cmd_heal(args, config, root, jobs_dir, incidents_dir) -> int:
    if not getattr(args, "recipe", None):
        return _run_heal(args, config, root, jobs_dir, incidents_dir, None)
    if not args.dry_run:
        return _run_heal(
            args, config, root, jobs_dir, incidents_dir, _register(args, incidents_dir)
        )
    with tempfile.TemporaryDirectory() as scratch:  # a dry run registers nowhere real
        shadow = Path(scratch)
        for row in load_all(incidents_dir)[0]:
            write_row(shadow, row)
        return _run_heal(args, config, root, jobs_dir, shadow, _register(args, shadow))


def _register(args, incidents_dir):
    return register_recipe(incidents_dir, args.recipe, now_iso=utc_iso(time.time())) or ""


def _run_heal(args, config, root, jobs_dir, incidents_dir, only_key) -> int:
    if only_key == "":
        print(f"refused: no readable resume.json at {args.recipe}")
        return 2
    result = run_heal(
        only_key=only_key,
        incidents_dir=incidents_dir,
        jobs_dir=jobs_dir,
        code_root=Path(args.code_root) if args.code_root else root,
        config=config,
        mirror_sha=args.staging_sha,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key in result["would_launch"]:
            print(f"would launch: {key}")
        for att in result["attempts"]:
            print(f"{att['action']}: {att['key']} - {att['reason']}")
        print(
            f"{len(result['launched'])} launched, {len(result['would_launch'])} would launch, "
            f"{len(result['outcomes'])} outcome(s) pushed"
        )
        for note in result["notes"]:
            print(f"note: {note}")
    return 0


def _cmd_backlog(args, config, root, jobs_dir) -> int:
    from datetime import datetime, timezone

    code_root = Path(args.code_root) if args.code_root else root
    now = datetime.now(timezone.utc)
    region, rows, drives, slots, known, register = launch.load_inputs(
        config, args.region, args.ledger_path, now
    )
    acquire, release = launch.lease_fns(code_root)
    result = launch.run_backlog(
        region=region, rows=rows, drives=drives, free_slots=slots, known_projects=known,
        stages_root=args.stages_root, output_root=args.output_root,
        sources_root=args.sources_root or args.output_root / "_sources", code_root=code_root,
        jobs_dir=jobs_dir, mirror_sha=args.staging_sha or code_sha(code_root), now=now,
        dry_run=args.dry_run, lease_acquire=acquire, lease_release=release,
        drive_running=lambda: default_drive_running(jobs_dir),
        preflight=launch.default_preflight, launcher=launch.default_launcher, register=register,
    )  # fmt: skip
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for skip in result["skipped"]:
            print(f"skipped {skip['project']}: {skip['reason']}")
        for row in result.get("storage", []):
            state = "ok" if row["ok"] else "SHORT"
            print(f"storage dev {row['device']}: {row['free_gb']:.1f} GB free, need "
                  f"{row['need_gb']:.1f} GB [{state}] ({'; '.join(row['uses'])})")  # fmt: skip
        if result.get("storage_bound"):
            print(f"bound: {result['storage_bound']}")
        print(f"{result['action']}: {result['reason']}")
    return launch.exit_code(result)
