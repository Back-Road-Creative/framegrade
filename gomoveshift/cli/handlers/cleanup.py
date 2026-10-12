"""Handler for the ``cleanup`` CLI command."""

import sys

from ..logging_setup import configure_logging


def cmd_cleanup(args, config):
    """Clean up intermediate pipeline files."""
    from ...paths import gms_intermediate_base, gms_output_dir, gms_staging_dir
    from ...pipeline.cleanup import PipelineCleanup

    configure_logging()

    output_base = gms_output_dir()
    staging_dir = gms_staging_dir()
    # Drive runs keep _stages in the ext4 stage cache, not under output_base.
    stage_cache_base = gms_intermediate_base()

    cleanup = PipelineCleanup(
        output_base=output_base,
        staging_dir=staging_dir,
        d_drive_base=getattr(args, "archive_base", None),
        stage_cache_base=stage_cache_base,
    )

    if args.project:
        candidates = [output_base / args.project, stage_cache_base / args.project]
        projects = [p for p in candidates if p.exists()][:1]
        if not projects:
            print(f"Error: Project not found: {candidates[0]}", file=sys.stderr)
            return 1
    else:
        projects = cleanup.discover_projects()

    if not projects:
        print("No projects found.")
        return 0

    plan = cleanup.plan_cleanup(projects, min_age_days=args.min_age)

    if not plan.files:
        print("Nothing to clean up.")
        return 0

    # Display plan
    total_bytes = sum(f["size"] for f in plan.files)
    prefix = "DRY RUN \u2014 " if not args.confirm else ""
    print(f"{prefix}Cleanup plan: {len(plan.files)} files, {total_bytes / (1024**3):.2f} GB")
    print()

    for f in plan.files:
        size_mb = f["size"] / (1024 * 1024)
        print(
            f"  {'DELETE' if args.confirm else 'WOULD DELETE'}: {f['path']} "
            f"({size_mb:.1f} MB) \u2014 {f['reason']}"
        )

    for d in getattr(plan, "source_dirs", None) or []:
        print(
            f"  {'RECLAIM' if args.confirm else 'WOULD RECLAIM'} SOURCES DIR: {d['path']} "
            f"({d['size'] / (1024**3):.2f} GB)"
        )

    if plan.skipped:
        print()
        print(f"Skipped {len(plan.skipped)} item(s):")
        for s in plan.skipped:
            print(f"  SKIP: {s['path']} \u2014 {s['reason']}")

    if not args.confirm:
        print()
        print("This is a dry run. Add --confirm to actually delete.")
        return 0

    # Execute
    print()
    deleted = cleanup.execute_plan(plan)
    print(f"\nDeleted {deleted} file(s).")
    return 0
