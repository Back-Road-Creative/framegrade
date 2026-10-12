"""Handler for the ``upload`` CLI command."""

import logging
import sys

from ..logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _assert_upload_dir_gated(shorts_dir, short_files):
    """Return an error string if the directory is ungated/stale, else ``None``.

    The standalone ``upload`` command must run the same pre-upload guards the
    pipeline stage path runs — never glob-ship whatever ``short_*.mp4`` sit on
    disk. Three checks, all upstream of the irreversible Drive upload:

    1. The run's own manifest (``social_posts.json``) must exist — its absence
       means the directory was not produced by a pipeline run.
    2. The short count on disk must match the planned count in that manifest,
       so a stale/pre-fix short left from an earlier run can't sneak through.
    3. The shorts quality gate (1080x1920 geometry / muted-by-design /
       integrity) must pass.
    """
    import json as _json

    from ...pipeline.pipeline_validator import PipelineValidator

    posts_file = shorts_dir / "social_posts.json"
    if not posts_file.exists():
        return (
            f"{posts_file} missing — this directory was not produced by a "
            "pipeline run; refusing to upload an ungated directory"
        )
    try:
        posts = _json.loads(posts_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"could not read {posts_file}: {exc}"
    planned = len(posts) if isinstance(posts, list) else None
    if planned != len(short_files):
        return (
            f"short count on disk ({len(short_files)}) does not match the "
            f"planned count ({planned}) in {posts_file.name} — refusing to "
            "ship a stale / mismatched directory"
        )
    result = PipelineValidator().validate_shorts(shorts_dir)
    if not result.passed:
        return f"shorts quality gate failed:\n{result.summary()}"
    return None


def cmd_upload(args, config):
    """Re-run Google Drive upload and optionally regenerate CSV for a completed project.

    State-file mutation note: ``_stages/pipeline_state.json``
    is produced by the main pipeline (``state_manager.save_state``) with a
    flat-dict shape unrelated to the drive-pipeline ``PipelineState`` dataclass.
    Mutation here goes through ``state_manager.append_stage_completed``, which
    preserves every unrelated top-level key byte-for-byte — so fields written
    by other producers (e.g. ``youtube_uploads_sha`` from the drive orchestrator,
    or anything a future refactor adds) do not get silently dropped on save.
    """
    import json

    from ...pipeline.state_manager import append_stage_completed
    from ...pipeline.uploader import (
        DEFAULT_CREDENTIALS_ENC,
        DEFAULT_TOKEN_ENC,
        PipelineUploader,
    )

    configure_logging()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    project_name = project_dir.name

    # Determine plaintext vs SOPS mode
    use_legacy = args.credentials is not None or args.token is not None

    if use_legacy:
        uploader = PipelineUploader(
            credentials_path=args.credentials,
            token_path=args.token,
        )
    else:
        uploader = PipelineUploader()

    # Handle --init-oauth: run OAuth browser flow, encrypt token, exit
    if args.init_oauth:
        if not DEFAULT_CREDENTIALS_ENC.exists():
            print("Google Drive credentials not configured.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Setup:", file=sys.stderr)
            print(
                "  1. Download OAuth client JSON from "
                "https://console.cloud.google.com/apis/credentials",
                file=sys.stderr,
            )
            print("  2. Encrypt with SOPS:", file=sys.stderr)
            print(
                f"     sops {DEFAULT_CREDENTIALS_ENC}  (or set GMS_DRIVE_CREDENTIALS_ENC)",
                file=sys.stderr,
            )
            print("     (paste JSON, save)", file=sys.stderr)
            print("  3. Re-run: gomoveshift upload --init-oauth <project_dir>", file=sys.stderr)
            return 1
        try:
            uploader.init_oauth()
            print("OAuth token obtained and encrypted successfully.")
            print(f"  Token: {DEFAULT_TOKEN_ENC}")
            return 0
        except Exception as e:
            print(f"OAuth flow failed: {e}", file=sys.stderr)
            return 1

    shorts_dir = project_dir / "shorts"
    short_files = sorted(shorts_dir.glob("short_*.mp4")) if shorts_dir.exists() else []
    if not short_files:
        print(f"Error: No shorts found in {shorts_dir}", file=sys.stderr)
        return 1

    # Refuse an ungated / stale directory BEFORE touching credentials, so a
    # pre-fix or stale short on disk can never reach Drive + the Metricool CSV.
    gate_err = _assert_upload_dir_gated(shorts_dir, short_files)
    if gate_err:
        print(f"Error: {gate_err}", file=sys.stderr)
        return 1

    # Credential existence check with appropriate messaging
    if use_legacy:
        if args.credentials and not args.credentials.exists():
            print(f"Error: Credentials file not found: {args.credentials}", file=sys.stderr)
            return 1
        if args.token and not args.token.exists():
            print(f"Error: Token file not found: {args.token}", file=sys.stderr)
            return 1
    elif not DEFAULT_CREDENTIALS_ENC.exists() or not DEFAULT_TOKEN_ENC.exists():
        print("Google Drive credentials not configured.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Setup:", file=sys.stderr)
        print(
            "  1. Download OAuth client JSON from "
            "https://console.cloud.google.com/apis/credentials",
            file=sys.stderr,
        )
        print("  2. Encrypt with SOPS:", file=sys.stderr)
        print(
            f"     sops {DEFAULT_CREDENTIALS_ENC}  (or set GMS_DRIVE_CREDENTIALS_ENC)",
            file=sys.stderr,
        )
        print("     (paste JSON, save)", file=sys.stderr)
        print("  3. Run initial OAuth flow:", file=sys.stderr)
        print("     gomoveshift upload --init-oauth <project_dir>", file=sys.stderr)
        print(
            f"  4. The token will be auto-encrypted to {DEFAULT_TOKEN_ENC}",
            file=sys.stderr,
        )
        return 1

    print(f"Uploading shorts for: {project_name}")
    print(f"  Shorts dir: {shorts_dir}")

    try:
        manifest = uploader.upload_shorts(
            shorts_dir=str(shorts_dir),
            project_name=project_name,
            drive_subfolder=project_name,
        )
        print(f"  Uploaded {len(manifest.files)} files to Google Drive")
    except Exception as e:
        print(f"Upload failed: {e}", file=sys.stderr)
        return 1

    # Update pipeline state — preserves every other top-level key.
    state_file = project_dir / "_stages" / "pipeline_state.json"
    if state_file.exists():
        append_stage_completed(state_file, "drive_upload")

    # Regenerate CSV with Drive links
    if not args.skip_csv:
        print("  Regenerating Metricool CSV with Drive links...")
        try:
            import yaml

            from ...pipeline.csv_generator import CSVGenerator
            from ...pipeline.scheduler import ReleaseSchedule, ScheduledSlot
            from ...pipeline.social_generator import SocialPost

            # Load social posts
            posts_file = shorts_dir / "social_posts.json"
            if not posts_file.exists():
                print("  Warning: social_posts.json not found \u2014 skipping CSV", file=sys.stderr)
            else:
                with open(posts_file) as f:
                    social_posts_data = json.load(f)
                post_objects = [
                    SocialPost(**p) if isinstance(p, dict) else p for p in social_posts_data
                ]

                # Load schedule
                schedule = None
                sched_file = project_dir / "release_schedule.yaml"
                if sched_file.exists():
                    with open(sched_file) as f:
                        sched_data = yaml.safe_load(f)
                    slots = [ScheduledSlot(**s) for s in sched_data.get("schedule", [])]
                    schedule = ReleaseSchedule(
                        project_name=sched_data.get("project_name", project_name),
                        location=sched_data.get("location", ""),
                        long_form_publish=sched_data.get("long_form_publish", ""),
                        window_start=sched_data.get("window_start", ""),
                        window_end=sched_data.get("window_end", ""),
                        num_shorts=sched_data.get("num_shorts", 0),
                        num_slots=sched_data.get("num_slots", 0),
                        slots=slots,
                    )

                csv_gen = CSVGenerator()
                csv_path = csv_gen.generate(
                    project_name=project_name,
                    social_posts=post_objects,
                    schedule=schedule,
                    manifest=manifest,
                    output_dir=str(project_dir),
                    validate=bool(manifest.files),
                )
                print(f"  CSV: {csv_path}")

                # Update pipeline state for csv_generation — same invariant.
                if state_file.exists():
                    append_stage_completed(state_file, "csv_generation")

        except Exception as e:
            print(f"  CSV regeneration failed: {e}", file=sys.stderr)

    print("Done!")
    return 0
