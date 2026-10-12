"""Handler for ``gomoveshift comments pull``; mirrors
``reconcile_playlists``'s injectable-``service`` idiom."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ...paths import gms_data_home
from ...pipeline.comments import (
    append_comments,
    build_triage_report,
    default_comments_path,
    fetch_youtube_comments,
    import_tiktok_comments,
    load_comments,
    load_video_metadata,
    parse_since,
)


def default_triage_path() -> Path:
    """Triage report location: ``comments-triage.md`` under the data home."""
    return gms_data_home() / "comments-triage.md"


def _optional_path_from_env(name: str) -> Path | None:
    """An optional metadata source: unset means "not configured" and is skipped."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def default_seo_ledger_path() -> Path | None:
    """SEO events ledger (``GMS_COMMENTS_SEO_LEDGER``); None when not configured."""
    return _optional_path_from_env("GMS_COMMENTS_SEO_LEDGER")


def default_videos_yaml_path() -> Path | None:
    """Site ``videos.yaml`` (``GMS_COMMENTS_VIDEOS_YAML``); None when not configured."""
    return _optional_path_from_env("GMS_COMMENTS_VIDEOS_YAML")


def run_comments_pull(
    *,
    channel_id: str | None,
    since: str = "90d",
    out_path: Path | None = None,
    credentials_dir: Path | None = None,
    service=None,
    out=None,
    err=None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    out_path = out_path or default_comments_path()

    if not channel_id:
        print("Error: --channel-id is required (no config default configured).", file=err)
        return 2
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        print(f"Error: {exc}", file=err)
        return 2

    if service is None:
        try:
            from googleapiclient.discovery import build

            from ...pipeline._yt_creds import load_youtube_credentials
            from ...pipeline.drive_phases.youtube_uploader import _build_credentials_obj

            creds = _build_credentials_obj(load_youtube_credentials(credentials_dir))
            service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Error: could not build the YouTube client ({exc}). "
                "Check SOPS credentials / google-api libs.",
                file=err,
            )
            return 2

    comments = fetch_youtube_comments(service, channel_id, since_dt)
    written = append_comments(out_path, comments)
    print(
        f"Pulled {len(comments)} comment(s); wrote {written} new line(s) to {out_path}",
        file=out,
    )
    return 0


def run_comments_triage(
    *,
    in_path: Path | None = None,
    out_path: Path | None = None,
    seo_ledger_path: Path | None = None,
    videos_yaml_path: Path | None = None,
    out=None,
    err=None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    in_path = in_path or default_comments_path()
    out_path = out_path or default_triage_path()
    seo_ledger_path = seo_ledger_path or default_seo_ledger_path()
    videos_yaml_path = videos_yaml_path or default_videos_yaml_path()

    if not in_path.exists():
        print(f"Error: no comments file at {in_path}. Run 'comments pull' first.", file=err)
        return 2

    comments = load_comments(in_path)
    if seo_ledger_path is None:
        print(
            "Note: GMS_COMMENTS_SEO_LEDGER is not set; skipping the SEO-ledger metadata lookup.",
            file=err,
        )
    if videos_yaml_path is None:
        print(
            "Note: GMS_COMMENTS_VIDEOS_YAML is not set; skipping the videos.yaml metadata lookup.",
            file=err,
        )
    metadata = load_video_metadata(seo_ledger_path, videos_yaml_path)
    report = build_triage_report(comments, metadata)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Triaged {len(comments)} comment(s); wrote report to {out_path}", file=out)
    return 0


def run_comments_import(
    *,
    platform: str,
    in_path: Path,
    video_id: str | None = None,
    out_path: Path | None = None,
    out=None,
    err=None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    out_path = out_path or default_comments_path()

    if platform != "tiktok":
        print(
            f"Error: unsupported --platform {platform!r} (only 'tiktok' is supported).",
            file=err,
        )
        return 2
    if not in_path.exists():
        print(f"Error: no file at {in_path}.", file=err)
        return 2

    try:
        comments = import_tiktok_comments(in_path, video_id)
    except ValueError as exc:
        print(f"Error: {exc}", file=err)
        return 2

    written = append_comments(out_path, comments)
    print(
        f"Imported {len(comments)} comment(s); wrote {written} new line(s) to {out_path}",
        file=out,
    )
    return 0


def cmd_comments(args, config):
    """Argparse entry point for ``gomoveshift comments``."""
    if args.comments_command == "pull":
        return run_comments_pull(
            channel_id=getattr(args, "channel_id", None),
            since=getattr(args, "since", "90d"),
            out_path=getattr(args, "out", None),
            credentials_dir=getattr(args, "credentials_dir", None),
        )
    if args.comments_command == "triage":
        return run_comments_triage(
            in_path=getattr(args, "in_path", None),
            out_path=getattr(args, "out_path", None),
        )
    if args.comments_command == "import":
        return run_comments_import(
            platform=getattr(args, "platform", None),
            in_path=getattr(args, "file", None),
            video_id=getattr(args, "video_id", None),
            out_path=getattr(args, "out", None),
        )
    print("Usage: gomoveshift comments {pull,triage,import}", file=sys.stderr)
    return 1
