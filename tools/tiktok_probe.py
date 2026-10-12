#!/usr/bin/env python3
"""One-shot, non-pipeline probe: attempt a SINGLE real TikTok upload with the
credentials the user has configured, to find out which unaudited-client path
(if any) TikTok's live API accepts. A RUNNER, not a pipeline path: no retries,
no queue, no state persistence. Refuses to run without ``--yes-live``.

Usage: tools/tiktok_probe.py <video-file> --mode inbox|self-only --yes-live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from gomoveshift.publishing.tiktok_adapter import (
    TikTokAdapterError,
    init_direct_post,
    query_creator_info,
    upload_video_chunks,
)
from gomoveshift.publishing.tiktok_auth import get_valid_tiktok_credentials

PROBE_TITLE = "GMS TikTok unaudited-path probe"


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="path to a real video file to attempt to upload")
    parser.add_argument("--mode", choices=("inbox", "self-only"), required=True)
    parser.add_argument("--yes-live", action="store_true", help="required to make a live call")
    return parser.parse_args(argv)


def _chunk_reader(video_path: Path, chunk_plan):
    def read(index: int) -> bytes:
        start = index * chunk_plan.chunk_size
        is_last = index == chunk_plan.total_chunk_count - 1
        length = chunk_plan.video_size - start if is_last else chunk_plan.chunk_size
        with video_path.open("rb") as f:
            f.seek(start)
            return f.read(length)

    return read


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.yes_live:
        print("Refusing to run without --yes-live.", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"video file not found: {args.video}", file=sys.stderr)
        return 2

    video_size = args.video.stat().st_size
    access_token = get_valid_tiktok_credentials()["access_token"]

    try:
        if args.mode == "inbox":
            init = init_direct_post(access_token, video_size=video_size, mode="inbox")
        else:
            creator_info = query_creator_info(access_token)
            init = init_direct_post(
                access_token,
                title=PROBE_TITLE,
                privacy_level="SELF_ONLY",
                audit_approved=True,  # diagnostic override for this one-shot probe
                creator_info=creator_info,
                video_size=video_size,
                mode="direct",
            )
    except TikTokAdapterError as exc:
        print(f"REJECTED at init ({args.mode}): {exc}", file=sys.stderr)
        return 1

    print(f"ACCEPTED at init ({args.mode}): publish_id={init['publish_id']}")
    try:
        upload_video_chunks(
            init["upload_url"], init["chunk_plan"], _chunk_reader(args.video, init["chunk_plan"])
        )
    except TikTokAdapterError as exc:
        print(f"REJECTED at chunk upload ({args.mode}): {exc}", file=sys.stderr)
        return 1
    print(f"UPLOAD ACCEPTED ({args.mode}): publish_id={init['publish_id']} — check TikTok")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
