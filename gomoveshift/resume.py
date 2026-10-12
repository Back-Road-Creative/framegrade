"""Resume capability for video stop detection.

Uses stops_report.json as an implicit checkpoint to resume
processing from where it left off after a crash or interruption.
"""

import json
from pathlib import Path
from typing import Dict, List


def get_checkpoint_state(input_dir: Path) -> Dict:
    """Load checkpoint from stops_report.json.

    Args:
        input_dir: Directory containing videos and potential checkpoint file.

    Returns:
        Dict with:
            - has_checkpoint: bool
            - completed_videos: List of video filenames that completed successfully
            - checkpoint_data: Full checkpoint data if exists
    """
    checkpoint_path = input_dir / "stops_report.json"

    if not checkpoint_path.exists():
        return {"has_checkpoint": False, "completed_videos": [], "checkpoint_data": None}

    try:
        content = checkpoint_path.read_text().strip()
        if not content:
            return {"has_checkpoint": False, "completed_videos": [], "checkpoint_data": None}

        data = json.loads(content)

        # Extract completed videos (those with 'stops' key and no 'error')
        completed = []
        videos = data.get("videos", {})

        for name, info in videos.items():
            if isinstance(info, dict) and "stops" in info and "error" not in info:
                completed.append(name)

        return {
            "has_checkpoint": len(completed) > 0,
            "completed_videos": completed,
            "checkpoint_data": data,
        }

    except (json.JSONDecodeError, ValueError):
        # Corrupted JSON - treat as fresh start
        return {"has_checkpoint": False, "completed_videos": [], "checkpoint_data": None}


def get_videos_to_process(input_dir: Path, force_fresh: bool = False) -> List[Path]:
    """Get list of videos that need processing.

    Args:
        input_dir: Directory containing video files.
        force_fresh: If True, ignore checkpoint and return all videos.

    Returns:
        List of Path objects for videos to process, sorted alphabetically.
    """
    # Find all video files
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}
    all_videos = []

    for ext in video_extensions:
        all_videos.extend(input_dir.glob(f"*{ext}"))
        all_videos.extend(input_dir.glob(f"*{ext.upper()}"))

    # Sort alphabetically
    all_videos = sorted(set(all_videos), key=lambda p: p.name.lower())

    if force_fresh:
        return all_videos

    # Get checkpoint state
    state = get_checkpoint_state(input_dir)

    if not state["has_checkpoint"]:
        return all_videos

    # Filter out completed videos
    completed_set = set(state["completed_videos"])
    remaining = [v for v in all_videos if v.name not in completed_set]

    return remaining


def merge_results(input_dir: Path, new_results: Dict) -> Dict:
    """Merge new results with existing checkpoint data.

    Args:
        input_dir: Directory containing checkpoint file.
        new_results: Dict mapping video names to their results.

    Returns:
        Merged results dict with all videos, sorted alphabetically.
    """
    state = get_checkpoint_state(input_dir)

    if state["checkpoint_data"]:
        merged_videos = dict(state["checkpoint_data"].get("videos", {}))
    else:
        merged_videos = {}

    # Add/overwrite with new results
    merged_videos.update(new_results)

    # Sort by video name
    sorted_videos = dict(sorted(merged_videos.items(), key=lambda x: x[0].lower()))

    return {"videos": sorted_videos}


def print_resume_status(input_dir: Path, total_videos: int) -> None:
    """Print resume status message if resuming from checkpoint.

    Args:
        input_dir: Directory containing checkpoint file.
        total_videos: Total number of videos in directory.
    """
    state = get_checkpoint_state(input_dir)

    if state["has_checkpoint"] and state["completed_videos"]:
        completed = len(state["completed_videos"])
        remaining = total_videos - completed
        print(f"Resuming: {completed}/{total_videos} already complete, {remaining} remaining")


def get_resume_start_index(input_dir: Path) -> int:
    """Get the starting index for display purposes.

    When resuming, we want [6/41] not [1/35] for the display.

    Args:
        input_dir: Directory containing checkpoint file.

    Returns:
        1-based index to start from (number of completed + 1).
    """
    state = get_checkpoint_state(input_dir)

    if not state["has_checkpoint"]:
        return 1

    return len(state["completed_videos"]) + 1
