"""Pipeline state persistence for resume capability.

Handles saving, loading, and validating pipeline state so that interrupted
runs can resume from the last completed stage.
"""

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .ffmpeg_runner import validate_video
from .video_detect import VideoProperties

logger = logging.getLogger(__name__)


def save_state(
    state_file: Path,
    project_name: str,
    source_video: Path,
    source_hash: str,
    stages_completed: list[str],
    current_video: Path,
    video_properties: VideoProperties,
    timings: dict,
    params: dict,
    keep_segments: Optional[list[tuple[float, float]]] = None,
) -> None:
    """Save pipeline state for resume capability."""
    state = {
        "project_name": project_name,
        "source_video": str(source_video),
        "source_hash": source_hash,
        "stages_completed": stages_completed,
        "current_video": str(current_video),
        "video_properties": {
            "width": video_properties.width,
            "height": video_properties.height,
            "fps": video_properties.fps,
            "duration": video_properties.duration,
            "bitrate_kbps": video_properties.bitrate_kbps,
            "codec": video_properties.codec,
            "audio_codec": video_properties.audio_codec,
            "file_size_bytes": video_properties.file_size_bytes,
        },
        "timings": timings,
        "params": params,
    }

    if keep_segments is not None:
        state["keep_segments"] = [[s, e] for s, e in keep_segments]

    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=state_file.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, state_file)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    logger.debug(f"  Saved pipeline state: {len(stages_completed)} stages complete")


def load_state(state_file: Path) -> Optional[Dict[str, Any]]:
    """Load pipeline state if it exists.

    If the main state file is corrupt (e.g. crash during write), attempts
    recovery from a ``.tmp`` sibling left by the atomic write pattern.
    """
    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("State file corrupt (%s): %s — attempting .tmp recovery", state_file, e)
        return _recover_from_tmp(state_file)
    except Exception as e:
        logger.warning(f"  Failed to load state file: {e}")
        return None


def _recover_from_tmp(state_file: Path) -> Optional[Dict[str, Any]]:
    """Try to load state from a .tmp sibling file."""
    tmp_candidates = sorted(state_file.parent.glob("*.tmp"), reverse=True)
    for tmp in tmp_candidates:
        try:
            with open(tmp, "r") as f:
                data = json.load(f)
            logger.warning("Recovered state from %s", tmp.name)
            return data
        except (json.JSONDecodeError, ValueError, OSError):
            continue
    logger.warning("No valid .tmp sibling found for %s — returning None", state_file)
    return None


def append_stage_completed(state_file: Path, stage_name: str) -> bool:
    """Idempotently append ``stage_name`` to ``stages_completed`` in the state file.

    Preserves every other top-level key byte-for-byte via an atomic
    read-mutate-write (replacement with a temp-file sibling and ``os.replace``).

    Guarantees
    ----------
    - Unknown top-level keys survive the round-trip unchanged. This is the
      invariant that protects the upload CLI handler (and any future
      mutation caller) from silently dropping fields written by other
      producers.
    - Mutation is idempotent: repeated calls with the same ``stage_name``
      do not grow the list.
    - Missing state file, corrupt JSON, and I/O errors are reported via
      ``logger.warning`` and return ``False`` rather than raising.

    Returns
    -------
    bool
        ``True`` when the file was read, mutated (or already contained
        the stage), and written successfully. ``False`` when the file is
        absent, unreadable, or corrupt — the caller sees a logged
        warning in that case and can proceed with whatever downstream
        action it was going to take anyway.
    """
    if not state_file.exists():
        logger.debug("append_stage_completed: no state file at %s", state_file)
        return False

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "append_stage_completed: state file %s corrupt (%s) — leaving untouched",
            state_file,
            e,
        )
        return False
    except OSError as e:
        logger.warning("append_stage_completed: failed to read %s: %s", state_file, e)
        return False

    if not isinstance(state, dict):
        logger.warning(
            "append_stage_completed: state file %s has non-dict root (%s) — leaving untouched",
            state_file,
            type(state).__name__,
        )
        return False

    stages = state.get("stages_completed")
    if not isinstance(stages, list):
        stages = []
        state["stages_completed"] = stages

    if stage_name in stages:
        return True  # idempotent no-op; file unchanged

    stages.append(stage_name)

    # Atomic replacement keeps every other top-level key unchanged.
    tmp_fd, tmp_path = tempfile.mkstemp(dir=state_file.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, state_file)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        logger.warning("append_stage_completed: write failed for %s", state_file, exc_info=True)
        return False

    return True


def can_resume(
    state: Dict[str, Any],
    source_hash: str,
    params: dict,
) -> bool:
    """Check if we can resume from saved state.

    Args:
        state: Previously saved pipeline state dict.
        source_hash: Pre-computed hash of the current source video.
        params: Current run parameters to compare against saved params.
    """
    if not state:
        return False

    # Check source hash matches
    if state.get("source_hash") != source_hash:
        logger.warning("  Source video has changed (hash mismatch) - cannot resume")
        return False

    # Check params match (GPS params excluded — can be added/changed on resume)
    _gps_keys = {"gps_path", "gps_date", "gps_range"}
    state_params = state.get("params", {})
    for key, value in params.items():
        if key in _gps_keys:
            continue
        if state_params.get(key) != value:
            logger.warning(f"  Parameter '{key}' changed - cannot resume")
            return False

    # Check current_video still exists and is valid
    current_video = Path(state.get("current_video", ""))
    if not current_video.exists():
        logger.warning(f"  Intermediate video missing: {current_video} - cannot resume")
        return False

    # Validate intermediate video integrity (catches corrupt files from killed encodes)
    # Only validate intermediates that differ from the source — don't delete the source
    source_video = Path(state.get("source_video", ""))
    if current_video != source_video:
        validation = validate_video(current_video, require_audio=False)
        if not validation.valid:
            logger.warning(
                f"  Intermediate video corrupt: {validation.reason} — "
                f"deleting {current_video.name} and starting fresh"
            )
            try:
                current_video.unlink()
            except OSError:
                pass
            return False

    return True
