"""Crash-safe JSON parsing for subprocess (ffprobe/ffmpeg) output.

An unguarded ``json.loads`` of ffprobe/ffmpeg stdout crashes
the pipeline mid-encode when the tool emits malformed bytes (truncated output on
a killed probe, a stray warning line, a binary blob on a corrupt source). A raw
``json.JSONDecodeError`` propagates as an opaque traceback that does not name
WHAT failed to parse.

``safe_json_loads`` wraps ``json.loads`` and re-raises a ``RuntimeError`` that
carries an operator-facing ``context`` string, so the failure points at the
producing command rather than the JSON decoder internals.

Scope rationale (verified per call site): this helper is applied
ONLY to the genuinely-unguarded subprocess-output parse sites
(``video_detect.detect_video_properties``, ``color_analysis.get_video_properties``).
The other sites considered were verified already-guarded and left unchanged:
``audio.py`` and all ``gopro_gps.py`` ffprobe parses already sit inside
``except json.JSONDecodeError`` / ``except Exception`` blocks that degrade
gracefully, and ``orchestrator_lock.py`` already catches the decode error and
treats the lock as stale. Routing those through a raising helper would not
change behaviour and would only inflate the diff. YAML config parse sites are
guarded separately in ``config.py`` (they raise ``ConfigError``, not
``RuntimeError``).
"""

import json
from typing import Any


def safe_json_loads(text: str, context: str) -> Any:
    """Parse ``text`` as JSON, raising a contextual ``RuntimeError`` on failure.

    Args:
        text: Raw JSON string (typically subprocess stdout).
        context: Operator-facing description of what produced ``text``,
            e.g. ``"ffprobe video properties"``.

    Returns:
        The decoded JSON value.

    Raises:
        RuntimeError: When ``text`` is not valid JSON. The message is
            ``"{context}: malformed JSON: {error}"``.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{context}: malformed JSON: {e}") from e
