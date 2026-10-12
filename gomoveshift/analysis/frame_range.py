"""The probed source range, for the analysis-side extractions.

GoPro footage is full-range YUV and carries no range tag, and every automatic
YUV->RGB (or ->gray) conversion assumes LIMITED on an untagged stream. Whatever
then reads those pixels — a histogram, an absolute HSV threshold, an edge count
— is reading numbers from a domain the footage never used.

The grade chain states the range with ``setrange=<probed>`` ahead of its first
filter; this is that same decision for the readers feeding analysis
rather than pixels, in the two shapes they come in. :func:`setrange_filter`
prefixes an ffmpeg chain. :func:`align_cv2_frame_range` corrects an OpenCV frame
after the fact — ``cv2.VideoCapture`` exposes no range control and has already
decoded under the limited-range assumption by the time any code can intervene —
and cannot recover a value clipped at 0 or 255, which is why new code should
prefer the ffmpeg form. Both are no-ops on a limited-range source, and the tests
pin them to the same numbers on a full-range one: a half-tagged codebase is
worse than an untagged one, because then two readers disagree and nothing says
which is right.
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

FULL = "full"
LIMITED = "limited"

#: ffprobe's spellings for a limited-range stream. Anything else — including the
#: empty string and ``unknown`` — is the untagged GoPro case.
_LIMITED_TAGS = frozenset({"tv", "mpeg", "limited"})
_PROBE_TIMEOUT_S = 10

#: 16..235 over 0..255 — the exact gain a full-range tag applies at the
#: conversion, so applying it by hand reproduces what ffmpeg would have done.
_LIMITED_SCALE = 219.0 / 255.0
_LIMITED_OFFSET = 16.0


@lru_cache(maxsize=128)
def _probe(path: str) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_range", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
        )  # fmt: skip
        tag = (out.stdout or "").strip().lower()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("range probe failed for %s (%s) — assuming %s", path, exc, FULL)
        tag = ""
    return LIMITED if tag in _LIMITED_TAGS else FULL


def probe_color_range(video_path) -> str:
    """``"full"`` or ``"limited"``, cached per run. An unreadable or untagged
    stream answers ``"full"``: raw GoPro footage is full-range by construction.
    Same vocabulary and default as the encode side, so a reader and a writer
    never disagree about what an untagged file means."""
    return _probe(str(video_path))


# Expose the cache control the lru_cache wrapper would otherwise hide.
probe_color_range.cache_clear = _probe.cache_clear


def setrange_filter(video_path) -> str:
    """The ``setrange=`` token for the head of an ffmpeg filter chain."""
    return f"setrange={probe_color_range(video_path)}"


def align_cv2_frame_range(frame: np.ndarray, video_path) -> np.ndarray:
    """Move an OpenCV-decoded frame into the probed source range.

    OpenCV decodes an untagged stream as limited-range, so a full-range source
    comes back in the wrong numeric domain — the one the absolute HSV thresholds
    and brightness scores downstream then read. A limited-range source (or a
    non-array) comes back untouched, so no caller has to branch."""
    if probe_color_range(video_path) != FULL or not isinstance(frame, np.ndarray):
        return frame
    scaled = frame.astype(np.float32) * _LIMITED_SCALE + _LIMITED_OFFSET
    return np.clip(scaled, 0, 255).astype(frame.dtype)
