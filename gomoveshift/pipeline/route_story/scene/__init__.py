"""Offline MapLibre terrain scene page and headless capture harness.

This package renders Route Story's cinematic export frames using real 3D
terrain (MapLibre GL JS) inside a pinned headless Chromium, fully offline and
deterministic. ``vendor.ensure_maplibre`` fetches and verifies the vendored
MapLibre build; ``capture.SceneCapture`` drives the scene page and reads back
pixels frame by frame.
"""

from __future__ import annotations
