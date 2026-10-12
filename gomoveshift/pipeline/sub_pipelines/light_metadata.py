"""light-metadata — fast, reversible work before the irreversible encode.

Stages (STAGE_ORDER indices 4-9): stop_removal, blur_detection,
chapter_generation, metadata_generation, schedule_generation,
pre_render_review.

Consumes from ctx: ``video_properties``, ``current_video``.
Produces on ctx: ``keep_segments`` (StopRemovalStage; also persisted in
the state file), ``metadata`` and ``metadata_file`` (MetadataGenerationStage,
which reads the chapter file written by the prior stage from disk),
``schedule`` and ``num_shorts`` (ScheduleGenerationStage), and an updated
``current_video`` after stop removal.

Notes preserved across the split:
- ``ScheduleGenerationStage`` defaults to 15 shorts when shorts_analysis
  hasn't run yet — shorts_analysis lives in the next sub-pipeline
  (render-and-derive). This is existing behavior.
- ``PreRenderReviewStage`` is a human-review gate. It sits exactly on the
  boundary between this sub-pipeline and render-and-derive (the first
  irreversible encode) — the split intentionally aligns with that gate.
"""

from __future__ import annotations

from ..stages import STAGE_ORDER
from .base import SubPipeline


class LightMetadata(SubPipeline):
    name = "light_metadata"
    stages = STAGE_ORDER[4:10]
