"""render-and-derive — irreversible encode + downstream derivatives.

Stages (STAGE_ORDER indices 10-13): single_pass_encode, thumbnails,
shorts_analysis, social.

Consumes from ctx: ``video_properties``, ``current_video``,
``keep_segments``, ``metadata``, ``schedule``.
Produces on ctx: ``final_video`` (SinglePassEncodeStage) and
thumbnail/shorts/social artifacts written under ``ctx.paths.project_dir``.

Notes preserved across the split:
- ``SinglePassEncodeStage`` is a META-stage: on success it appends three
  names to ``ctx.stages_completed`` (color_correction, ost_overlays,
  audio_normalization) in addition to its own ``single_pass_encode``
  marker. ``SubPipeline.run()`` is idempotent w.r.t. a stage re-adding
  its own name (covered by tests/test_sub_pipelines.py
  ``test_idempotent_when_stage_re_adds_itself``), so the meta-stage's
  ``should_run()`` override and the loop's append-if-missing guard
  cooperate without double-counting.
- single_pass_encode is the first irreversible stage in the pipeline;
  the human-review gate at the end of ``light-metadata`` exists exactly
  to bracket this boundary.
"""

from __future__ import annotations

from ..stages import STAGE_ORDER
from .base import SubPipeline


class RenderAndDerive(SubPipeline):
    name = "render_and_derive"
    stages = STAGE_ORDER[10:14]
