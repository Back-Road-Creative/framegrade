"""ingest-and-prep — validate the source, sync analytics, detect properties, stage to WSL.

Stages (STAGE_ORDER indices 0-3): validation, analytics_sync,
detect_properties, copy_to_wsl.

Produces on ctx: ``video_properties`` (PropertyDetectionStage), updated
``current_video`` (WSLCopyStage moves the working copy to fast WSL-native
storage so downstream encodes don't pay drvfs overhead).
"""

from __future__ import annotations

from ..stages import STAGE_ORDER
from .base import SubPipeline


class IngestAndPrep(SubPipeline):
    name = "ingest_and_prep"
    stages = STAGE_ORDER[0:4]
