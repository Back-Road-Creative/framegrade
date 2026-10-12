"""Sub-pipeline groupings of the 19-stage long form pipeline.

Each sub-pipeline owns an ordered slice of ``STAGE_ORDER`` and iterates
with the same ``BaseStage`` / ``StageResult`` contract as the legacy run
loop. The top-level orchestrator invokes the four sub-pipelines in
sequence; state-file persistence (``pipeline_state.json``) stays at the
orchestrator level so the resume contract is unchanged.

The ``SubPipeline`` base plus the four groupings (ingest-and-prep,
light-metadata, render-and-derive, publish-and-archive) replace the
legacy ``STAGE_ORDER`` loop in the orchestrator.
``STAGE_ORDER`` itself remains the canonical flat list in
``pipeline.stages`` and is what ``test_stage_ordering`` validates.
"""

from .base import SubPipeline, SubPipelineResult
from .ingest_and_prep import IngestAndPrep
from .light_metadata import LightMetadata
from .publish_and_archive import PublishAndArchive
from .render_and_derive import RenderAndDerive

__all__ = [
    "IngestAndPrep",
    "LightMetadata",
    "PublishAndArchive",
    "RenderAndDerive",
    "SubPipeline",
    "SubPipelineResult",
]
