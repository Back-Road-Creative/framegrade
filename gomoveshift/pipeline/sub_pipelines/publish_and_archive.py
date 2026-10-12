"""publish-and-archive — upload, manifest, validate, archive, cleanup.

Stages (STAGE_ORDER indices 14-18): validation_gate, drive_upload,
csv_gen, d_drive_copy, cleanup.

Consumes from ctx: ``final_video`` and the metadata/social/thumbnail
artifacts produced by render-and-derive.

Invariant: ``validation_gate`` MUST run before ``drive_upload`` (and
``d_drive_copy``) so its audio/content hard-fail can block the
IRREVERSIBLE Drive/YouTube upload — not merely the local D: archive
copy. A validation failure is fatal, so ``SubPipeline.run`` aborts
before any upload stage. The slice ordering here is the same as
``STAGE_ORDER``; keeping the stages adjacent makes the invariant
visible at the call site, not buried in a flat list.

cleanup is intentionally last in this sub-pipeline (and last overall);
the orchestrator writes its run-summary JSON after this sub-pipeline
returns, not from inside the sub-pipeline.
"""

from __future__ import annotations

from ..stages import STAGE_ORDER
from .base import SubPipeline


class PublishAndArchive(SubPipeline):
    name = "publish_and_archive"
    stages = STAGE_ORDER[14:19]
