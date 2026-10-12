"""Stage protocol and result types for the pipeline stage architecture.

Every pipeline stage implements StageProtocol: a should_run() guard and a
run() method that operates on PipelineContext. StageResult carries success/failure
with optional non-fatal semantics so the pipeline can continue past soft errors.
"""

from dataclasses import dataclass
from typing import Protocol

from .context import PipelineContext


class StageProtocol(Protocol):
    """Interface every pipeline stage implements."""

    name: str

    def should_run(self, ctx: PipelineContext) -> bool:
        """Return False to skip (already complete or user-skipped)."""
        ...

    def run(self, ctx: PipelineContext) -> "StageResult":
        """Execute the stage. Mutates ctx in place. Returns success/failure."""
        ...


@dataclass
class StageResult:
    """Outcome of a single stage execution.

    Attributes:
        success: Whether the stage completed without error.
        error: Human-readable error message (empty on success).
        fatal: If True and success=False, the pipeline aborts. If False,
               the pipeline logs the error and continues (soft failure).
    """

    success: bool
    error: str = ""
    fatal: bool = True


class BaseStage:
    """Convenience base class for pipeline stages.

    Provides default should_run() logic: skip if the stage name is in
    ctx.skip_stages or ctx.stages_completed. Subclasses override run().
    """

    name: str = ""

    def should_run(self, ctx: PipelineContext) -> bool:
        """Skip if stage is in the skip set or already completed."""
        if self.name in ctx.skip_stages:
            return False
        if self.name in ctx.stages_completed:
            return False
        return True

    def run(self, ctx: PipelineContext) -> StageResult:
        """Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")
