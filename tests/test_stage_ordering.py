"""Tests for pipeline stage ordering invariants."""


def test_validation_gate_runs_before_d_drive_copy():
    """ValidationGateStage must run before DDriveCopyStage to prevent copying unvalidated files."""
    from gomoveshift.pipeline.stages import STAGE_ORDER, DDriveCopyStage, ValidationGateStage

    val_idx = next(i for i, s in enumerate(STAGE_ORDER) if isinstance(s, ValidationGateStage))
    copy_idx = next(i for i, s in enumerate(STAGE_ORDER) if isinstance(s, DDriveCopyStage))
    assert val_idx < copy_idx, "ValidationGateStage must run before DDriveCopyStage"


def test_sub_pipeline_stages_are_slices_of_stage_order():
    """The four sub-pipelines must share STAGE_ORDER's objects, not hand copies.

    Concatenating the sub-pipeline stage lists in order must reproduce
    STAGE_ORDER exactly — same objects (identity), same order, no gaps or
    dupes — so the stage graph has a single source of truth.
    """
    from gomoveshift.pipeline.stages import STAGE_ORDER
    from gomoveshift.pipeline.sub_pipelines import (
        IngestAndPrep,
        LightMetadata,
        PublishAndArchive,
        RenderAndDerive,
    )

    combined = (
        IngestAndPrep().stages
        + LightMetadata().stages
        + RenderAndDerive().stages
        + PublishAndArchive().stages
    )
    assert len(combined) == len(STAGE_ORDER)
    for got, want in zip(combined, STAGE_ORDER):
        assert got is want, f"{type(got).__name__} is a copy, not the STAGE_ORDER object"
