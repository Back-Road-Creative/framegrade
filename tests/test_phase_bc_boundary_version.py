"""A boundary-proof version change must re-run phase B+C on resume.

The route record phase D reads is built in phase B+C and only rebuilt when
``PHASE_BC_CODE_VERSION`` changes. A ``BOUNDARY_REVIEW_VERSION`` bump without a
phase B+C bump leaves every in-flight drive on its old proof (Galley Head
2025-09-20: a v2-era record with no Cork proof, refused as region_gaps).
"""

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    BOUNDARY_VERSION_BY_PHASE_BC,
    PHASE_BC_CODE_VERSION,
)
from gomoveshift.pipeline.orchestrator_fingerprint import (
    PHASE_CODE_VERSIONS,
    invalidate_phases_if_code_version_stale,
)
from gomoveshift.pipeline.route_claims import BOUNDARY_REVIEW_VERSION


def test_boundary_version_bump_needs_a_phase_bc_bump():
    assert BOUNDARY_VERSION_BY_PHASE_BC.get(PHASE_BC_CODE_VERSION) == BOUNDARY_REVIEW_VERSION, (
        "BOUNDARY_REVIEW_VERSION changed: bump PHASE_BC_CODE_VERSION and pair the new "
        "version in BOUNDARY_VERSION_BY_PHASE_BC, so resumed drives rebuild their route record"
    )


def test_a_checkpoint_from_before_the_coastal_proof_reruns_phase_bc_and_d():
    saved = {
        "phase_code_versions": {"phase_a": PHASE_CODE_VERSIONS["phase_a"], "phase_bc_day_first": 4}
    }
    resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
    assert invalidate_phases_if_code_version_stale(saved, resumed) == ["phase_a"]
