"""Superlong sidecar integrity: missing keys are progress, not corruption.

The superlong sidecar (``superlong_uploads.json``) is an APPEND-AS-YOU-GO
progress log — a part record lands the moment its upload finishes (and
again on a quota abort), never as one complete manifest write. Resuming a
mid-upload part plan therefore ALWAYS has plan keys the sidecar hasn't
seen yet. ``assert_superlong_sidecar_integrity`` used to share the
long-form sidecar's "sidecar keys are a strict subset of the plan ==
corruption" contract (see the long-form contract's own tests, correct for
THAT sidecar — a complete manifest written once). Applied to a progress
log, a routine resume raised ``SidecarIntegrityError`` and pointed the
operator at ``--force-sidecar-reset``, which discards a good record of a
completed multi-gigabyte upload.

Fix: missing keys are acceptable when the sidecar's recorded plan
fingerprint agrees with the plan it is being checked against (or no
fingerprint was recorded). Phantom keys stay fatal unconditionally.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.exceptions import SidecarIntegrityError
from gomoveshift.pipeline.youtube_sidecar import (
    PLAN_FINGERPRINT_KEY,
    assert_superlong_sidecar_integrity,
    save_superlong_sidecar,
)


def _state(parts_by_date: dict) -> SimpleNamespace:
    """``{date: [part_index, ...]}`` → minimal ``state.superlong_parts``."""
    parts = {
        d: {"records": [{"path": "x", "part_index": i} for i in ix]}
        for d, ix in parts_by_date.items()
    }
    return SimpleNamespace(superlong_parts=parts)


class TestSuperlongSidecarProgress:
    def test_mid_upload_resume_with_matching_plan_fingerprint_is_progress(self, tmp_path):
        """2-part plan, 1 part uploaded, sidecar stamped for THIS plan —
        a normal resume must pass, not raise."""
        state = _state({"2026-03-01": [1, 2]})
        save_superlong_sidecar(tmp_path, {"2026-03-01:01": {"video_id": "a"}}, state=state)

        assert_superlong_sidecar_integrity(tmp_path, state)  # must not raise

    def test_mid_upload_resume_without_a_recorded_fingerprint_is_still_progress(self, tmp_path):
        """Production repro: a sidecar written by the pre-fix code path
        (no fingerprint stamp at all) with 1 of 2 parts uploaded still
        must not raise on resume."""
        state = _state({"2026-03-01": [1, 2]})
        save_superlong_sidecar(tmp_path, {"2026-03-01:01": {"video_id": "a"}})  # no state=

        assert_superlong_sidecar_integrity(tmp_path, state)  # must not raise

    def test_phantom_key_still_raises_even_with_matching_fingerprint(self, tmp_path):
        """A sidecar entry the current plan never produced is fatal
        regardless of progress/fingerprint — the phantom-ID guard
        stays unconditional."""
        state = _state({"2026-03-01": [1, 2]})
        uploads = {
            "2026-03-01:01": {"video_id": "a"},
            "2026-03-01:99": {"video_id": "ghost"},  # not in the plan
        }
        save_superlong_sidecar(tmp_path, uploads, state=state)

        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_superlong_sidecar_integrity(tmp_path, state)
        assert exc_info.value.phantom_indices == ["2026-03-01:99"]

    def test_missing_keys_raise_when_recorded_fingerprint_does_not_match(self, tmp_path):
        """Sidecar keys are a clean subset of the CURRENT plan (no
        phantoms) but were stamped against a DIFFERENT plan — the
        overlap is coincidental, not progress, and must still raise."""
        old_state = _state({"2026-03-01": [1, 2]})
        save_superlong_sidecar(tmp_path, {"2026-03-01:01": {"video_id": "a"}}, state=old_state)

        # Re-planned: the uploaded part's key survives, but the drive
        # was re-scoped to a different second part.
        new_state = _state({"2026-03-01": [1, 3]})

        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_superlong_sidecar_integrity(tmp_path, new_state)
        assert exc_info.value.phantom_indices == []
        assert "2026-03-01:03" in exc_info.value.missing_indices


class TestProductionPathStampsTheFingerprint:
    """A fingerprint nothing writes is dead scaffolding.

    Every assertion above calls ``save_superlong_sidecar(..., state=...)``
    directly. That proves the mechanism but NOT that production uses it —
    and if the coordinator omits ``state=``, real sidecars carry no
    fingerprint, the "same plan" check silently degrades to "any subset
    passes", and a re-planned split reusing a part index would resume
    against boundaries it never uploaded.
    """

    def test_the_coordinator_stamps_the_plan_fingerprint_on_a_real_upload(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from gomoveshift.config import Config
        from gomoveshift.pipeline import youtube_sidecar
        from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

        from _state_helpers import make_state

        part = tmp_path / "sl_01.mp4"
        part.write_bytes(b"x")
        state = make_state(
            region="usa",
            superlong_parts={
                "2026-03-01": {
                    "part_count": 1,
                    "records": [
                        {
                            "path": str(part),
                            "part_index": 1,
                            "title": "Superlong Part",
                            "description": "d",
                        }
                    ],
                }
            },
        )
        yt_mock = MagicMock()
        yt_mock.upload_if_needed.return_value = {"video_id": "PARTID"}
        config = Config()
        config.seo.storage_path = str(tmp_path)

        with (
            patch.object(
                esc, "_resolve_youtube_uploader_class", return_value=MagicMock(return_value=yt_mock)
            ),
            patch.object(esc, "_upload_lf_to_youtube", return_value={"video_id": "abc"}),
        ):
            esc._upload_lfs_to_youtube(
                drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
                regular_lf_paths=["/v/lf1.mp4"],
                metadata_per_lf={1: {"title": "T1", "description": "d", "tags": []}},
                drive_schedule=SimpleNamespace(
                    lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00")]
                ),
                youtube_uploads={},
                state=state,
                output_base=tmp_path,
                config=config,
                save_state=lambda s: None,
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )

        assert youtube_sidecar.load_superlong_plan_fingerprint(tmp_path) is not None, (
            "production upload wrote a sidecar with no plan fingerprint — "
            "the subset relaxation would then accept ANY subset on resume"
        )
        # ...and the marker must NOT appear as an upload entry, or the
        # integrity check would read it as a phantom key.
        assert PLAN_FINGERPRINT_KEY not in youtube_sidecar.load_superlong_sidecar(tmp_path)
