"""`deep_combine` refuses a stack whose noise does not average down.

An earlier run measured the scoped product passing to 30 chunks and failing at 34, and recommended against
shipping 30 as a constant.  The gate measures instead: if any sample box falls below 80%
efficiency, the combine refuses BEFORE anything irreversible is written, and names the box.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_noise_gate as gate


def test_sample_boxes_sit_inside_the_shipped_scoped_frame():
    """A box outside the crop would gate on pixels nobody receives."""
    from gomoveshift.pipeline.photos.astro_combine import scoped_frame_slice

    shape = (3728, 5600)
    rows, cols = scoped_frame_slice(shape)
    for name, (brow, bcol) in gate.noise_sample_boxes(shape).items():
        assert rows.start <= brow.start and brow.stop <= rows.stop, name
        assert cols.start <= bcol.start and bcol.stop <= cols.stop, name


def test_sample_boxes_are_the_audited_geometry():
    """512 px boxes at mid-height, centre and +/-25% of the width -- the audited geometry."""
    boxes = gate.noise_sample_boxes((3728, 5600))
    assert sorted(boxes) == ["centre", "left", "right"]
    for brow, bcol in boxes.values():
        assert brow.stop - brow.start == 512
        assert bcol.stop - bcol.start == 512
    centres = {n: (b[1].start + b[1].stop) // 2 for n, b in boxes.items()}
    assert centres["centre"] == 2800
    assert centres["right"] - centres["centre"] == 1400
    assert centres["centre"] - centres["left"] == 1400


def test_small_frames_get_no_boxes_rather_than_bogus_ones():
    assert gate.noise_sample_boxes((64, 64)) == {}


def test_report_is_advisory_when_there_are_no_boxes():
    report = gate.check_noise_floor({}, frames_per_chunk=10)
    assert report.passed is True
    assert report.failing_boxes == ()


def _small_boxes(shape):
    """Sample boxes sized for a test frame -- production boxes are 512 px and never fit here."""
    return {"centre": (slice(2, 14), slice(2, 14))}


def test_deep_combine_refuses_a_stack_whose_noise_does_not_average_down(tmp_path, monkeypatch):
    """The gate must actually fire end to end, not just pass because no box fits the frame."""
    from gomoveshift.pipeline.photos import astro_combine as combine
    from test_astro_combine import _mutate_manifest, _write_sky_artifact

    monkeypatch.setattr(combine, "noise_sample_boxes", _small_boxes)
    rng = np.random.default_rng(11)
    pattern = rng.normal(0.0, 1.0, size=(40, 20, 3)).astype(np.float32)
    paths = []
    for index in range(6):
        # Structure that ramps across the stack -- the failure mode the gate exists for.
        sky = (0.2 + 0.05 * pattern * index).astype(np.float32)
        paths.append(
            _write_sky_artifact(
                tmp_path, index, sky, ground=np.full(sky.shape, 0.1, dtype=np.float32)
            )
        )
    for path in paths:
        _mutate_manifest(
            path, lambda manifest: manifest["mask_provenance"].update(resolved_boundary_row=20)
        )

    with pytest.raises(combine.AstroCombineError) as excinfo:
        combine.deep_combine(
            paths, reference=paths[0], fit=lambda moving, reference: np.eye(2, 3, dtype=np.float64)
        )
    assert "does not average down" in str(excinfo.value)
    assert "centre" in str(excinfo.value)

    # The escape hatch exists and is explicit.
    result = combine.deep_combine(
        paths,
        reference=paths[0],
        fit=lambda moving, reference: np.eye(2, 3, dtype=np.float64),
        enforce_noise_floor=False,
    )
    assert result.noise_floor.passed is False
    assert result.noise_floor.failing_boxes == ("centre",)
