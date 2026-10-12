"""The uniform-lighting (global) grade must bake to a LUT, not an `eq` chain.

`eq` drags the stream to 8-bit yuv420p mid-chain while `lut3d` stays gbrp10le,
so the string chain graded every day master through an 8-bit bottleneck ahead
of its 10-bit encode. Rationale: docs/pipeline-internals.md.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.color import (
    CorrectionParams,
    build_filter_chain,
    params_to_lut3d_cube,
)

# A realistic uniform grade: every _build_single_filter branch fires but spatial.
# The process-wide temp dir, captured before any test redirects ``tempfile``.
SHARED_TMPDIR = Path(tempfile.gettempdir())

GRADED = CorrectionParams(
    shadow_lift=11.3,
    highlight_pull=9.9,
    saturation_mult=1.092,
    gamma=1.02,
    white_balance_r=0.05,
    white_balance_b=-0.05,
)


@pytest.fixture
def cube(tmp_path):
    return params_to_lut3d_cube(GRADED, output_path=tmp_path / "global.cube")


def test_single_span_with_a_lut_emits_lut3d_not_eq(cube):
    """The whole point: no 8-bit-collapsing filter in the global grade."""
    out = build_filter_chain([GRADED], lut_paths=[cube])

    assert f"lut3d={cube}" in out
    for collapsing in ("eq=", "curves=", "colorbalance="):
        assert collapsing not in out, f"{collapsing} survived in {out!r}"


def test_spatial_filters_ride_after_the_lut(cube):
    """Denoise/sharpen cannot live in a 3D LUT — they must still be emitted."""
    params = CorrectionParams(
        shadow_lift=11.3, saturation_mult=1.092, denoise_strength=4.0, sharpen_amount=0.6
    )
    out = build_filter_chain([params], lut_paths=[cube])

    assert out.index(f"lut3d={cube}") < out.index("hqdn3d=")
    assert "unsharp=" in out
    assert "eq=" not in out


def test_identity_params_stay_null_and_bake_nothing(tmp_path):
    """An identity grade must not gain a pointless LUT pass."""
    identity_cube = params_to_lut3d_cube(CorrectionParams(), output_path=tmp_path / "id.cube")
    assert build_filter_chain([CorrectionParams()], lut_paths=[identity_cube]) == "null"


def test_no_lut_path_keeps_the_legacy_string_chain():
    """Callers that have not opted in are unchanged (documented fallback)."""
    out = build_filter_chain([GRADED])
    assert "curves=" in out and "eq=" in out
    assert "lut3d=" not in out


class TestCorrectionsToFilter:
    """single_pass wires the bake for the global path and cleans the temp dir."""

    @pytest.fixture(autouse=True)
    def own_temp_root(self, tmp_path, monkeypatch):
        """Give every bake in this class its own temp root.

        ``_corrections_to_filter`` mkdtemps under ``tempfile.gettempdir()``,
        so on the shared root these tests both litter it (the caller owns the
        returned dir) and read it — the leak check below then fails on a
        concurrent xdist worker's directory instead of a real leak.
        """
        root = tmp_path / "tmproot"
        root.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(root))
        monkeypatch.setenv("TMPDIR", str(root))
        assert Path(tempfile.gettempdir()) == root, "temp-root redirect did not take"
        return root

    @staticmethod
    def _proc():
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        config = MagicMock()
        config.output.audio_bitrate = "192k"
        return SinglePassProcessor(config)

    def test_global_correction_bakes_one_lut(self):
        filter_str, method, lut_dir = self._proc()._corrections_to_filter([GRADED], None)

        assert method == "auto_global"
        assert "lut3d=" in filter_str
        assert "eq=" not in filter_str
        # Returned so ``process``'s finally-block can remove it.
        assert lut_dir is not None and lut_dir.is_dir()
        assert list(lut_dir.glob("*.cube"))

    def test_identity_correction_needs_no_lut_dir(self):
        filter_str, method, lut_dir = self._proc()._corrections_to_filter(
            [CorrectionParams()], None
        )

        assert filter_str == "null"
        assert lut_dir is None

    def test_bake_failure_degrades_to_the_string_chain_without_leaking(self):
        """A LUT that cannot be written must not fail the whole encode."""
        before = set(Path(tempfile.gettempdir()).glob("gms_span_luts_*"))
        with patch(
            "gomoveshift.pipeline.single_pass.params_to_lut3d_cube",
            side_effect=OSError("disk full"),
        ):
            filter_str, method, lut_dir = self._proc()._corrections_to_filter([GRADED], None)

        assert method == "auto_global"
        assert "curves=" in filter_str  # legacy chain, still a valid grade
        assert lut_dir is None
        assert set(Path(tempfile.gettempdir()).glob("gms_span_luts_*")) == before

    def test_a_sibling_workers_lut_dir_cannot_fail_the_leak_check(self):
        """The leak check must read only this test's own temp root.

        Reproduces a parallel-run flake: another worker creates a
        ``gms_span_luts_*`` dir in the shared temp dir inside the snapshot
        window, and the leak assertion above fails on someone else's
        directory. The intruder here is created mid-window on purpose.
        """
        intruders: list[Path] = []

        def _intrude_then_fail(*_args, **_kwargs):
            intruders.append(Path(tempfile.mkdtemp(prefix="gms_span_luts_", dir=SHARED_TMPDIR)))
            raise OSError("disk full")

        try:
            before = set(Path(tempfile.gettempdir()).glob("gms_span_luts_*"))
            with patch(
                "gomoveshift.pipeline.single_pass.params_to_lut3d_cube",
                side_effect=_intrude_then_fail,
            ):
                _filter_str, _method, lut_dir = self._proc()._corrections_to_filter([GRADED], None)

            assert intruders, "the interference never happened — test is not exercising the race"
            assert lut_dir is None
            assert set(Path(tempfile.gettempdir()).glob("gms_span_luts_*")) == before
        finally:
            for leftover in intruders:
                shutil.rmtree(leftover, ignore_errors=True)
