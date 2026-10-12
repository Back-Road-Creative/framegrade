"""
Golden reference regression test — CPU vs baked-LUT color parity.

Proves that the LUT baker (``params_to_lut3d_cube``, used by the CPU adaptive
per-span path) produces output within <2 LSB (2/255 ≈ 0.0078 in [0,1] space)
of the CPU filter chain math for all sample points.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.color import (
    CorrectionParams,
    SegmentStats,
    _apply_colorbalance,
    _apply_curves,
    _apply_gamma,
    _apply_saturation,
    build_filter_chain,
    compute_correction,
    params_to_lut3d_cube,
)


# ── Tolerance ──────────────────────────────────────────────────────────

MAX_DELTA = 2.0 / 255.0  # 2 LSB in [0,1] space ≈ 0.00784

# ── Helpers ────────────────────────────────────────────────────────────

GOLDEN_DIR = Path(__file__).parent / "golden"


def _cpu_pipeline(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Run the full CPU color math chain on an (N,3) float64 array.

    Mirrors color_filters._build_single_filter's order: curves (highlight knee
    folded into the top control point, fires at highlight_pull>=5) ->
    colorbalance -> eq(sat+gamma). The knee replaced the old whole-range
    colorlevels multiply that darkened midtones; both
    the CPU chain and the baked LUT fold it identically."""
    out = _apply_curves(rgb, params)
    out = _apply_colorbalance(out, params)
    out = _apply_saturation(out, params.saturation_mult)
    out = _apply_gamma(out, params.gamma)
    return np.clip(out, 0.0, 1.0)


def _lut_lookup(lut_path: Path, rgb: np.ndarray, cube_size: int) -> np.ndarray:
    """Read a .cube file and trilinear-approximate the output for each sample.

    For parity testing we use the nearest grid vertex rather than full
    trilinear interpolation — this is sufficient because we test at grid
    points (or near them) and the tolerance accounts for quantisation.
    """
    data_lines: list[str] = []
    with open(lut_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX")):
                continue
            data_lines.append(stripped)

    lut = np.array(
        [[float(v) for v in line.split()] for line in data_lines],
        dtype=np.float64,
    )
    assert lut.shape == (cube_size**3, 3), f"Expected {cube_size**3} entries, got {lut.shape[0]}"

    results = np.empty_like(rgb)
    for i in range(rgb.shape[0]):
        r, g, b = rgb[i]
        # Nearest grid index
        ri = int(round(r * (cube_size - 1)))
        gi = int(round(g * (cube_size - 1)))
        bi = int(round(b * (cube_size - 1)))
        ri = max(0, min(cube_size - 1, ri))
        gi = max(0, min(cube_size - 1, gi))
        bi = max(0, min(cube_size - 1, bi))
        flat_idx = bi * cube_size * cube_size + gi * cube_size + ri
        results[i] = lut[flat_idx]
    return results


# ── Sample Points ──────────────────────────────────────────────────────

# LUT grid size used for parity tests — all sample points must land exactly
# on grid vertices to avoid quantisation error in nearest-neighbour lookup.
PARITY_CUBE_SIZE = 65  # step = 1/64


def _grid(r: int, g: int, b: int) -> list[float]:
    """Convert grid indices (0..64) to [0,1] coordinates on a size-65 grid."""
    s = PARITY_CUBE_SIZE - 1  # 64
    return [r / s, g / s, b / s]


# 9 sample points — all land exactly on the 65-point grid
SAMPLE_POINTS = np.array(
    [
        _grid(0, 0, 0),  # black
        _grid(64, 64, 64),  # white
        _grid(32, 32, 32),  # mid-gray
        _grid(64, 0, 0),  # pure red
        _grid(0, 64, 0),  # pure green
        _grid(0, 0, 64),  # pure blue
        _grid(6, 6, 6),  # shadow  (~0.094)
        _grid(58, 58, 58),  # highlight (~0.906)
        _grid(51, 38, 32),  # skin tone (~0.797, 0.594, 0.500)
    ],
    dtype=np.float64,
)

# ── Realistic correction params ────────────────────────────────────────

REALISTIC_PARAMS = CorrectionParams(
    shadow_lift=8.0,
    highlight_pull=5.0,
    contrast_curve=0.10,
    white_balance_r=0.02,
    white_balance_b=-0.01,
    saturation_mult=1.15,
    gamma=1.05,
)


# ── Mathematical Parity Tests ──────────────────────────────────────────


class TestCpuGpuMathParity:
    """CPU filter chain math vs LUT baker output — must match within 2 LSB."""

    def test_realistic_params_parity(self, tmp_path: Path) -> None:
        """All 9 sample points should match within 2 LSB for realistic params."""
        lut_path = tmp_path / "parity.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(REALISTIC_PARAMS, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, REALISTIC_PARAMS)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        max_err = float(np.max(delta))
        assert max_err < MAX_DELTA, (
            f"Max per-channel delta {max_err:.6f} exceeds 2-LSB tolerance {MAX_DELTA:.6f}\n"
            f"Per-point max delta: {np.max(delta, axis=1)}"
        )

    def test_identity_params_parity(self, tmp_path: Path) -> None:
        """Default (identity) params: both CPU and LUT should be passthrough."""
        identity = CorrectionParams()
        lut_path = tmp_path / "identity.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(identity, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, identity)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        # CPU path with identity should be passthrough
        cpu_delta = np.abs(cpu_out - SAMPLE_POINTS)
        assert float(np.max(cpu_delta)) < MAX_DELTA, "CPU identity should be passthrough"

        # LUT path with identity should be passthrough
        lut_delta = np.abs(lut_out - SAMPLE_POINTS)
        assert float(np.max(lut_delta)) < MAX_DELTA, "LUT identity should be passthrough"

    def test_extreme_shadow_lift_parity(self, tmp_path: Path) -> None:
        """Extreme shadow_lift=15 — max allowed value."""
        params = CorrectionParams(shadow_lift=15.0)
        lut_path = tmp_path / "extreme_shadow.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"Extreme shadow lift: max delta {float(np.max(delta)):.6f}"
        )

    def test_extreme_saturation_parity(self, tmp_path: Path) -> None:
        """Max saturation_mult=1.25 — colours pushed outward."""
        params = CorrectionParams(saturation_mult=1.25)
        lut_path = tmp_path / "extreme_sat.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"Extreme saturation: max delta {float(np.max(delta)):.6f}"
        )

    def test_extreme_contrast_parity(self, tmp_path: Path) -> None:
        """Max contrast_curve=0.18 — strong S-curve."""
        params = CorrectionParams(
            shadow_lift=5.0,
            highlight_pull=5.0,
            contrast_curve=0.18,
        )
        lut_path = tmp_path / "extreme_contrast.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"Extreme contrast: max delta {float(np.max(delta)):.6f}"
        )

    def test_single_channel_r_correction_parity(self, tmp_path: Path) -> None:
        """Only white_balance_r active — single-channel correction."""
        params = CorrectionParams(white_balance_r=0.05)
        lut_path = tmp_path / "r_only.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"R-only correction: max delta {float(np.max(delta)):.6f}"
        )

    def test_single_channel_b_correction_parity(self, tmp_path: Path) -> None:
        """Only white_balance_b active."""
        params = CorrectionParams(white_balance_b=-0.05)
        lut_path = tmp_path / "b_only.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"B-only correction: max delta {float(np.max(delta)):.6f}"
        )

    def test_gamma_only_parity(self, tmp_path: Path) -> None:
        """Only gamma=1.15 active — max allowed value."""
        params = CorrectionParams(gamma=1.15)
        lut_path = tmp_path / "gamma_only.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"Gamma-only correction: max delta {float(np.max(delta)):.6f}"
        )

    def test_all_params_combined_parity(self, tmp_path: Path) -> None:
        """All corrections at strong (but valid) values simultaneously."""
        params = CorrectionParams(
            shadow_lift=12.0,
            highlight_pull=10.0,
            contrast_curve=0.15,
            white_balance_r=0.04,
            white_balance_b=-0.03,
            saturation_mult=1.20,
            gamma=1.10,
        )
        lut_path = tmp_path / "all_combined.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"All combined: max delta {float(np.max(delta)):.6f}"
        )


# ── LUT File Format Regression Test ───────────────────────────────────


class TestLutFormatRegression:
    """Golden reference: verify .cube file structure and sample values."""

    def test_header_fields(self, tmp_path: Path) -> None:
        """Header must contain TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX."""
        lut_path = tmp_path / "header.cube"
        params_to_lut3d_cube(REALISTIC_PARAMS, cube_size=17, output_path=lut_path)
        text = lut_path.read_text()
        assert 'TITLE "GMS Color Correction"' in text
        assert "LUT_3D_SIZE 17" in text
        assert "DOMAIN_MIN 0.0 0.0 0.0" in text
        assert "DOMAIN_MAX 1.0 1.0 1.0" in text

    def test_line_count(self, tmp_path: Path) -> None:
        """Total lines = header (5: TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX, blank)
        + LUT_3D_SIZE^3 data lines + trailing newline."""
        cube_size = 17
        lut_path = tmp_path / "linecount.cube"
        params_to_lut3d_cube(REALISTIC_PARAMS, cube_size=cube_size, output_path=lut_path)
        lines = lut_path.read_text().splitlines()
        # Filter out header and empty lines
        data_lines = [
            ln
            for ln in lines
            if ln.strip()
            and not ln.startswith(("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX"))
        ]
        assert len(data_lines) == cube_size**3, (
            f"Expected {cube_size**3} data lines, got {len(data_lines)}"
        )

    def test_golden_sample_values(self, tmp_path: Path) -> None:
        """Bake realistic params at size 17 and verify specific entries
        against golden reference values stored on disk."""
        golden_path = GOLDEN_DIR / "realistic_17_samples.json"
        cube_size = 17
        lut_path = tmp_path / "golden.cube"
        params_to_lut3d_cube(REALISTIC_PARAMS, cube_size=cube_size, output_path=lut_path)

        # Read back specific sample entries
        data_lines: list[str] = []
        with open(lut_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX")):
                    continue
                data_lines.append(s)

        # Sample indices: black(0), mid-gray, white, pure-R corner
        indices = {
            "black": 0,
            "mid_gray": (cube_size // 2) * cube_size**2
            + (cube_size // 2) * cube_size
            + cube_size // 2,
            "white": cube_size**3 - 1,
            "pure_red": cube_size - 1,  # R=max, G=0, B=0
        }

        actual: dict[str, list[float]] = {}
        for name, idx in indices.items():
            vals = [float(v) for v in data_lines[idx].split()]
            actual[name] = [round(v, 6) for v in vals]

        if golden_path.exists():
            expected = json.loads(golden_path.read_text())
            for name in indices:
                for ch in range(3):
                    assert abs(actual[name][ch] - expected[name][ch]) < 1e-5, (
                        f"Golden mismatch at {name}[{ch}]: "
                        f"got {actual[name][ch]}, expected {expected[name][ch]}"
                    )
        else:
            # First run: write golden reference
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(json.dumps(actual, indent=2) + "\n")
            pytest.skip("Golden reference written — re-run to validate")


# ── Filter String Regression Tests ─────────────────────────────────────


class TestFilterStringRegression:
    """Verify CPU filter chain strings match expected format."""

    def test_cpu_filter_chain_format(self) -> None:
        """CPU chain for realistic params should contain curves, colorbalance, eq."""
        chain = build_filter_chain([REALISTIC_PARAMS])
        assert "curves=m=" in chain, "CPU chain must include curves filter"
        assert "colorbalance=" in chain, "CPU chain must include colorbalance"
        assert "eq=" in chain, "CPU chain must include eq filter"
        assert "saturation=" in chain, "eq must set saturation"
        assert "gamma=" in chain, "eq must set gamma"

    def test_cpu_identity_returns_null(self) -> None:
        """CPU chain with identity params should return 'null' (passthrough)."""
        chain = build_filter_chain([CorrectionParams()])
        assert chain == "null"

    def test_cpu_filter_uses_space_separated_curves_points(self) -> None:
        """Curves points must be space-separated (not colon — known regression)."""
        chain = build_filter_chain([REALISTIC_PARAMS])
        # Extract the curves=m='...' portion
        for segment in chain.split(","):
            if segment.startswith("curves=m="):
                inner = segment.split("'")[1]
                # Points should be space-separated, each point is x/y
                points = inner.split(" ")
                assert len(points) == 4, f"Expected 4 control points, got {len(points)}"
                for pt in points:
                    parts = pt.split("/")
                    assert len(parts) == 2, f"Point {pt!r} should be x/y format"
                break
        else:
            pytest.fail("curves=m= not found in filter chain")

    def test_denoise_sharpen_excluded_from_lut(self, tmp_path: Path) -> None:
        """Denoise/sharpen are spatial — must NOT appear in LUT, only in filter string."""
        params = CorrectionParams(
            shadow_lift=5.0,
            denoise_strength=4.0,
            sharpen_amount=1.0,
        )
        lut_path = tmp_path / "spatial.cube"
        params_to_lut3d_cube(params, cube_size=17, output_path=lut_path)

        # LUT file should not contain any trace of denoise/sharpen
        lut_text = lut_path.read_text()
        assert "hqdn3d" not in lut_text
        assert "unsharp" not in lut_text

        # They ride the CPU filter chain instead (spatial, not LUT-bakeable).
        chain = build_filter_chain([params])
        assert "hqdn3d" in chain
        assert "unsharp" in chain


# ── Identity Passthrough Tests ─────────────────────────────────────────


class TestIdentityPassthrough:
    """Default/zero params should produce identity (passthrough) on both paths."""

    def test_cpu_identity_math(self) -> None:
        """CPU math with default params should return input unchanged."""
        identity = CorrectionParams()
        out = _cpu_pipeline(SAMPLE_POINTS, identity)
        delta = np.abs(out - SAMPLE_POINTS)
        assert float(np.max(delta)) < 1e-10, f"CPU identity max delta: {float(np.max(delta))}"

    def test_lut_identity_math(self, tmp_path: Path) -> None:
        """LUT with default params should return input unchanged at grid points."""
        identity = CorrectionParams()
        lut_path = tmp_path / "identity_check.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(identity, cube_size=cube_size, output_path=lut_path)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)
        delta = np.abs(lut_out - SAMPLE_POINTS)
        assert float(np.max(delta)) < MAX_DELTA, f"LUT identity max delta: {float(np.max(delta))}"

    def test_all_zero_corrections(self, tmp_path: Path) -> None:
        """Explicitly zeroed-out corrections (not defaults) should be identity."""
        zero = CorrectionParams(
            shadow_lift=0.0,
            highlight_pull=0.0,
            contrast_curve=0.0,
            white_balance_r=0.0,
            white_balance_b=0.0,
            saturation_mult=1.0,
            gamma=1.0,
        )
        lut_path = tmp_path / "zero.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(zero, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, zero)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        cpu_delta = np.abs(cpu_out - SAMPLE_POINTS)
        lut_delta = np.abs(lut_out - SAMPLE_POINTS)

        assert float(np.max(cpu_delta)) < 1e-10, "CPU zero should be identity"
        assert float(np.max(lut_delta)) < MAX_DELTA, "LUT zero should be identity"


# ── Edge Case Tests ────────────────────────────────────────────────────


class TestEdgeCases:
    """Extreme parameter combinations — ensure no NaN, no crash, parity holds."""

    def test_max_all_corrections(self, tmp_path: Path) -> None:
        """Every parameter at its documented maximum."""
        params = CorrectionParams(
            shadow_lift=15.0,
            highlight_pull=15.0,
            contrast_curve=0.18,
            white_balance_r=0.05,
            white_balance_b=0.05,
            saturation_mult=1.25,
            gamma=1.15,
        )
        lut_path = tmp_path / "max_all.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        # No NaN in either output
        assert not np.any(np.isnan(cpu_out)), "CPU output contains NaN"
        assert not np.any(np.isnan(lut_out)), "LUT output contains NaN"

        # Both in [0, 1]
        assert np.all(cpu_out >= 0.0) and np.all(cpu_out <= 1.0), "CPU out of range"
        assert np.all(lut_out >= 0.0) and np.all(lut_out <= 1.0), "LUT out of range"

        # Parity
        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA, (
            f"Max-all corrections: max delta {float(np.max(delta)):.6f}"
        )

    def test_min_gamma_parity(self, tmp_path: Path) -> None:
        """Gamma=0.9 — minimum documented value (darkens midtones)."""
        params = CorrectionParams(gamma=0.9)
        lut_path = tmp_path / "min_gamma.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA

    def test_negative_white_balance_both(self, tmp_path: Path) -> None:
        """Both R and B shifts at negative maximum."""
        params = CorrectionParams(white_balance_r=-0.05, white_balance_b=-0.05)
        lut_path = tmp_path / "neg_wb.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA

    def test_dehaze_produced_contrast_stays_in_parity(self, tmp_path: Path) -> None:
        """Dehaze rides the existing contrast_curve, so the elevated contrast
        it produces on hazy footage must bake identically in CPU and LUT paths."""
        hazy = SegmentStats(black_point=65.0, white_point=125.0, mid_tone=98.0, saturation=0.32)
        params = compute_correction(hazy)
        # The dehaze increment pushes contrast past the base S-curve ceiling.
        assert params.contrast_curve > 0.06

        lut_path = tmp_path / "dehaze.cube"
        cube_size = PARITY_CUBE_SIZE
        params_to_lut3d_cube(params, cube_size=cube_size, output_path=lut_path)

        cpu_out = _cpu_pipeline(SAMPLE_POINTS, params)
        lut_out = _lut_lookup(lut_path, SAMPLE_POINTS, cube_size)

        delta = np.abs(cpu_out - lut_out)
        assert float(np.max(delta)) < MAX_DELTA
