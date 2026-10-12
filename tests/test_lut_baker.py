"""
Tests for LUT baker — bake CorrectionParams into .cube 3D LUT files.

TDD: Tests written first, before implementation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.color import CorrectionParams


class TestIdentityLUT:
    """Identity (neutral) params should produce an identity LUT."""

    def test_identity_params_produce_identity_lut(self):
        """All-default CorrectionParams → diagonal LUT (input == output)."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams()  # all defaults = identity
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "identity.cube"
            result = params_to_lut3d_cube(params, cube_size=17, output_path=out)
            assert result.exists()

            # Parse and check diagonal points
            data = _parse_cube_file(result)
            size = data["size"]
            lut = data["lut"]

            # (0,0,0) → (0,0,0)
            r, g, b = lut[0]
            assert abs(r) < 0.005 and abs(g) < 0.005 and abs(b) < 0.005

            # (0.5,0.5,0.5) — midpoint on diagonal
            mid_idx = size // 2
            flat_idx = mid_idx * size * size + mid_idx * size + mid_idx
            r, g, b = lut[flat_idx]
            expected = mid_idx / (size - 1)
            assert abs(r - expected) < 0.005
            assert abs(g - expected) < 0.005
            assert abs(b - expected) < 0.005

            # (1,1,1) → (1,1,1)
            r, g, b = lut[-1]
            assert abs(r - 1.0) < 0.005 and abs(g - 1.0) < 0.005 and abs(b - 1.0) < 0.005


class TestNonIdentityLUT:
    """Known non-identity params should shift values predictably."""

    def test_shadow_lift_boosts_dark_values(self):
        """shadow_lift=10 should boost values near black above zero."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(shadow_lift=10.0)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "shadow.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)
            data = _parse_cube_file(out)
            lut = data["lut"]

            # Black point (0,0,0) should be lifted above zero
            r, g, b = lut[0]
            assert r > 0.01, f"Shadow lift should boost black: got r={r}"
            assert g > 0.01, f"Shadow lift should boost black: got g={g}"
            assert b > 0.01, f"Shadow lift should boost black: got b={b}"

    def test_highlight_pull_reduces_whites(self):
        """highlight_pull=10 should pull whites below 1.0."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(highlight_pull=10.0)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "highlight.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)
            data = _parse_cube_file(out)
            lut = data["lut"]

            # White point (1,1,1) should be pulled below 1.0
            r, g, b = lut[-1]
            assert r < 0.99, f"Highlight pull should reduce white: got r={r}"
            assert g < 0.99, f"Highlight pull should reduce white: got g={g}"
            assert b < 0.99, f"Highlight pull should reduce white: got b={b}"

    def test_saturation_boost_increases_chroma(self):
        """saturation_mult=1.2 should increase color difference from gray."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(saturation_mult=1.2)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sat.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)
            data = _parse_cube_file(out)
            size = data["size"]
            lut = data["lut"]

            # Test a saturated red point: (0.75, 0.25, 0.25)
            # Index: r=12, g=4, b=4 for size=17
            ri, gi, bi = 12, 4, 4
            flat_idx = bi * size * size + gi * size + ri
            r, g, b = lut[flat_idx]
            input_r = ri / (size - 1)
            input_g = gi / (size - 1)
            # Output red should be further from the mean than the input
            input_mean = (input_r + input_g + input_g) / 3.0
            output_mean = (r + g + b) / 3.0
            input_chroma = abs(input_r - input_mean)
            output_chroma = abs(r - output_mean)
            assert output_chroma > input_chroma * 1.05, (
                f"Saturation boost should increase chroma: "
                f"input={input_chroma:.4f}, output={output_chroma:.4f}"
            )

    def test_gamma_brightens_midtones(self):
        """gamma=1.1 should brighten midtones (raise values below 1.0)."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(gamma=1.1)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "gamma.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)
            data = _parse_cube_file(out)
            size = data["size"]
            lut = data["lut"]

            # Midpoint gray (0.5, 0.5, 0.5) should be brighter
            mid = size // 2
            flat_idx = mid * size * size + mid * size + mid
            r, g, b = lut[flat_idx]
            expected_identity = mid / (size - 1)
            # gamma > 1 in eq filter means brighter midtones
            assert r > expected_identity, (
                f"Gamma 1.1 should brighten mid: got {r:.4f}, expected > {expected_identity:.4f}"
            )


class TestCubeFileFormat:
    """Verify .cube file conforms to the standard format."""

    def test_header_has_lut_size(self):
        """Output .cube must have LUT_3D_SIZE header."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "format.cube"
            params_to_lut3d_cube(params, cube_size=64, output_path=out)

            text = out.read_text()
            assert "LUT_3D_SIZE 64" in text

    def test_correct_number_of_data_lines(self):
        """64^3 = 262144 data lines for a size-64 LUT."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "count.cube"
            params_to_lut3d_cube(params, cube_size=64, output_path=out)

            data = _parse_cube_file(out)
            assert len(data["lut"]) == 64**3, f"Expected {64**3} data lines, got {len(data['lut'])}"

    def test_domain_min_max_present(self):
        """Output .cube must have DOMAIN_MIN and DOMAIN_MAX headers."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "domain.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)

            text = out.read_text()
            assert "DOMAIN_MIN 0.0 0.0 0.0" in text
            assert "DOMAIN_MAX 1.0 1.0 1.0" in text

    def test_small_lut_size(self):
        """A size-4 LUT should have 4^3=64 data lines."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "small.cube"
            params_to_lut3d_cube(params, cube_size=4, output_path=out)

            data = _parse_cube_file(out)
            assert data["size"] == 4
            assert len(data["lut"]) == 64


class TestRoundTrip:
    """Write .cube, read it back, verify it parses."""

    def test_round_trip_parse(self):
        """Write a LUT, read it back, verify all values are valid floats in [0,1]."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(
            shadow_lift=5.0,
            highlight_pull=3.0,
            saturation_mult=1.1,
            contrast_curve=0.05,
            gamma=1.05,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "roundtrip.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)

            data = _parse_cube_file(out)
            assert data["size"] == 17
            assert len(data["lut"]) == 17**3

            # All values should be valid floats, clamped to [0, 1]
            for i, (r, g, b) in enumerate(data["lut"]):
                assert 0.0 <= r <= 1.0, f"R out of range at line {i}: {r}"
                assert 0.0 <= g <= 1.0, f"G out of range at line {i}: {g}"
                assert 0.0 <= b <= 1.0, f"B out of range at line {i}: {b}"


class TestDeterminism:
    """Same params must always produce identical .cube output."""

    def test_deterministic_output(self):
        """Two runs with same params produce byte-identical files."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(
            shadow_lift=8.0,
            highlight_pull=5.0,
            saturation_mult=1.15,
            contrast_curve=0.1,
            white_balance_r=-0.02,
            white_balance_b=0.03,
            gamma=1.05,
        )
        with tempfile.TemporaryDirectory() as td:
            out1 = Path(td) / "run1.cube"
            out2 = Path(td) / "run2.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out1)
            params_to_lut3d_cube(params, cube_size=17, output_path=out2)

            assert out1.read_bytes() == out2.read_bytes(), "LUT output must be deterministic"


class TestWhiteBalance:
    """White balance shifts should affect per-channel values."""

    def test_red_shift_affects_red_channel(self):
        """Positive white_balance_r should boost red relative to identity."""
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        # Note: white_balance_r is applied as colorbalance rs/rm, which shifts
        # red in shadows and midtones
        params = CorrectionParams(white_balance_r=0.05)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "wb.cube"
            params_to_lut3d_cube(params, cube_size=17, output_path=out)
            data = _parse_cube_file(out)
            size = data["size"]
            lut = data["lut"]

            # Check a mid-gray point
            mid = size // 2
            flat_idx = mid * size * size + mid * size + mid
            r, g, b = lut[flat_idx]
            # Red should be shifted relative to green
            assert r != g, "White balance R shift should differentiate R from G"


# ── Helpers ──────────────────────────────────────────────────────────────


def _parse_cube_file(path: Path) -> dict:
    """Parse a .cube LUT file into size and list of (R, G, B) tuples."""
    size = 0
    lut = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("TITLE"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if line.startswith("DOMAIN_MIN") or line.startswith("DOMAIN_MAX"):
                continue
            # Data line: three floats
            parts = line.split()
            if len(parts) == 3:
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                lut.append((r, g, b))
    return {"size": size, "lut": lut}
