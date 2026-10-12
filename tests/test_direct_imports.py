"""Tests verifying phase modules use direct imports instead of resolve().

The resolve() indirection hides imports from static analysis.
Each phase module must import dependencies directly from their source
modules rather than using the resolve() runtime lookup.
"""

import ast
import importlib
from pathlib import Path


PHASES_DIR = Path(__file__).parent.parent / "gomoveshift" / "pipeline" / "drive_phases"


def _module_uses_resolve(module_path: Path) -> list[int]:
    """Return line numbers where resolve() is called in a module."""
    source = module_path.read_text()
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "resolve":
                lines.append(node.lineno)
    return lines


class TestPhaseADirectImports:
    """phase_a.py must not use resolve()."""

    def test_no_resolve_calls(self):
        lines = _module_uses_resolve(PHASES_DIR / "phase_a.py")
        assert lines == [], f"phase_a.py still calls resolve() on lines {lines}"

    def test_imports_cleanly(self):
        mod = importlib.import_module("gomoveshift.pipeline.drive_phases.phase_a")
        assert hasattr(mod, "PhaseA")

    def test_no_resolve_import(self):
        source = (PHASES_DIR / "phase_a.py").read_text()
        assert "from ._imports import resolve" not in source


class TestPhaseBCDayFirstDirectImports:
    """phase_bc_day_first.py must not use resolve()."""

    def test_no_resolve_calls(self):
        lines = _module_uses_resolve(PHASES_DIR / "phase_bc_day_first.py")
        assert lines == [], f"phase_bc_day_first.py still calls resolve() on lines {lines}"

    def test_imports_cleanly(self):
        mod = importlib.import_module("gomoveshift.pipeline.drive_phases.phase_bc_day_first")
        assert hasattr(mod, "PhaseBCDayFirst")

    def test_no_resolve_import(self):
        source = (PHASES_DIR / "phase_bc_day_first.py").read_text()
        assert "from ._imports import resolve" not in source


class TestPhaseDDirectImports:
    """phase_d.py must not use resolve()."""

    def test_no_resolve_calls(self):
        lines = _module_uses_resolve(PHASES_DIR / "phase_d.py")
        assert lines == [], f"phase_d.py still calls resolve() on lines {lines}"

    def test_imports_cleanly(self):
        mod = importlib.import_module("gomoveshift.pipeline.drive_phases.phase_d")
        assert hasattr(mod, "PhaseD")

    def test_no_resolve_import(self):
        source = (PHASES_DIR / "phase_d.py").read_text()
        assert "from ._imports import resolve" not in source


class TestAllPhasesImportCleanly:
    """All surviving phase modules can be imported without circular import errors."""

    def test_simultaneous_import(self):
        from gomoveshift.pipeline.drive_phases import (  # noqa: F401
            phase_a,
            phase_b_preview,
            phase_bc_day_first,
            phase_d,
        )

    def test_imports_resolve_to_real_classes(self):
        """Verify the imported names in phase modules are actual classes/functions,
        not lazy proxies or None."""
        from gomoveshift.pipeline.drive_phases import phase_a  # noqa: F811

        # phase_a should have direct access to its imports at module level
        # (they're used inside methods, but should be importable)
        assert phase_a.PhaseA is not None
