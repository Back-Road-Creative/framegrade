"""Phase D consolidated Metricool CSV write failure must propagate.

Previously ``build_consolidated_shorts_csv`` failure was swallowed with
``logger.warning("Consolidated shorts CSV failed: %s", e)`` — the silent
failure made it possible for a Phase D to complete "successfully" with no
operator-visible CSV, which is the canonical Metricool deliverable.

The try/except has been removed; this regression test exercises the
removal directly to catch a re-introduction.
"""

from __future__ import annotations

import inspect

from gomoveshift.pipeline.drive_phases import phase_d


def test_phase_d_does_not_swallow_consolidated_csv_failure():
    """phase_d.py source must not log-and-continue around the CSV builder."""
    src = inspect.getsource(phase_d)
    assert "Consolidated shorts CSV failed" not in src, (
        "phase_d.py still swallows build_consolidated_shorts_csv failures; "
        "this silent-warn pattern shipped a 'green' Phase D with no CSV in "
        "a real run — propagate the RuntimeError."
    )
