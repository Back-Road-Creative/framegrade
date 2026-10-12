"""The suite must never read the host's real free RAM.

``gomoveshift.pipeline.preflight.check_ram`` guards the real pipeline against
a starved box; in a test it guards nothing and turns a loaded CI runner into
red tests (ingest/orchestrator tests failing with ``Insufficient free RAM``).
``conftest.py`` pins the probe for every test; a test that wants a specific reading still patches it itself.
"""

from __future__ import annotations

from gomoveshift.pipeline import preflight

_GB = 1_073_741_824


def test_free_ram_probe_is_pinned_far_above_every_preflight_floor():
    assert preflight._virtual_memory().available >= 64 * _GB


def test_check_ram_cannot_fail_from_host_pressure():
    # The largest floor any caller passes is 4.0 GB; 16 GB proves the pin,
    # not the host, answers.
    preflight.check_ram(min_free_gb=16.0)
