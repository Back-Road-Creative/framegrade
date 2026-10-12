"""Run-level preflight accounts for planned ``--superlong`` output
space, not just the normal source x1.5 working set.

Before this, the ONLY superlong space check was ``check_superlong_output_space``
— called PER-DAY, MID-RUN from ``day_pipeline`` once a day master already
exists. A ``--superlong`` run could clear the run-level ``run_preflight``
staging gate and then fill the disk hours into a live encode. This proves
the run-level gate now refuses upfront when superlong is enabled and the
extra output-space margin won't fit, and proves a superlong-disabled run's
estimate is unaffected (the existing behaviour these tests mirror:
``tests/test_preflight.py::TestPreflightStagingSpace``).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

_GB = 1_073_741_824


def _make_sources(src: Path, total_gb: int, n: int = 3):
    src.mkdir(exist_ok=True)
    for i in range(n):
        f = src / f"GX{i:02d}0001.MP4"
        f.write_bytes(b"x")
    return sorted(src.glob("GX*.MP4"))


def _fake_stat(total_gb, n=3):
    def _stat(self_path, *, follow_symlinks=True):
        if str(self_path).endswith(".MP4"):
            return type("S", (), {"st_size": total_gb * _GB // n, "st_mode": 0o100644})()
        from os import stat as os_stat

        return os_stat(str(self_path))

    return _stat


def _fake_disk_usage(staging_free_gb, other_free_gb=500):
    def _usage(path):
        if "staging" in str(path):
            return type("U", (), {"free": staging_free_gb * _GB})()
        return type("U", (), {"free": other_free_gb * _GB})()

    return _usage


class TestRunLevelSuperlongSpace:
    def test_superlong_enabled_raises_when_output_margin_missing(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "src"
        _make_sources(src, total_gb=50)
        out = tmp_path / "out"
        staging = tmp_path / "staging"

        # 50 GB source -> normal peak ~75 GB (x1.5); 100 GB free comfortably
        # covers that alone but NOT the added superlong margin (raw concat +
        # final coexist x2, x1.2 safety -> +120 GB, total ~195 GB needed).
        with (
            patch("shutil.disk_usage", side_effect=_fake_disk_usage(staging_free_gb=100)),
            patch.object(Path, "stat", _fake_stat(50)),
        ):
            with pytest.raises(RuntimeError, match="(?i)peak|cache|intermediate"):
                run_preflight(
                    source_dir=src,
                    output_dir=out,
                    staging_dir=staging,
                    require_tools=None,
                    superlong_enabled=True,
                )

    def test_superlong_disabled_estimate_unchanged(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "src"
        _make_sources(src, total_gb=50)
        out = tmp_path / "out"
        staging = tmp_path / "staging"

        # Same source + same free space, superlong OFF (also the default):
        # the pre-existing x1.5 estimate alone fits in 100 GB, so the run
        # proceeds — proves the new term is additive, not a change to the
        # existing baseline.
        with (
            patch("shutil.disk_usage", side_effect=_fake_disk_usage(staging_free_gb=100)),
            patch.object(Path, "stat", _fake_stat(50)),
        ):
            run_preflight(
                source_dir=src,
                output_dir=out,
                staging_dir=staging,
                require_tools=None,
            )
