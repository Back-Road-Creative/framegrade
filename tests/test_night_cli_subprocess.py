"""The supported worker exports and resumes through its real entry point."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from gomoveshift.pipeline.photos.astro_linear import write_linear_artifact
from gomoveshift.pipeline.photos.astro_stack import StackResult
from gomoveshift.pipeline.photos.night_project import NightStore
from gomoveshift.pipeline.photos.night_render import validate_render


# A hang guard, not a speed assertion. Each run is a fresh ``python -m gomoveshift``
# whose ~8-10 s on this host is mostly import (the CLI loads every handler; the
# photos package pulls in scipy.optimize). Full xdist suites and CI share this
# host (a load average of 23 on 12 cores was seen), and one resume run hit 45 s in
# a suite whose other tests passed. The guard only
# has to catch a worker that never exits.
_HANG_GUARD_SECONDS = 300


def test_real_cli_exports_and_resumes_verified_project(tmp_path):
    memory = Path("/proc/meminfo")
    if not memory.exists():
        pytest.skip("night admission requires Linux memory availability")
    available = next(
        int(line.split()[1]) * 1024
        for line in memory.read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 2 * 1024**3 + 16 * 12 * 200:
        pytest.skip("host lacks the worker's 2 GiB memory reserve")
    if shutil.disk_usage(tmp_path).free < 1024**3 + 16 * 1024**2 + 16 * 12 * 12:
        pytest.skip("temporary output lacks the worker's free-space reserve")
    source = tmp_path / "capture.linear"
    rgb = np.full((12, 16, 3), 0.1, np.float32)
    write_linear_artifact(
        source,
        StackResult(
            rgb=rgb,
            n_frames=2,
            mean_rejected_fraction=0,
            sky_sum=rgb * 2,
            sky_count=np.full((12, 16), 2, np.uint16),
            ground=rgb,
        ),
        reference_raw="synthetic.NEF",
        capture_identity={},
        mask_provenance={"resolved_boundary_row": 10},
        source_frames=["synthetic.NEF"],
        registration_provenance=[{"frame": "synthetic.NEF"}],
        photometry_provenance=[{"frame": "synthetic.NEF"}],
        algorithm_version="test",
    )
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    workspace = tmp_path / "projects"
    service = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    # Only the service itself: Waypost (seo_manager) and gps-journey come from
    # the venv's pinned installs, and an inherited PYTHONPATH entry could shadow
    # a pin.
    environment["PYTHONPATH"] = str(service)
    environment["GMS_NIGHT_ADMISSION_LOCK"] = str(tmp_path / "admission.lock")
    command = [
        sys.executable,
        "-m",
        "gomoveshift",
        "photos",
        "night",
        "--workspace",
        str(workspace),
        "--brand",
        "test",
        "--wait-seconds",
        "0",
    ]

    def run(*arguments):
        result = subprocess.run(
            [*command, *arguments],
            cwd=service,
            env=environment,
            text=True,
            capture_output=True,
            timeout=_HANG_GUARD_SECONDS,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run("--source-root", str(tmp_path), "--sources", str(source))
    store = NightStore(workspace)
    projects = store.list("test")
    assert len(projects) == 1
    project = projects[0]
    assert project["status"] == "complete"
    assert project["quality_status"] == "review_required"
    item = project["items"][0]
    assert item["attempts"] == 1
    output = Path(item["output"])
    assert validate_render(source, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["state"] == "complete"
    assert receipt["quality_review_required"] is True
    with tifffile.TiffFile(output / "master.tif") as master:
        assert master.asarray().shape == (12, 16, 3)
        assert master.asarray().dtype == np.uint16
        assert master.pages[0].tags[34675].value
    with Image.open(output / "preview.jpg") as preview:
        preview.load()
        assert preview.size == (16, 12)
        assert preview.info["icc_profile"]

    def fingerprints():
        return {
            p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in output.iterdir()
        }

    preserved = fingerprints()
    attempts = set(output.parent.iterdir())
    run("--project", project["id"], "--resume")
    resumed = store.get("test", project["id"])
    assert resumed["status"] == "complete"
    assert resumed["items"][0]["attempts"] == 1
    assert resumed["items"][0]["output"] == str(output)
    assert set(output.parent.iterdir()) == attempts
    assert fingerprints() == preserved
    assert original == {p.name: p.read_bytes() for p in source.iterdir()}
