"""The supported CLI renders and resumes real typed night products."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_linear import write_linear_artifact
from gomoveshift.pipeline.photos.astro_stack import StackResult
from gomoveshift.pipeline.photos.night_products import validate_product_receipt
from gomoveshift.pipeline.photos.night_project import NightStore


def _require_admission_capacity(root: Path) -> None:
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
    if shutil.disk_usage(root).free < 1024**3 + 128 * 1024**2:
        pytest.skip("temporary output lacks the worker's free-space reserve")


def _linear(path: Path, value: float) -> None:
    shape = (128, 160)
    rng = np.random.default_rng(20260916)
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    mono = np.full(shape, 0.02, np.float32)
    for x, y, flux in np.column_stack(
        (
            rng.uniform(8, shape[1] - 8, 60),
            rng.uniform(8, shape[0] - 8, 60),
            rng.uniform(0.2, 0.7, 60),
        )
    ):
        mono += value * flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4**2))
    rgb = np.repeat(mono[..., None], 3, axis=2)
    write_linear_artifact(
        path,
        StackResult(
            rgb=rgb,
            n_frames=2,
            mean_rejected_fraction=0,
            sky_sum=rgb * 2,
            sky_count=np.full(shape, 2, np.uint16),
            ground=rgb,
        ),
        reference_raw=f"{path.stem}.NEF",
        capture_identity={},
        mask_provenance={"mode": "automatic", "resolved_boundary_row": shape[0]},
        source_frames=[f"{path.stem}.NEF"],
        registration_provenance=[{"frame": f"{path.stem}.NEF"}],
        photometry_provenance=[{"frame": f"{path.stem}.NEF"}],
        algorithm_version="test",
    )


def _fingerprints(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_real_cli_renders_typed_stills_and_resumes_without_rewriting(tmp_path):
    _require_admission_capacity(tmp_path)
    collection = tmp_path / "card"
    collection.mkdir()
    _linear(collection / "stack-a.linear", 0.08)
    _linear(collection / "stack-b.linear", 0.12)
    source_before = _fingerprints(collection)
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

    def run(*arguments: str) -> None:
        result = subprocess.run(
            [*command, *arguments],
            cwd=service,
            env=environment,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run(
        "--source-root",
        str(tmp_path),
        "--collection",
        str(collection),
        "--products",
        "best_stills",
        "deep_still",
    )
    store = NightStore(workspace)
    projects = store.list("test")
    assert len(projects) == 1
    project = projects[0]
    assert project["schema"] == 2
    assert project["status"] == "complete"
    assert [task["product"] for task in project["tasks"]] == ["best_stills", "deep_still"]
    assert all(task["status"] == "complete" for task in project["tasks"])
    for task in project["tasks"]:
        assert validate_product_receipt(project, task, Path(task["output"])) is not None

    project_root = workspace / project["id"]
    output_before = _fingerprints(project_root)
    attempts_before = {path.name for path in project_root.iterdir()}
    run("--project", project["id"], "--resume")
    resumed = store.get("test", project["id"])

    assert resumed["status"] == "complete"
    assert all(task["attempts"] == 1 for task in resumed["tasks"])
    assert {path.name for path in project_root.iterdir()} == attempts_before
    assert _fingerprints(project_root) == output_before
    assert _fingerprints(collection) == source_before
    assert all(
        json.loads(Path(task["receipt"]).read_text())["state"] == "complete"
        for task in resumed["tasks"]
    )
