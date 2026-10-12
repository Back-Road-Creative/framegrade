"""``PhotosConfig`` validators must not import the pipeline while validating.

Config validation runs at MODULE IMPORT time (``pipeline/longform_metadata.py``
calls ``get_config()`` at import), so a validator that imports
``gomoveshift.pipeline.*`` re-enters the half-built package and every command
dies with ``ImportError: cannot import name 'GOLDEN_COLUMNS' ... circular
import`` — an error naming CSV generation, nowhere near the operator's config
file. A lazy import inside the validator does not help: the validator is
already running inside an import.

The fix mirrors the ``border`` precedent — the valid set is a literal in
``config.py`` and a sync test here pins it to the live upstream. Test-time
imports sit outside the cycle, so they are safe.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from gomoveshift.config import PHOTO_LUT_OVER_STYLES, PHOTO_SOCIAL_ASPECTS, PhotosConfig
from gomoveshift.pipeline.photos.resizer import SOCIAL_ASPECTS, aspect_ratio
from gomoveshift.pipeline.photos.styles import STYLES

SERVICE_DIR = Path(__file__).resolve().parents[1]

# One key per hazard: each was set in config.yaml and took the whole CLI down.
HAZARD_KEYS = {"lut_over": "moody", "social_aspect": "square"}


# ── 1. the repro: a fresh interpreter, the key in config.yaml ─────────
# A same-process import will NOT reproduce this — pytest has already imported
# everything in a benign order, so the cycle never forms.


def _write_photo(path: Path) -> Path:
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run_cli_with_config(tmp_path: Path, yaml_text: str) -> subprocess.CompletedProcess:
    """Run ``photos develop --dry-run`` in a clean interpreter, cwd = tmp_path."""
    (tmp_path / "config.yaml").write_text(yaml_text)
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    env = dict(os.environ, PYTHONPATH=str(SERVICE_DIR), HOME=str(tmp_path))
    code = (
        "import sys; sys.argv = ['gomoveshift', 'photos', 'develop', "
        f"{str(photo)!r}, '--styles', 'natural', '--dry-run']\n"
        "from gomoveshift.cli import main\n"
        "sys.exit(main())\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("key,value", sorted(HAZARD_KEYS.items()))
def test_config_key_does_not_break_the_cli(tmp_path, key, value):
    result = _run_cli_with_config(tmp_path, f"photos:\n  {key}: {value}\n")
    assert "ImportError" not in result.stderr, result.stderr[-2000:]
    assert result.returncode == 0, result.stderr[-2000:]


def test_no_validator_imports_the_pipeline():
    """Static guard: no ``field_validator`` body may import the pipeline."""
    tree = ast.parse((SERVICE_DIR / "gomoveshift" / "config.py").read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = " ".join(ast.dump(d) for d in node.decorator_list)
        if "field_validator" not in decorators:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom) and "pipeline" in (child.module or ""):
                offenders.append(f"{node.name} -> {child.module}")
    assert offenders == [], f"validators import the pipeline at validation time: {offenders}"


# ── 2. the literals stay in sync with the live sources of truth ───────


def test_lut_over_literal_matches_styles():
    assert PHOTO_LUT_OVER_STYLES == tuple(STYLES)


def test_social_aspect_literal_matches_resizer():
    assert PHOTO_SOCIAL_ASPECTS == tuple(SOCIAL_ASPECTS)


# ── 3. validation still rejects bad values, naming key and options ────


@pytest.mark.parametrize("key,options", [("lut_over", STYLES), ("social_aspect", SOCIAL_ASPECTS)])
def test_bad_value_is_still_rejected(key, options):
    with pytest.raises(ValidationError) as exc:
        PhotosConfig(**{key: "rainbow"})
    message = str(exc.value)
    assert key in message
    for option in options:
        assert option in message


# ── 4. good values still accepted and still valid downstream ──────────


def test_every_style_is_accepted_and_known_to_the_engine():
    assert PhotosConfig().lut_over == ""
    for style in PHOTO_LUT_OVER_STYLES:
        assert PhotosConfig(lut_over=style).lut_over == style
        assert style in STYLES


def test_every_aspect_is_accepted_and_resolves_to_a_ratio():
    assert PhotosConfig().social_aspect == ""
    assert aspect_ratio("") is None
    for label in PHOTO_SOCIAL_ASPECTS:
        assert PhotosConfig(social_aspect=label).social_aspect == label
        assert aspect_ratio(label) > 0
