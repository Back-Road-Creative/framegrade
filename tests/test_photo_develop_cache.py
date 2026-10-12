"""Cache-integrity tests for the reserved develop-cache pair.

``DEVELOP_ALGO_VERSION`` / ``develop_cache_key`` have no consumer yet; these
tests pin the integrity contract a future cache store inherits: everything
that can change a developed frame's pixels — the decode path, the grade
appliers, spatial tunables of any type, decoder identity, the decoder
library itself, the operator's own knobs, the render depth, and the LUT cube
a look rendered through — must change the key.
"""

import inspect
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.color_analysis import analyze_frame
from gomoveshift.pipeline.color_correction import compute_correction
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos import (
    astro_calibrate,
    astro_develop,
    astro_meteors,
    astro_orchestrator,
    astro_register,
    astro_sequence,
    astro_stack,
    astro_triage,
    looks,
    masks,
)


def _photo(tmp_path):
    frame = np.full((16, 24, 3), (140, 120, 100), dtype=np.uint8)
    path = tmp_path / "a.jpg"
    Image.fromarray(frame, mode="RGB").save(path)
    return path, compute_correction(analyze_frame(frame))


def _key(path, params, **varied):
    """A key with every render input at its neutral default, so a test states
    only the ONE input it varies — the difference under test cannot be an
    accident of a hand-copied argument list."""
    return dev.develop_cache_key(
        path, params, **{"decoder": "pil", "options": dev.DevelopOptions(), "depth": 8, **varied}
    )


def _cube(bias: float, size: int = 3):
    """A tiny identity-plus-bias cube: two cubes differ in their TABLE only."""
    from pycube_lut import Cube

    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)
    table = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1) + bias
    return Cube(
        size=size,
        table=np.ascontiguousarray(table, dtype=np.float32),
        domain_min=np.zeros(3, dtype=np.float32),
        domain_max=np.ones(3, dtype=np.float32),
    )


# ── Decode + grade-applier coverage in the algo hash ─────────────────


def test_decode_sources_are_hashed():
    """Decode output IS the cached pixels' input — editing any decode fn must invalidate."""
    inputs = dev._develop_algo_inputs()
    for fn in (
        dev.decode,
        dev._decode_pil,
        dev._decode_raw,
        dev._decode_he_via_dng,
        dev._decode_raw_preview,
    ):
        assert inspect.getsource(fn) in inputs, fn.__name__


def test_apply_chain_sources_are_hashed():
    """GRADE_ALGO_VERSION covers only the correction producers; the appliers move pixels."""
    inputs = dev._develop_algo_inputs()
    for fn in (cc._apply_curves, cc._apply_colorbalance, cc._apply_saturation, cc._apply_gamma):
        assert inspect.getsource(fn) in inputs, fn.__name__


def test_raw_postprocess_constants_participate(monkeypatch):
    baseline = dev._develop_algo_version()
    monkeypatch.setattr(dev, "_RAW_POSTPROCESS", {**dev._RAW_POSTPROCESS, "output_bps": 8})
    assert dev._develop_algo_version() != baseline


# ── Non-float tunables ───────────────────────────────────────────────


def test_an_int_tunable_participates(monkeypatch):
    baseline = dev._develop_algo_version()
    monkeypatch.setattr(dev, "MEASURE_MAX_DIM", dev.MEASURE_MAX_DIM + 1)
    assert dev._develop_algo_version() != baseline


def test_the_noise_kernel_participates(monkeypatch):
    baseline = dev._develop_algo_version()
    monkeypatch.setattr(dev, "_NOISE_KERNEL", dev._NOISE_KERNEL * 2.0)
    assert dev._develop_algo_version() != baseline


def test_function_valued_globals_never_reach_the_hash():
    """repr() of a function embeds a memory address — process-unstable by nature."""
    for _name, serialized in dev._tunable_constants():
        assert " at 0x" not in serialized, _name


# ── Decoder identity + library version in the cache key ──────────────


def test_decoder_identity_distinguishes_preview_dng_and_raw(tmp_path):
    path, params = _photo(tmp_path)
    keys = {_key(path, params, decoder=d) for d in ("rawpy", "rawpy-dng", "rawpy-preview", "pil")}
    assert len(keys) == 4


def test_blank_decoder_hard_fails(tmp_path):
    path, params = _photo(tmp_path)
    options = dev.DevelopOptions()
    with pytest.raises(ValueError, match="decoder"):
        dev.develop_cache_key(path, params, options=options, depth=8)
    with pytest.raises(ValueError, match="decoder"):
        dev.develop_cache_key(path, params, decoder="   ", options=options, depth=8)


def test_decoder_library_version_is_in_the_key(tmp_path, monkeypatch):
    path, params = _photo(tmp_path)
    baseline = _key(path, params, decoder="rawpy")
    monkeypatch.setattr(dev, "_decoder_runtime", lambda: "rawpy=9.9.9,libraw=(9,9,9)")
    assert _key(path, params, decoder="rawpy") != baseline


# ── Render inputs the grade params cannot see ────────────────────────


def test_a_spatial_knob_changes_the_key(tmp_path):
    """``CorrectionParams`` is the COLOUR chain; the spatial stages ride on top
    of it and move pixels of their own. A vignetted render and a clean one of
    one file must never share an entry."""
    path, params = _photo(tmp_path)
    assert _key(path, params, options=dev.DevelopOptions(vignette=0.4)) != _key(path, params)


def test_masked_layers_change_the_key(tmp_path):
    """``layers`` is a colour move the shared ``params`` record never carries."""
    path, params = _photo(tmp_path)
    layered = dev.DevelopOptions(layers=({"mask": "sky", "vibrance": 0.3},))
    assert _key(path, params, options=layered) != _key(path, params)


def test_every_options_field_reaches_the_key(tmp_path):
    """Derived from the dataclass, not a hand-kept knob list: the next field
    added to ``DevelopOptions`` is keyed the moment it exists."""
    path, params = _photo(tmp_path)
    baseline = _key(path, params)
    moves = {float: 0.5, int: 1, str: "x", tuple: (0.5,), bool: True}
    for f in fields(dev.DevelopOptions):
        moved = moves[type(f.default)]
        assert moved != f.default, f"{f.name}: pick a move that is not the default"
        options = replace(dev.DevelopOptions(), **{f.name: moved})
        assert _key(path, params, options=options) != baseline, f.name


def test_render_depth_changes_the_key(tmp_path):
    """8 vs 16 quantizes the grade differently — same params, different pixels."""
    path, params = _photo(tmp_path)
    assert _key(path, params, depth=16) != _key(path, params, depth=8)


def test_two_cubes_under_one_name_do_not_share_a_key(tmp_path):
    """Every LUT variant is stamped with the SHARED under-grade params, so the
    cube's own table is the only thing that tells two film looks apart."""
    path, params = _photo(tmp_path)
    keys = {_key(path, params, lut=("portra", _cube(b))) for b in (0.0, 0.05)}
    assert len(keys) == 2


def test_a_lut_surface_is_not_the_styled_render_it_rides_on(tmp_path):
    path, params = _photo(tmp_path)
    veiled = _key(path, params, lut=("portra", _cube(0.0)))
    assert veiled != _key(path, params)
    assert veiled != _key(path, params, lut=("velvia", _cube(0.0))), "the cube NAME counts too"


def test_options_cannot_be_omitted(tmp_path):
    """Required, not defaulted: a caller cannot silently key a vignetted render
    as if it were the clean one."""
    path, params = _photo(tmp_path)
    with pytest.raises(TypeError, match="options"):
        dev.develop_cache_key(path, params, decoder="pil", depth=8)


def test_identical_inputs_repeat_the_same_key(tmp_path):
    """Still a deterministic function of its inputs — a cache that cannot hit
    is as useless as one that hits wrongly."""
    path, params = _photo(tmp_path)
    options = dev.DevelopOptions(vignette=0.4, grain=0.2, layers=({"mask": "sky"},))
    keys = {
        _key(path, params, options=options, depth=16, lut=("portra", _cube(0.05))) for _ in range(2)
    }
    assert len(keys) == 1


def test_decoder_runtime_names_the_library():
    assert "rawpy" in dev._decoder_runtime()


# ── Pure function of the sources ─────────────────────────────────────


def test_algo_version_is_stable_across_processes():
    """A fresh interpreter and a live recompute both agree with the import-time value."""
    assert dev._develop_algo_version() == dev.DEVELOP_ALGO_VERSION

    service_dir = Path(dev.__file__).resolve().parents[3]
    fresh = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gomoveshift.pipeline.photos.develop import DEVELOP_ALGO_VERSION; "
            "print(DEVELOP_ALGO_VERSION)",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(service_dir)},
        cwd=service_dir,
        check=True,
    )
    assert fresh.stdout.strip() == dev.DEVELOP_ALGO_VERSION


def test_every_spatial_stage_the_chain_calls_is_hashed():
    """``_HASHED_SOURCES`` must cover every stage ``_spatial_chain`` composes.

    Listing the stages by hand is how three of them (``_scanline``, ``_glow``,
    ``_light_leak``) were shipped OUTSIDE the hash: each one's PR added a stage
    to the chain and forgot the tuple, so retuning that stage's body would not
    change ``DEVELOP_ALGO_VERSION`` and a cached develop could replay with the
    old maths. Derive the list from the chain itself instead of restating it —
    the next stage is covered the moment it is wired, with nothing to remember.
    """
    called = [n for n in dev._spatial_chain.__code__.co_names if callable(getattr(dev, n, None))]
    assert called, "the chain calls nothing — introspection broke, not the contract"
    missing = sorted(n for n in called if getattr(dev, n) not in dev._HASHED_SOURCES)
    assert not missing, f"spatial stages missing from _HASHED_SOURCES: {missing}"


# ── Masked-look coverage: the fingerprint stamped on every sidecar ────


def test_every_callable_the_look_chain_reaches_is_hashed():
    """The same derivation, one layer out: the masked-look path composes helpers
    from develop AND masks, and the sidecar stamps this fingerprint as provenance
    on every developed photo. Derived from the four composing functions, so a call
    added inside one of them is covered the moment it is wired."""
    inputs = dev._develop_algo_inputs()
    owners = (dev._spatial_chain, dev._look_params, dev._layered, dev.look_variant_options)
    reached = []
    for owner in owners:
        assert inspect.getsource(owner) in inputs, owner.__name__
        for name in owner.__code__.co_names:
            for namespace in (dev, masks):
                fn = getattr(namespace, name, None)
                if inspect.isfunction(fn) and fn.__module__ in (dev.__name__, masks.__name__):
                    reached.append(fn)
    assert reached, "the look chain calls nothing — introspection broke, not the contract"
    for fn in reached:
        assert inspect.getsource(fn) in inputs, fn.__name__


def test_every_mask_the_registry_can_render_is_hashed():
    """A mask IS pixels: retuning ``_tone``'s window or a ``MASKS`` lambda moves
    a layer's boundary. Derived from the module namespace AND the registry — the
    luma windows are lambdas that live in no namespace to enumerate."""
    inputs = dev._develop_algo_inputs()
    owned = [
        value
        for value in vars(masks).values()
        if inspect.isfunction(value) and value.__module__ == masks.__name__
    ]
    assert len(owned) >= 5, "mask introspection broke, not the contract"
    for fn in owned + list(masks.MASKS.values()):
        assert inspect.getsource(fn) in inputs, getattr(fn, "__qualname__", "?")


def test_a_mask_tunable_participates(monkeypatch):
    """Read live, like every other tunable: source text alone would not notice a
    retune of a constant that decides where a masked layer's edge falls."""
    baseline = dev._develop_algo_inputs()
    monkeypatch.setattr(masks, "LUMA_EDGES", (0.2, 0.8))
    assert dev._develop_algo_inputs() != baseline


def test_a_builtin_look_recipe_participates(monkeypatch):
    """Retuning a shipped recipe moves that look's pixels, so the fingerprint must
    move with it — and, being sorted, dict ORDER alone must NOT move it."""
    original = looks.BUILTIN_LOOKS
    baseline = dev._develop_algo_inputs()
    name = sorted(original)[0]
    retuned = {**original, name: {**original[name], "vibrance": 0.123}}
    assert retuned != original, "pick a value the recipe does not already carry"
    monkeypatch.setattr(looks, "BUILTIN_LOOKS", retuned)
    assert dev._develop_algo_inputs() != baseline
    monkeypatch.setattr(looks, "BUILTIN_LOOKS", {k: original[k] for k in sorted(original)[::-1]})
    assert dev._develop_algo_inputs() == baseline


# ── Astro-pixel module coverage ────────────────────────────────────────


def test_astro_module_sources_are_hashed():
    """The eight astro pixel-producing modules ride the normal develop chain but
    wire through none of GRADE_ALGO_VERSION, _HASHED_SOURCES, or bracket's grouping
    — without this, a retune of e.g. astro_develop's stretch target changes output
    pixels while the provenance stamp claims the algorithm is unchanged."""
    inputs = dev._develop_algo_inputs()
    for module in (
        astro_calibrate,
        astro_sequence,
        astro_triage,
        astro_register,
        astro_stack,
        astro_develop,
        astro_meteors,
        astro_orchestrator,
    ):
        assert inspect.getsource(module) in inputs, module.__name__


def test_orchestrator_in_the_hash_does_not_close_the_import_cycle():
    """The orchestrator is the one astro module that reaches back into ``develop``
    (for the HE->DNG converter and the TIFF writer), so hashing it is the point
    where a cycle could close. It stays open only while those reaches are local
    imports inside function bodies — a module-level one fails this from either
    import direction, which is why both are exercised in a fresh interpreter."""
    service_dir = Path(dev.__file__).resolve().parents[3]
    photos = "gomoveshift.pipeline.photos"
    for first, second in (
        (f"{photos}.astro_orchestrator", f"{photos}.develop"),
        (f"{photos}.develop", f"{photos}.astro_orchestrator"),
    ):
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {first}; import {second}; "
                f"import {photos}.develop as d; d._develop_algo_inputs()",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(service_dir)},
            cwd=service_dir,
            check=True,
        )


def test_an_astro_tunable_participates(monkeypatch):
    """Read live, like every other tunable: a retune of the star-detection
    threshold moves which pixels stack, so the fingerprint must move with it."""
    baseline = dev._develop_algo_inputs()
    monkeypatch.setattr(astro_triage, "STAR_DETECT_SIGMA", astro_triage.STAR_DETECT_SIGMA + 1.0)
    assert dev._develop_algo_inputs() != baseline
