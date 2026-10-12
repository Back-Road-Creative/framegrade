"""Star trails: the streaming maximum blend, its calibration reuse, and the CLI wiring.
Synthetic float32 frames only -- no RAW fixtures; decode/develop/export are monkeypatched at
their seams, the ``test_astro_orchestrator.py`` idiom."""

import gc
import inspect
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gomoveshift.pipeline.photos.astro_orchestrator as om
import gomoveshift.pipeline.photos.astro_sequence as astro_sequence_mod
import gomoveshift.pipeline.photos.astro_trails as trails
import gomoveshift.pipeline.photos.develop as develop_mod
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.pipeline.photos.astro_calibrate import AstroDarkIdentityMismatchError
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos.preflight import estimate_astro_stack_ram_gb

from test_astro_calibrate import _identity  # the CaptureIdentity builder, defined once

MP = 20.8768  # a typical 20.9 MP body's megapixels, as EXIF reports them


def _frame(lit, shape=(8, 8), sky=0.02, value=0.9):
    frame = np.full((*shape, 3), sky, dtype=np.float32)
    for y, x in lit:
        frame[y, x] = value
    return frame


def _developed(rgb, sky_mask=None):
    return SimpleNamespace(rgb_uint16=rgb.astype(np.uint16))


def _never_chunk(*a, **k):
    pytest.fail("star trails must never be chunked for RAM")


def test_every_frames_star_survives_at_its_own_position_instead_of_being_averaged():
    """The arc: a star that moved a pixel per frame reads FULL brightness at all five
    positions -- a mean would leave ~1/5 of it at each, i.e. no trail."""
    trail, n_frames = trails.accumulate_max([_frame([(2, i)]) for i in range(5)])
    assert n_frames == 5
    assert [trail[2, i, 0] for i in range(5)] == pytest.approx([0.9] * 5)
    assert trail[7, 7, 0] == pytest.approx(0.02)  # untouched sky stays sky


@pytest.mark.parametrize("n", [3, 30])
def test_exactly_one_input_frame_is_resident_however_many_frames_there_are(n):
    """The streaming claim on memory SHAPE: the only live input frame is the one just
    handed over -- never a list of all n."""
    refs, live_after_each = [], []

    def _stream():
        for i in range(n):
            frame = _frame([(1, i % 8)])
            refs.append(weakref.ref(frame))
            yield frame
            del frame
            gc.collect()
            live_after_each.append(sum(r() is not None for r in refs))

    trail, count = trails.accumulate_max(_stream())
    assert count == n and trail.shape == (8, 8, 3)
    assert max(live_after_each) == 1


def test_render_trail_calibrates_every_light_and_streams_them_as_an_iterator(monkeypatch, tmp_path):
    """Max-blend makes dark/hot-pixel calibration non-optional (an uncorrected hot pixel is
    permanent), and frames must reach the blend as an ITERATOR -- a materialised list of the
    night is the memory ceiling this path exists to avoid."""
    calls, seen, fold = [], {}, trails.accumulate_max
    # The master-dark seam is now the validating one; identity refusal has its own test below.
    monkeypatch.setattr(
        trails, "build_validated_master_dark", lambda li, dk: np.full((4, 4), 3.0, np.float32)
    )
    monkeypatch.setattr(trails, "hot_pixel_map", lambda md: np.zeros((4, 4), bool))

    def _decode(path, master_dark, hotmap):
        calls.append((master_dark, hotmap))
        return SimpleNamespace(rgb=_frame([(0, 0)], shape=(4, 4))), None

    def _spy(frames):
        seen["frames"] = frames
        return fold(frames)

    monkeypatch.setattr(trails, "decode_with_he_fallback", _decode)
    monkeypatch.setattr(trails, "accumulate_max", _spy)
    monkeypatch.setattr(trails, "develop_stack", _developed)
    monkeypatch.setattr(develop_mod, "export_tiff16", lambda rgb, out_path: Path(out_path))
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(3)]

    progress = []
    path, used, _dropped = trails.render_trail(
        lights,
        tmp_path / "t.tif",
        [tmp_path / "d.NEF"],
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert used == 3 and path == tmp_path / "t.tif"
    assert len(calls) == 3 and all(md is not None and hm is not None for md, hm in calls)
    assert iter(seen["frames"]) is seen["frames"]  # a generator, never a list of the night
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_trails_flag_renders_one_master_per_run_and_never_chunks_for_ram(
    monkeypatch, tmp_path, capsys
):
    """15 frames x 20.8768 MP costs the stack path 18.35 GB and a 90-frame night ~63.9 GB,
    which is why it chunks; a streaming maximum inherits neither, so a run is never split
    and the gate it calls carries no frame-count term at all."""
    lights = [tmp_path / f"IMG_{i:04d}.NEF" for i in range(9)]
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "/opt/dngc")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [tuple(lights)])
    monkeypatch.setattr(astro_sequence_mod, "chunk_sequence", _never_chunk)
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: MP)
    gates, seen = [], {}
    monkeypatch.setattr(trails, "preflight_trail", lambda d, **kw: gates.append(kw))

    def _render(paths, out_path, dark_paths=(), sky_mask=None):
        seen["paths"] = list(paths)
        return Path(out_path), len(seen["paths"]), ()

    monkeypatch.setattr(trails, "render_trail", _render)
    argv = f"photos astro {' '.join(str(p) for p in lights)} --trails --output-dir {tmp_path}"

    assert photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None) == 0

    assert len(seen["paths"]) == 9  # the whole run, one master
    assert "frame_count" not in gates[0]
    assert "frame_count" not in inspect.signature(trails.estimate_trail_ram_gb).parameters
    assert trails.estimate_trail_ram_gb(MP) < 0.2 * estimate_astro_stack_ram_gb(90, MP)
    assert "astro_trail_" in capsys.readouterr().out  # never over a stack of the same frames


def test_a_ten_second_dark_among_fifteen_second_darks_refuses_before_any_light_decodes(
    monkeypatch, tmp_path
):
    """A max blend keeps the brightest value each pixel ever took, so an under-subtracting
    dark burns in permanently -- the trails path must refuse the same mismatch the stack
    path does, naming the file and the field. A typical setting (15 s exposures at
    ISO 6400) with one 10 s frame among the darks."""
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(3)]
    darks = [tmp_path / f"DSC_000{n}.NEF" for n in (7, 8, 9)]
    identities = {p: _identity(path=p, iso=6400.0, exposure_s=15.0) for p in [*lights, *darks]}
    identities[darks[-1]] = _identity(path=darks[-1], iso=6400.0, exposure_s=10.0)
    # Patched on the ORCHESTRATOR: trails must reach the real validating seam, not a stub.
    monkeypatch.setattr(om, "read_capture_identity", lambda p: identities[Path(p)], raising=False)
    monkeypatch.setattr(om, "read_raw_mosaic", lambda p: np.zeros((4, 4), np.uint16))

    def _boom(*a, **k):
        raise AssertionError("refuse before decoding any light (~1 min/frame)")

    monkeypatch.setattr(trails, "decode_with_he_fallback", _boom)

    with pytest.raises(
        AstroDarkIdentityMismatchError,
        match=r"DSC_0009\.NEF: dark exposure_s does not match light .*15\.0.*10\.0",
    ):
        trails.render_trail(lights, tmp_path / "t.tif", darks)


def test_a_trail_without_darks_reads_no_capture_identity_and_calibrates_nothing(
    monkeypatch, tmp_path
):
    calls = []

    def _decode(path, master_dark, hotmap):
        calls.append((master_dark, hotmap))
        return SimpleNamespace(rgb=_frame([(0, 0)], shape=(4, 4))), None

    def _boom(*a, **k):
        raise AssertionError("the no-darks path must read no capture identity")

    monkeypatch.setattr(om, "read_capture_identity", _boom, raising=False)
    monkeypatch.setattr(trails, "decode_with_he_fallback", _decode)
    monkeypatch.setattr(trails, "develop_stack", _developed)
    monkeypatch.setattr(develop_mod, "export_tiff16", lambda rgb, out_path: Path(out_path))
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(3)]

    path, used, dropped = trails.render_trail(lights, tmp_path / "t.tif")

    assert used == 3 and dropped == () and path == tmp_path / "t.tif"
    assert calls == [(None, None)] * 3
