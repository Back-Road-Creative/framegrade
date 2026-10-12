"""Night timelapse: deflicker, streaming render, CLI wiring. Synthetic float32 frames only;
decode and ffmpeg are monkeypatched at their seams, the ``test_astro_trails.py`` idiom."""

import gc
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gomoveshift.pipeline.photos.astro_sequence as astro_sequence_mod
import gomoveshift.pipeline.photos.astro_timelapse as timelapse
import gomoveshift.pipeline.photos.develop as develop_mod
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos.reel import build_encode_cmd

SHAPE = (24, 24)  # even on both axes, so the render's own even-crop is a no-op here
PATCH = (slice(2, 6), slice(2, 6))  # one input value in every frame: the flicker probe
MP = 20.8768  # a typical 20.9 MP body's megapixels, as EXIF reports them


def _night_frame(i):
    """A night that brightens, stars turning over, ``PATCH`` pinned at 0.35 in every frame --
    a change THERE in the output is the tone curve moving, never the sky."""
    rng = np.random.default_rng(i)
    frame = np.full((*SHAPE, 3), 0.02 + 0.004 * i, dtype=np.float32)
    frame += (0.002 * rng.standard_normal((*SHAPE, 3))).astype(np.float32)
    for s in range(3 + i):  # more stars visible as the night darkens
        frame[(7 + s * 2) % SHAPE[0], (5 + s * 3 + i) % SHAPE[1]] = 0.8
    frame[PATCH] = 0.35
    return np.clip(frame, 0.0, None).astype(np.float32)


class _FakeProc:
    """ffmpeg stand-in: own pipe, records each frame, touches the output (the render's success
    test is exit 0 + file exists); ``hook`` fires per write. No ``kill``: never reached here."""

    hook = last = None

    def __init__(self, cmd, stdin=None, stderr=None):
        self.cmd, self.stdin, self.frames = cmd, self, []
        _FakeProc.last = self

    def write(self, data):
        self.frames.append(data)
        if _FakeProc.hook:
            _FakeProc.hook()

    def close(self):
        pass

    def wait(self, timeout=None):
        Path(self.cmd[-1]).write_bytes(b"")
        return 0


def _render(monkeypatch, tmp_path, decode, n, **kwargs):  # -> (result, the fake ffmpeg)
    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(timelapse, "decode_with_he_fallback", decode)
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(n)]
    return timelapse.render_timelapse(lights, tmp_path / "night.mp4", **kwargs), _FakeProc.last


def test_progress_reports_each_encoded_frame(monkeypatch, tmp_path):
    progress = []

    def _decode(path, master_dark, hotmap):
        return SimpleNamespace(rgb=_night_frame(int(Path(path).stem[4:]))), None

    (out, used, rejected), _proc = _render(
        monkeypatch,
        tmp_path,
        _decode,
        3,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert (out, used, rejected) == (tmp_path / "night.mp4", 3, ())
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_one_tone_from_the_middle_frame_and_one_frame_resident_at_a_time(monkeypatch, tmp_path):
    """Deflicker: a patch whose INPUT is identical in all five synthetic 24x24 frames rendered
    here must come out identical -- the shared base holds it at 190 of 255, where per-frame
    fitting spreads it 176-206. That tone is the MIDDLE frame's (a night drifts one way, so the
    middle halves the worst case either side); no frame outlives its turn."""
    seen, refs, live, real = {}, [], [], timelapse.develop_stack

    def _decode(path, master_dark, hotmap):
        frame = _night_frame(int(Path(path).stem[4:]))
        refs.append(weakref.ref(frame))
        return SimpleNamespace(rgb=frame), None

    def _spy(rgb, sky_mask=None):
        seen.setdefault("sky", float(rgb[0, 0, 0]))  # frame i's sky is 0.02 + 0.004i, plus noise
        return real(rgb, sky_mask=sky_mask)

    def _count():  # per pipe write: (frames still resident, frames decoded so far)
        gc.collect()
        live.append((sum(r() is not None for r in refs), len(refs)))

    monkeypatch.setattr(timelapse, "develop_stack", _spy)
    monkeypatch.setattr(_FakeProc, "hook", staticmethod(_count))
    (out, used, rejected), proc = _render(monkeypatch, tmp_path, _decode, 5)

    assert (used, rejected, out) == (5, (), tmp_path / "night.mp4")
    assert seen["sky"] == pytest.approx(0.02 + 0.004 * 2, abs=0.002)  # frame 2 of 0..4
    assert [n for _, n in live] == [2, 3, 4, 5, 6]  # decode and write interleave, not two phases
    assert max(r for r, _ in live) == 0  # released BEFORE the write, reference frame included
    written = [np.frombuffer(b, np.uint8).reshape(*SHAPE, 3) for b in proc.frames]
    levels = [float(f[PATCH][..., 0].mean()) for f in written]
    assert max(levels) - min(levels) <= 1.0, levels


def test_a_frame_that_cannot_decode_is_named_and_never_padded_over(monkeypatch, tmp_path):
    """5 in, 1 undecodable: 4 written, the 5th named, nothing duplicated or interpolated to
    pad the clip back out; the encode is the reel's own invocation, unchanged."""

    def _decode(path, master_dark, hotmap):
        if Path(path).stem == "IMG_3":
            return None, "unpack failed"
        return SimpleNamespace(rgb=_night_frame(int(Path(path).stem[4:]))), None

    (out, used, rejected), proc = _render(monkeypatch, tmp_path, _decode, 5)

    assert used == 4 and len(proc.frames) == 4
    assert [(p.name, why) for p, why in rejected] == [("IMG_3.NEF", "unpack failed")]
    assert proc.cmd == build_encode_cmd(out, timelapse.TIMELAPSE_FPS, (SHAPE[1], SHAPE[0]))


def test_every_frame_rejected_raises_and_leaves_no_stub_behind(monkeypatch, tmp_path):
    """ffmpeg creates an empty but valid container (ftyp/free/empty mdat) the moment it
    starts, so ``out.is_file()`` alone accepts a stub and ``used == 0`` must be checked.
    A render that produced no frames must RAISE, and must not leave the stub behind for
    an operator to find."""

    calls = {"n": 0}

    def _decode(path, master_dark, hotmap):
        # Call 1 is the reference (its tone must exist or the render refuses earlier and
        # never opens ffmpeg at all); every frame of the sequence itself then fails.
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(rgb=_night_frame(2)), None
        return None, "unpack failed"

    with pytest.raises(timelapse.TimelapseError) as excinfo:
        _render(monkeypatch, tmp_path, _decode, 5)

    assert "no frames" in str(excinfo.value)
    assert not (tmp_path / "night.mp4").exists()


def test_fps_is_the_callers_choice_and_reaches_both_the_encode_and_the_duration(
    monkeypatch, tmp_path
):
    """90 frames at 24 fps is 3.75s -- and a 15-frame sequence is 0.6s, which flashes a
    picture and ends. Frame count is fixed by the capture, so fps is the only lever a
    render has over how long the clip runs."""

    def _decode(path, master_dark, hotmap):
        return SimpleNamespace(rgb=_night_frame(int(Path(path).stem[4:]))), None

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(timelapse, "decode_with_he_fallback", _decode)
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(5)]
    out, used, _ = timelapse.render_timelapse(lights, tmp_path / "night.mp4", fps=8)

    assert used == 5
    assert _FakeProc.last.cmd == build_encode_cmd(out, 8, (SHAPE[1], SHAPE[0]))


def test_timelapse_flag_renders_one_clip_per_run_and_reports_its_duration(
    monkeypatch, tmp_path, capsys
):
    """A timelapse wants the whole run: chunking exists for stack RAM and field rotation,
    neither of which a streaming, never-registered render has -- so --stack-len is ignored."""
    lights = [tmp_path / f"IMG_{i:04d}.NEF" for i in range(9)]
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "/opt/dngc")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [tuple(lights)])
    monkeypatch.setattr(astro_sequence_mod, "chunk_sequence", lambda *a, **k: pytest.fail("chunk"))
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: MP)
    monkeypatch.setattr(timelapse, "preflight_timelapse", lambda d, **kw: None)
    monkeypatch.setattr(
        timelapse, "render_timelapse", lambda ps, o, *a, **kw: (Path(o), len(ps), ())
    )
    argv = f"photos astro {' '.join(str(p) for p in lights)} --timelapse --output-dir {tmp_path}"

    assert photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None) == 0

    out = capsys.readouterr().out
    assert "9 frames at 24 fps = 0.4s of video" in out
    assert "9 frame(s) rendered, 0 dropped" in out  # the whole run, one clip
    assert "astro_timelapse_" in out and ".mp4" in out
    with pytest.raises(SystemExit):
        _build_parser().parse_args(f"{argv} --trails".split())


def test_scale_height_shrinks_every_frame_and_the_encode_geometry_together(monkeypatch, tmp_path):
    """A full-sensor render develops each 20.9 MP frame whole: ``remove_gradient`` fits a
    polynomial over tens of millions of points and the stretch allocates several 500 MB
    float64 temporaries, so a full-size render is far slower per frame than the decode and
    produces a video nothing plays.

    Scaling must happen BEFORE develop to buy any of that back, which means the reference
    frame scales too -- ``base.sky_mask`` is what ``develop_stack_with_params`` fits the
    gradient under, so a full-size mask against a scaled frame is a shape error, not a
    slow path. Asserting the encode geometry AND the payload size together is what pins
    that: a frame scaled after develop would still reach ffmpeg at the wrong size.
    """

    def _decode(path, master_dark, hotmap):
        return SimpleNamespace(rgb=_night_frame(int(Path(path).stem[4:]))), None

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(timelapse, "decode_with_he_fallback", _decode)
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(4)]
    out, used, rejected = timelapse.render_timelapse(
        lights, tmp_path / "night.mp4", scale_height=12
    )

    assert (used, rejected) == (4, ())
    # SHAPE is square, so half the height is half the width: 24x24 -> 12x12.
    assert _FakeProc.last.cmd == build_encode_cmd(out, timelapse.TIMELAPSE_FPS, (12, 12))
    assert {len(f) for f in _FakeProc.last.frames} == {12 * 12 * 3}


def test_scale_height_above_the_sensor_never_upscales(monkeypatch, tmp_path):
    """A 1080p request against a smaller source must not invent pixels: the cap is a
    ceiling, not a target. Guards a request for --timelapse-height 2160 on a
    body whose frames are shorter than that."""

    def _decode(path, master_dark, hotmap):
        return SimpleNamespace(rgb=_night_frame(int(Path(path).stem[4:]))), None

    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(timelapse, "decode_with_he_fallback", _decode)
    lights = [tmp_path / f"IMG_{i}.NEF" for i in range(3)]
    out, used, _ = timelapse.render_timelapse(lights, tmp_path / "night.mp4", scale_height=1080)

    assert used == 3
    assert _FakeProc.last.cmd == build_encode_cmd(
        out, timelapse.TIMELAPSE_FPS, (SHAPE[1], SHAPE[0])
    )


def test_cli_defaults_the_timelapse_height_and_takes_zero_as_uncapped(
    monkeypatch, tmp_path, capsys
):
    """4K is the DEFAULT, not an opt-in, and 0 is the way back to the full sensor.

    Uncapped is not a defensible default: develop fits a gradient polynomial over every
    pixel, so on a 20.9 MP body a full-sensor night costs many times more per frame than
    at 2160, for a file no player wants and no display can show. A caller who genuinely
    wants every sensor row asks for 0; nobody should pay that cost by omission.
    """
    lights = [tmp_path / f"IMG_{i:04d}.NEF" for i in range(6)]
    seen = {}
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "/opt/dngc")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [tuple(lights)])
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: MP)
    monkeypatch.setattr(timelapse, "preflight_timelapse", lambda d, **kw: None)

    def _fake(ps, o, *a, **kw):
        seen["scale_height"] = kw.get("scale_height")
        return (Path(o), len(ps), ())

    monkeypatch.setattr(timelapse, "render_timelapse", _fake)
    argv = f"photos astro {' '.join(str(p) for p in lights)} --timelapse --output-dir {tmp_path}"

    assert photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None) == 0
    assert seen["scale_height"] == timelapse.TIMELAPSE_HEIGHT_DEFAULT == 2160

    parsed = _build_parser().parse_args(f"{argv} --timelapse-height 0".split())
    assert photos_handler.cmd_photos(parsed, None) == 0
    assert seen["scale_height"] is None  # 0 means "no cap", never a zero-pixel render

    parsed = _build_parser().parse_args(f"{argv} --timelapse-height -4".split())
    assert photos_handler.cmd_photos(parsed, None) == 1
    assert "--timelapse-height must be positive" in capsys.readouterr().err


def test_cli_threads_timelapse_height_through_and_refuses_a_bad_one(monkeypatch, tmp_path, capsys):
    """The render already took ``scale_height``; only the CLI could not reach it. Validated
    upfront beside --timelapse-fps rather than inside cv2.resize hours into a render."""
    lights = [tmp_path / f"IMG_{i:04d}.NEF" for i in range(6)]
    seen = {}
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "/opt/dngc")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [tuple(lights)])
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: MP)
    monkeypatch.setattr(timelapse, "preflight_timelapse", lambda d, **kw: None)

    def _fake(ps, o, *a, **kw):
        seen["scale_height"] = kw.get("scale_height")
        return (Path(o), len(ps), ())

    monkeypatch.setattr(timelapse, "render_timelapse", _fake)
    argv = f"photos astro {' '.join(str(p) for p in lights)} --timelapse --output-dir {tmp_path}"

    parsed = _build_parser().parse_args(f"{argv} --timelapse-height 1080".split())
    assert photos_handler.cmd_photos(parsed, None) == 0
    assert seen["scale_height"] == 1080
