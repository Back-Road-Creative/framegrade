"""Frame-index pairing of the privacy mask with the picture.

The final encode's ``alphamerge`` / ``overlay`` pair by TIMESTAMP, and the mask is
stamped ``j / fps``. A source with timestamp gaps (a stitched cut file) therefore
took mask content from the wrong frame. These tests build a real source with a
gap and run the real filtergraph on CPU, then read back which picture frame and
which mask frame every output frame carries.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from redactcam import timeline as br

from gomoveshift.config import Config
from gomoveshift.pipeline import blur_gate
from gomoveshift.pipeline.single_pass import SinglePassProcessor
from gomoveshift.pipeline.stages.blur_detection import BlurDetectionStage

W, H, N, FPS = 64, 96, 24, 30
GAP_AT, GAP_S = 12, 0.5  # frames >= GAP_AT are shifted 0.5 s later: 15 frames of gap

pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg required",
)


def _ff(*args, stdin=None):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args], input=stdin, capture_output=True, timeout=120
    )
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout


def _picture_bytes(n: int) -> bytes:
    """Frame j: grey 128 with a white 2-row bar at y = 2j (identifies the picture frame)."""
    out = bytearray()
    for j in range(n):
        f = np.full((H, W), 128, np.uint8)
        f[2 * j : 2 * j + 2, :] = 235
        out += f.tobytes()
    return bytes(out)


def _mask_bytes(n: int) -> bytes:
    """Frame j: black with a white 4x4 square at x = 2j, y = 60 (identifies the mask frame)."""
    out = bytearray()
    for j in range(n):
        f = np.zeros((H, W), np.uint8)
        f[60:64, 2 * j : 2 * j + 4] = 255
        out += f.tobytes()
    return bytes(out)


def _raw_in(rate=FPS):
    return ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(rate), "-i", "-"]


def make_source(path: Path, n: int = N, gap: bool = True) -> Path:
    vf = f"setpts=(N/{FPS}+if(gte(N\\,{GAP_AT})\\,{GAP_S}\\,0))/TB" if gap else "null"
    _ff(
        *_raw_in(),
        "-vf",
        vf,
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        str(path),
        stdin=_picture_bytes(n),
    )
    return path


def make_mask(path: Path, n: int = N) -> Path:
    _ff(*_raw_in(), "-c:v", "ffv1", str(path), stdin=_mask_bytes(n))
    return path


def _render(src: Path, mask: Path, **kw) -> np.ndarray:
    fc = SinglePassProcessor(Config())._build_blur_filter_complex(
        color_filter="null",
        location="",
        font_path="",
        duration=2.0,
        blur_filter="negate",
        fade_video=False,
        video_width=W,
        video_height=H,
        skip_ost=True,
        output_fps="30",
        **kw,
    )
    raw = _ff(
        "-i", str(src), "-i", str(mask), "-filter_complex", fc,
        "-map", "[vout]", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    )  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def _pairs(frames: np.ndarray) -> list[tuple[int, int]]:
    """(picture index, mask index) carried by each output frame."""
    out = []
    for f in frames:
        pic = int(np.argmax(f[:56].astype(int).mean(axis=1))) // 2
        band = f[56:68].astype(int)
        col = int(np.argmax(np.abs(band - np.median(band)).max(axis=0)))
        out.append((pic, col // 2))
    return out


@pytest.fixture(scope="module")
def gap_inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("gap")
    return make_source(d / "src.mkv"), make_mask(d / "mask.mkv")


def test_source_really_has_a_timestamp_gap(gap_inputs):
    src, _ = gap_inputs
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", str(src)],
        capture_output=True, text=True,
    ).stdout.split()  # fmt: skip
    t = [float(x) for x in out]
    assert len(t) == N
    assert t[GAP_AT] - t[GAP_AT - 1] > 0.4


@pytest.mark.parametrize("kw", [{}, {"index_fps": float(FPS)}], ids=["default", "explicit-rate"])
def test_mask_frame_j_lands_on_picture_frame_j_across_a_timestamp_gap(gap_inputs, kw):
    src, mask = gap_inputs
    pairs = _pairs(_render(src, mask, **kw))
    assert len(pairs) == N
    assert [p for p, _ in pairs] == list(range(N))
    assert all(p == m for p, m in pairs), pairs


def test_gapless_source_still_pairs_one_to_one(tmp_path):
    src = make_source(tmp_path / "s.mkv", gap=False)
    pairs = _pairs(_render(src, make_mask(tmp_path / "m.mkv")))
    assert pairs == [(j, j) for j in range(N)]


# ── three-way count gate ─────────────────────────────────────────────────────


def _sidecar(path: Path, frame_count: int) -> Path:
    tl = br.build_timeline(
        {0: [(1, 1, 4, 4)]},
        source_hash="h",
        source_fps=float(FPS),
        source_frame_count=frame_count,
        frame_width=W,
        frame_height=H,
        box_dilation_px=0,
        sample_fps=8.0,
        conf=0.4,
    )
    br.write_sidecar(path, tl)
    return path


def test_pairing_gate_passes_when_all_three_counts_agree(tmp_path):
    v = make_source(tmp_path / "v.mkv")
    m = make_mask(tmp_path / "m.mkv")
    assert blur_gate.frame_pairing_error(v, m, _sidecar(tmp_path / "s.json", N)) is None


@pytest.mark.parametrize(
    "mask_n,side_n",
    [(N - 3, N), (N, N + 8), (N + 5, N + 5)],
    ids=["short-mask", "long-sidecar", "both-long"],
)
def test_pairing_gate_blocks_on_any_disagreement_and_names_the_three_numbers(
    tmp_path, mask_n, side_n
):
    v = make_source(tmp_path / "v.mkv")
    m = make_mask(tmp_path / "m.mkv", n=mask_n)
    err = blur_gate.frame_pairing_error(v, m, _sidecar(tmp_path / "s.json", side_n))
    assert err and "BLUR GATE" in err
    for n in (N, mask_n, side_n):
        assert str(n) in err


def test_pairing_gate_fails_closed_when_the_sidecar_is_missing(tmp_path):
    v = make_source(tmp_path / "v.mkv")
    m = make_mask(tmp_path / "m.mkv")
    assert blur_gate.frame_pairing_error(v, m, tmp_path / "absent.json")


# ── resume: a sidecar/mask from the other index space is rebuilt, never reused ─


def test_stale_count_sidecar_is_redetected_not_reused(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from tests.test_blur_detection import _ctx

    ctx = _ctx(tmp_path)
    ctx.current_video = make_source(tmp_path / "v.mkv")
    monkeypatch.setattr("gomoveshift.pipeline.stop_processor.compute_file_hash", lambda p: "h")
    sidecar = ctx.paths.stages_dir / "proj_blur_regions.json"
    _sidecar(sidecar, N + 8)  # a pre-fix detection: more frames than the container
    mask = ctx.paths.stages_dir / "proj_blur_mask.mkv"
    make_mask(mask, n=N + 8)
    calls = []

    def fake_detect(*a, **k):
        calls.append(1)
        return {0: [(1, 1, 4, 4)]}, N, 1, 0

    monkeypatch.setattr("redactcam.detect_and_track", fake_detect)
    monkeypatch.setattr(
        "redactcam.materialize_mask_video", lambda tl, p, **k: make_mask(p, n=tl.source_frame_count)
    )
    monkeypatch.setattr(br, "probe_video", lambda p: (float(FPS), N, W, H))
    res = BlurDetectionStage().run(ctx)
    assert res.success, res.error
    assert calls, "a count-mismatched sidecar was reused instead of rebuilt"
    assert br.load_sidecar(sidecar).source_frame_count == N
    del MagicMock


def test_detection_that_disagrees_with_the_container_is_a_blocking_error(tmp_path, monkeypatch):
    from tests.test_blur_detection import _ctx

    ctx = _ctx(tmp_path)
    ctx.current_video = make_source(tmp_path / "v.mkv")
    monkeypatch.setattr("gomoveshift.pipeline.stop_processor.compute_file_hash", lambda p: "h")
    monkeypatch.setattr(
        "redactcam.detect_and_track", lambda *a, **k: ({0: [(1, 1, 4, 4)]}, N + 8, 1, 0)
    )
    monkeypatch.setattr(
        "redactcam.materialize_mask_video", lambda tl, p, **k: make_mask(p, n=tl.source_frame_count)
    )
    monkeypatch.setattr(br, "probe_video", lambda p: (float(FPS), N, W, H))
    res = BlurDetectionStage().run(ctx)
    assert not res.success
    for n in (N, N + 8):
        assert str(n) in res.error


# ── reuse signature carries the renderer identity ────────────────────────────


def test_renderer_identity_names_redactcam_version_and_decode_mode(monkeypatch):
    monkeypatch.delenv("REDACTCAM_DECODE", raising=False)
    a = blur_gate.renderer_identity()
    assert set(a) == {"redactcam_version", "decode_mode"} and a["redactcam_version"]
    monkeypatch.setenv("REDACTCAM_DECODE", "cv2")
    assert blur_gate.renderer_identity()["decode_mode"] == "cv2"
    assert blur_gate.renderer_identity() != a


def test_verdict_signature_changes_with_the_renderer(tmp_path, monkeypatch):
    cfg = Config()
    p = tmp_path / "x.bin"
    p.write_bytes(b"x")
    monkeypatch.delenv("REDACTCAM_DECODE", raising=False)
    a = blur_gate._cabin_coverage_signature(cfg.drive, p, p, p, None)
    monkeypatch.setenv("REDACTCAM_DECODE", "nvdec")
    b = blur_gate._cabin_coverage_signature(cfg.drive, p, p, p, None)
    assert a != b
    assert "redactcam_version" in a


def test_day_mask_evidence_changes_with_the_renderer(monkeypatch):
    from gomoveshift.pipeline import day_master_encoder as dme

    kw = dict(
        person_on=False, person_model=None, vehicle_model=None, face_model=None, plate_model=None
    )
    monkeypatch.delenv("REDACTCAM_DECODE", raising=False)
    a = dme._current_blur_evidence(Config().drive, **kw)
    monkeypatch.setenv("REDACTCAM_DECODE", "cv2")
    assert dme._current_blur_evidence(Config().drive, **kw) != a


# ── probe cost: nb_frames first, count_packets only as the fallback ───────────


class _Probe:
    """Stub ``subprocess.run`` in blur_gate; ``answers`` maps entry name -> stdout/None."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, cmd, **kw):
        entry = next(c for c in cmd if c.startswith("stream="))
        if entry == "stream=nb_frames":
            assert "-count_packets" not in cmd, "nb_frames probe must not read the file"
        self.calls.append(entry)
        out = self.answers.get(entry)
        if out is None:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def _stub(monkeypatch, answers):
    probe = _Probe(answers)
    monkeypatch.setattr(blur_gate.subprocess, "run", probe)
    return probe


def test_nb_frames_is_read_without_counting_packets(monkeypatch):
    probe = _stub(monkeypatch, {"stream=nb_frames": "306377\n"})
    assert blur_gate.container_frame_count(Path("/x/day_raw.mp4")) == 306377
    assert probe.calls == ["stream=nb_frames"]  # a 91 GB file is never read through


@pytest.mark.parametrize("missing", ["N/A\n", "0\n", "\n", "-3\n"])
def test_missing_nb_frames_falls_back_to_count_packets(monkeypatch, missing):
    probe = _stub(monkeypatch, {"stream=nb_frames": missing, "stream=nb_read_packets": "24\n"})
    assert blur_gate.container_frame_count(Path("/x/mask.mkv")) == 24
    assert probe.calls == ["stream=nb_frames", "stream=nb_read_packets"]


def test_both_probes_failing_is_minus_one(monkeypatch):
    _stub(monkeypatch, {})
    assert blur_gate.container_frame_count(Path("/x/v.mp4")) == -1
    _stub(monkeypatch, {"stream=nb_frames": "N/A\n", "stream=nb_read_packets": "N/A\n"})
    assert blur_gate.container_frame_count(Path("/x/v.mp4")) == -1


# ── day-master reuse: a cached mask on another frame count is rebuilt ─────────


def test_day_master_rebuilds_a_cached_mask_whose_count_differs_from_the_source(
    tmp_path, monkeypatch
):
    from gomoveshift.pipeline import day_master_encoder as dme
    from tests import test_day_master_blur as tdmb

    tdmb.TestBuildDayBlurMask()._no_network_models(monkeypatch)
    monkeypatch.setattr("redactcam.timeline.probe_video", lambda p: (30.0, 90, 64, 96))
    detects = []

    def _detect(*a, **k):
        detects.append(1)
        return ({0: [(10, 10, 20, 20)]}, 90, 1, 0)

    monkeypatch.setattr("redactcam.detect_and_track", _detect)
    monkeypatch.setattr(
        "redactcam.materialize_mask_video", lambda tl, p, **k: (p.write_bytes(b"mask"), p)[1]
    )
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    cfg = Config()
    cfg.drive.blur_vehicle_cabin_frac = 0.0
    mask, sidecar = tmp_path / "m.mkv", tmp_path / "r.json"
    count = {"n": 90}  # only the probe is stubbed; _counts_allow_reuse is the real one
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: count["n"])

    dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
    dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
    assert detects == [1], "control: matching counts must still reuse"
    count["n"] = 80  # the source (or mask) now counts another number of frames
    dme._build_day_blur_mask(raw, cfg, mask, sidecar, "d")
    assert detects == [1, 1], "a count-mismatched cached mask was reused"
