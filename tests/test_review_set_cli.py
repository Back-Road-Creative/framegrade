"""``review-set`` CLI wiring (synthetic fixtures only)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.cli import _COMMAND_MAP, cmd_review_set
from gomoveshift.cli.handlers import review_set as h
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline import review_set_scan as rs
from gomoveshift.pipeline.stop_mask import MotionState


def _scores_npz(tmp_path, n=20):
    a = np.zeros((n, rs.NUM_COLUMNS))
    a[:, 0] = np.arange(n)
    a[:, rs.COL_BRIGHTNESS] = 0.6
    a[:, rs.COL_COLOR_VIBRANCY] = 0.5
    p = tmp_path / "scores_synthetic.npz"
    np.savez_compressed(p, scores=a)
    return p


def _mask_npz(tmp_path, n=20):
    states = np.zeros(n, dtype=np.int8)
    states[5:10] = int(MotionState.STOPPED)
    p = tmp_path / "stop_mask_2026-01-02.npz"
    np.savez_compressed(p, mask=states == int(MotionState.STOPPED), states=states)
    return p


def _run(argv, capsys):
    args = _build_parser().parse_args(argv)
    rc = cmd_review_set(args, SimpleNamespace())
    return rc, capsys.readouterr()


def test_registered_and_wired():
    assert _COMMAND_MAP["review-set"] is cmd_review_set
    assert "review-set" in (_build_parser().epilog or "")


def test_parse_all_three():
    p = _build_parser()
    a = p.parse_args(["review-set", "scan", "--scores", "s.npz"])
    assert (a.command, a.review_set_action, a.out) == ("review-set", "scan", None)
    a = p.parse_args(["review-set", "stops", "--stop-mask", "m.npz", "--out", "o.json"])
    assert a.review_set_action == "stops" and a.receipt is None
    a = p.parse_args(
        ["review-set", "render", "--source", "a", "--output", "b", "--out-dir", "d",
         "--pair", "1:2", "--pair", "3:4", "--synthetic"]
    )  # fmt: skip
    assert a.pair == ["1:2", "3:4"] and a.synthetic and a.excerpt_seconds == 3.0


def test_action_required():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["review-set"])


def test_existing_scan_command_unchanged():
    a = _build_parser().parse_args(["scan", "somefile.mp4"])
    assert a.command == "scan" and not hasattr(a, "review_set_action")


def test_scan_stdout_and_out(tmp_path, capsys):
    scores, mask = _scores_npz(tmp_path), _mask_npz(tmp_path)
    rc, cap = _run(
        ["review-set", "scan", "--scores", str(scores), "--stop-mask", str(mask)], capsys
    )
    assert rc == 0
    doc = json.loads(cap.out)
    assert doc["conditions"]["daylight"]["windows"]
    assert doc["conditions"]["creeping"]["status"] == "ok"
    out = tmp_path / "sub" / "scan.json"
    rc, _ = _run(["review-set", "scan", "--scores", str(scores), "--out", str(out)], capsys)
    assert rc == 0 and json.loads(out.read_text())["conditions"]["gps_gap"]["status"] == "no_input"


def test_scan_gps_and_tonemap_passed_through(tmp_path, capsys):
    scores = _scores_npz(tmp_path)
    gps = tmp_path / "gps.json"
    gps.write_text(json.dumps([[0, 40.0, -100.0], [19, 40.1, -100.1]]))
    tm = tmp_path / "x.tonemap.json"
    tm.write_text(json.dumps({"tonemap": "hable"}))
    rc, cap = _run(
        ["review-set", "scan", "--scores", str(scores), "--gps", str(gps), "--tonemap", str(tm)],
        capsys,
    )
    conds = json.loads(cap.out)["conditions"]
    assert rc == 0
    assert conds["hdr_gradients"]["status"] == "ok"
    assert conds["gps_gap"]["status"] == "ok"


def test_scan_calls_library_with_arguments(tmp_path, capsys, monkeypatch):
    seen = {}

    def fake(scores, **kw):
        seen["shape"], seen["kw"] = scores.shape, kw
        return {"conditions": {}}

    monkeypatch.setattr(rs, "scan_review_set", fake)
    _run(["review-set", "scan", "--scores", str(_scores_npz(tmp_path))], capsys)
    assert seen["shape"] == (20, rs.NUM_COLUMNS)
    assert seen["kw"] == {"states": None, "gps_points": None, "tonemap": None}


def test_stops_without_and_with_receipt(tmp_path, capsys):
    mask = _mask_npz(tmp_path)
    out = tmp_path / "stops.json"
    rc, _ = _run(["review-set", "stops", "--stop-mask", str(mask), "--out", str(out)], capsys)
    doc = json.loads(out.read_text())
    assert rc == 0 and doc["sequence"] == "2026-01-02"
    assert doc["output_times"]["available"] is False


def test_stops_passes_receipt_and_sequence(tmp_path, capsys, monkeypatch):
    from gomoveshift.pipeline import review_set_stops as st

    seen = {}
    monkeypatch.setattr(
        st, "build_stops", lambda p, r=None, *, sequence=None: seen.update(r=r, s=sequence) or {}
    )
    monkeypatch.setattr(st, "write_stops", lambda doc, path: seen.update(path=path))
    receipt = tmp_path / "r.json"
    receipt.write_text("{}")
    out = tmp_path / "o.json"
    rc, _ = _run(
        ["review-set", "stops", "--stop-mask", str(_mask_npz(tmp_path)), "--receipt",
         str(receipt), "--sequence", "seqA", "--out", str(out)],
        capsys,
    )  # fmt: skip
    assert rc == 0 and seen == {"r": receipt, "s": "seqA", "path": out}


def _lavfi(path, color):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c={color}:s=64x48:r=10:d=3", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


def test_render_end_to_end_synthetic(tmp_path, capsys):
    src, out = tmp_path / "src.mp4", tmp_path / "out.mp4"
    _lavfi(src, "red")
    _lavfi(out, "blue")
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps([[0.5, 0.5]]))
    dest = tmp_path / "set"
    from gomoveshift.config import get_config

    args = _build_parser().parse_args(
        ["review-set", "render", "--source", str(src), "--output", str(out), "--pairs",
         str(pairs), "--pair", "1.5:1.0", "--out-dir", str(dest), "--excerpt-seconds", "1",
         "--synthetic"]
    )  # fmt: skip
    assert cmd_review_set(args, get_config()) == 0
    capsys.readouterr()
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov["synthetic"] is True and prov["pairs"] == [[0.5, 0.5], [1.5, 1.0]]
    assert (dest / "pair_001_output.png").is_file()
    # a second run into the now non-empty dir is refused with one line, no traceback
    rc, cap = _run(
        ["review-set", "render", "--source", str(src), "--output", str(out), "--pair", "0:0",
         "--out-dir", str(dest)],
        capsys,
    )  # fmt: skip
    err = cap.err.splitlines()
    assert rc == 1 and err[-1].startswith("Error:") and "Traceback" not in cap.err


def test_render_passes_arguments(tmp_path, capsys, monkeypatch):
    from gomoveshift.pipeline import review_set_render as rr

    seen = {}

    def fake(source, output, pairs, out_dir, **kw):
        seen.update(source=source, output=output, pairs=pairs, out_dir=out_dir, **kw)
        return {"pairs": pairs}

    monkeypatch.setattr(rr, "render_review_set", fake)
    cfg = SimpleNamespace()
    args = _build_parser().parse_args(
        ["review-set", "render", "--source", "a.mp4", "--output", "b.mp4", "--pair", "1:2",
         "--out-dir", "d", "--excerpt-seconds", "2.5"]
    )  # fmt: skip
    assert cmd_review_set(args, cfg) == 0
    assert seen["pairs"] == [(1.0, 2.0)] and seen["excerpt_seconds"] == 2.5
    assert seen["synthetic"] is False and seen["config"] is cfg


@pytest.mark.parametrize(
    "argv",
    [
        ["review-set", "scan", "--scores", "{t}/missing.npz"],
        ["review-set", "stops", "--stop-mask", "{t}/missing.npz", "--out", "{t}/o.json"],
        ["review-set", "render", "--source", "{t}/a", "--output", "{t}/b", "--pair", "1:2",
         "--out-dir", "{t}/d"],
        ["review-set", "render", "--source", "{t}/a", "--output", "{t}/b", "--pair", "bad",
         "--out-dir", "{t}/d"],
        ["review-set", "render", "--source", "{t}/a", "--output", "{t}/b", "--out-dir", "{t}/d"],
    ],
)  # fmt: skip
def test_errors_exit_nonzero_one_line(argv, tmp_path, capsys):
    rc, cap = _run([a.format(t=tmp_path) for a in argv], capsys)
    assert rc == 1
    assert cap.err.startswith("Error:") and cap.err.count("\n") == 1 and "Traceback" not in cap.err


def test_stops_schema_error_is_one_line(tmp_path, capsys):
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, other=np.zeros(3))
    rc, cap = _run(
        ["review-set", "stops", "--stop-mask", str(bad), "--out", str(tmp_path / "o")], capsys
    )
    assert rc == 1 and cap.err.count("\n") == 1


def test_parse_pair():
    assert h.parse_pair("1.5:2") == (1.5, 2.0)
    with pytest.raises(ValueError):
        h.parse_pair("1:2:3")
