"""``photos astro``: parser surface, dispatch ordering, and the resource gate.

No NEF fixtures ship here: the pipeline seams are monkeypatched directly (the
``test_photo_develop_cli.py`` idiom), except the early-return test, which relies
on the never-raise EXIF readers giving a genuine "no run found" for a missing path."""

import contextlib
import datetime
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy

import gomoveshift.pipeline.photos.astro_calibrate as astro_calibrate_mod
import gomoveshift.pipeline.photos.astro_orchestrator as astro_orch_mod
import gomoveshift.pipeline.photos.astro_sequence as astro_sequence_mod
import gomoveshift.pipeline.photos.develop as develop_mod
import gomoveshift.pipeline.photos.exif_gps as exif_gps_mod
import gomoveshift.pipeline.photos.preflight as astro_preflight_mod
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos.astro_calibrate import (
    AstroCalibrationError,
    AstroHighEfficiencyError,
)
from gomoveshift.pipeline.photos.astro_triage import FrameMetrics, StarShape


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    # The HE->DNG preflight probes the host for a converter install, which is
    # present on the operator's WSL box and absent in CI. Pin it so every test
    # below asserts its own subject on any machine; the gate's own tests repatch.
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "/opt/dngc")


def _parse(argv):
    return _build_parser().parse_args(argv.split())


def _run(argv):
    return photos_handler.cmd_photos(_parse(argv), None)


def _prepared_stub():
    # Real FrameMetrics, not a SimpleNamespace: prepare_frames' contract carries the
    # PSF shape on every frame, and a stub thinner than the dataclass lets a consumer
    # of a real field pass here while failing on a card. Healthy numbers, so the tests
    # below (none of which are about shape) see no shape line.
    return _shaped_prepared(1, **_HEALTHY)


def _shaped_prepared(n, *, major, minor, ellipticity, fwhm):
    """``prepare_frames``' result with REAL ``FrameMetrics``, so ``classify_shape``
    runs on them exactly as it does on a card."""
    shape = StarShape(
        fwhm_major_px=major,
        fwhm_minor_px=minor,
        ellipticity=ellipticity,
        orientation_deg=118.0,
        hfr_px=4.9,
        saturated=False,
    )
    metrics = tuple(
        FrameMetrics(
            path=Path(f"IMG_{i:04d}.NEF"),
            star_count=310,
            background_median=0.031,
            fwhm_px=fwhm,
            streaks=[],
            usable=True,
            shape=shape,
        )
        for i in range(n)
    )
    return SimpleNamespace(
        frames=tuple(range(n)),
        metrics=metrics,
        report=SimpleNamespace(cloud_frames=(), dew_frames=()),
        reference_index=0,
        failed=(),
    )


# Measured over 90 out-of-focus frames from one card: fwhm_px
# 10.81-11.49 at ellipticity 0.049-0.077 before the refocus pause, 6.82-8.66 after.
_DEFOCUSED = dict(major=11.34, minor=10.66, ellipticity=0.06, fwhm=11.0)
_HEALTHY = dict(major=7.60, minor=7.20, ellipticity=0.026, fwhm=7.4)


def _report_stub(out_path, below_floor_fraction=0.0):
    return SimpleNamespace(
        image_path=Path(out_path),
        linear_artifact_path=None,
        sidecar_path=None,
        stack=SimpleNamespace(
            n_frames=3,
            coverage=SimpleNamespace(
                min_count=3, median_count=3.0, below_floor_fraction=below_floor_fraction
            ),
        ),
        develop=None,
        rejected_frames=(),
        chroma_sigma_before=0.0,
        chroma_sigma_after=0.0,
        preview_path=Path(out_path).with_suffix(".preview.jpg"),
    )


def _mock_single_chunk(monkeypatch, path):
    """One light file -> one run -> one chunk, plus a stubbed megapixel peek."""
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [(path,)])
    monkeypatch.setattr(
        astro_sequence_mod, "chunk_sequence", lambda run, stack_len=None, max_span_s=None: [run]
    )
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 24.0)


def test_astro_parser_flags_and_defaults():
    args = _parse("photos astro a.NEF b.NEF --output-dir /tmp/out")
    assert args.paths == ["a.NEF", "b.NEF"]
    assert args.darks == () and args.output_dir == "/tmp/out"
    assert args.stack_len is None and args.horizon is None and args.triage_only is False
    assert args.overwrite is False
    assert args.keep_linear is False and args.linear_only is False
    gate_names = ("max_frame_gap", "max_chunk_span", "min_sequence_frames")
    assert all(getattr(args, n) is None for n in gate_names)

    full = _parse(
        "photos astro a.NEF --darks d1.NEF d2.NEF --output-dir /tmp/out "
        "--stack-len 10 --horizon 0.42 --triage-only --overwrite "
        "--max-frame-gap 1000 --max-chunk-span 20000 --min-sequence-frames 3 "
        "--keep-linear --linear-only"
    )
    assert full.darks == ["d1.NEF", "d2.NEF"] and full.stack_len == 10
    assert full.horizon == 0.42 and full.triage_only is True
    assert full.overwrite is True
    assert full.keep_linear is True and full.linear_only is True
    assert [getattr(full, n) for n in gate_names] == [1000.0, 20000.0, 3]
    with pytest.raises(SystemExit):
        _parse("photos astro a.NEF")  # --output-dir is required


def test_expand_raw_paths_expands_directory_sorted_and_passes_files_through(tmp_path):
    """A directory argument expands to the RAW files it contains, sorted by name;
    a non-RAW file inside is skipped; a plain file argument passes through as-is."""
    card = tmp_path / "card"
    card.mkdir()
    for name in ("IMG_0003.NEF", "IMG_0001.NEF", "IMG_0002.NEF"):
        (card / name).write_bytes(b"x")
    (card / "notes.txt").write_bytes(b"x")
    loose = tmp_path / "IMG_0000.NEF"
    loose.write_bytes(b"x")

    expanded = photos_handler._expand_raw_paths([str(loose), str(card)], "photos astro")

    assert expanded == [
        str(loose),
        str(card / "IMG_0001.NEF"),
        str(card / "IMG_0002.NEF"),
        str(card / "IMG_0003.NEF"),
    ]


def test_expand_raw_paths_empty_directory_raises_naming_it(tmp_path):
    empty_dir = tmp_path / "empty_card"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match=str(empty_dir)):
        photos_handler._expand_raw_paths([str(empty_dir)], "photos astro")


def test_astro_directory_argument_reaches_detect_sequences_expanded(tmp_path, monkeypatch):
    """The full CLI path: a directory handed to `photos astro` must reach
    detect_sequences already expanded to its (sorted) RAW files, not as one
    opaque path -- this is the "1 file(s)" under-processing bug."""
    card = tmp_path / "card"
    card.mkdir()
    for name in ("IMG_0002.NEF", "IMG_0001.NEF"):
        (card / name).write_bytes(b"x")

    seen = {}

    def _detect(ps, **kw):
        seen["paths"] = ps
        return []

    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", _detect)

    _run(f"photos astro {card} --output-dir {tmp_path / 'out'}")

    assert seen["paths"] == [str(card / "IMG_0001.NEF"), str(card / "IMG_0002.NEF")]


def test_astro_darks_directory_argument_is_also_expanded(tmp_path, monkeypatch):
    light = tmp_path / "IMG_0001.NEF"
    light.write_bytes(b"x")
    darks_dir = tmp_path / "darks"
    darks_dir.mkdir()
    for name in ("DARK_0002.NEF", "DARK_0001.NEF"):
        (darks_dir / name).write_bytes(b"x")

    _mock_single_chunk(monkeypatch, light)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    argv = f"photos astro {light} --darks {darks_dir} --output-dir {tmp_path / 'out'}"
    parsed = _parse(argv)
    photos_handler.cmd_photos(parsed, None)

    assert parsed.darks == [str(darks_dir / "DARK_0001.NEF"), str(darks_dir / "DARK_0002.NEF")]


def test_astro_early_returns_before_orchestrator_is_constructed(tmp_path, monkeypatch, capsys):
    """Must never construct ``PhotoOrchestrator`` (fails with the planted error if
    the dispatch moved below it); also proves "no run found" is a clean 1, since
    the never-raise EXIF readers give an empty run for a missing path -- and that
    the message names the diagnosis, not just "no run found" and a dead end."""
    from gomoveshift.pipeline.photos import orchestrator as orch_mod

    def _boom(*a, **k):
        raise AssertionError("PhotoOrchestrator must not be constructed for `photos astro`")

    monkeypatch.setattr(orch_mod, "PhotoOrchestrator", _boom)
    missing = tmp_path / "missing.NEF"
    assert _run(f"photos astro {missing} --output-dir {tmp_path / 'out'}") == 1
    err = capsys.readouterr().err
    assert "No intervalometer run found" in err
    assert "no frame has a readable capture time" in err


def _raise(exc):
    if exc:
        raise exc


def _stub_libraw_probe(monkeypatch, *, imread_exc=None, unpack_exc=None):
    """Stub the gate's LibRaw probe (open, then unpack); returns the probed paths."""
    probed = []

    @contextlib.contextmanager
    def _open():
        yield SimpleNamespace(unpack=lambda: _raise(unpack_exc))

    def _imread(p):
        probed.append(p)
        _raise(imread_exc)
        return _open()

    monkeypatch.setattr(rawpy, "imread", _imread)
    return probed


@pytest.mark.parametrize(
    "imread_exc",
    [None, rawpy.LibRawFileUnsupportedError("Unsupported file format or not RAW file")],
)
def test_astro_refuses_only_what_libraw_actually_cannot_unpack(
    tmp_path, monkeypatch, capsys, imread_exc
):
    """The gate must fire on undecodable-ness, never on a .nef suffix: a lossless card
    LibRaw reads fine still stacks on a machine with no converter, exactly as before, and
    a file that will not open at all is an open failure the decode path reports properly,
    not the preflight's to refuse on."""
    paths = [tmp_path / f"IMG_{i:04d}.NEF" for i in range(3)]
    for p in paths:
        p.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, paths[0])
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "")
    probed = _stub_libraw_probe(monkeypatch, imread_exc=imread_exc)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    joined = " ".join(str(p) for p in paths)
    assert _run(f"photos astro {joined} --output-dir {tmp_path / 'out'}") == 0
    assert "photos.he_dng_converter_exe" not in capsys.readouterr().err
    assert len(probed) == 1  # one candidate answers for the card, not 90 opens on a real one


def test_astro_refuses_an_undecodable_he_card_before_any_decode(tmp_path, monkeypatch, capsys):
    """Upstream gate: LibRaw cannot unpack the card and no converter resolves -- refuse
    once, up front, instead of one warning per frame and a master silently stacked from
    whichever frames survived. Fires before grouping, so the message is the diagnosis
    rather than a downstream "no run found"."""
    lights = [tmp_path / f"IMG_{i:04d}.NEF" for i in range(3)]
    dark = tmp_path / "DARK_0001.NEF"
    for p in [*lights, dark]:
        p.write_bytes(b"x")
    probe = tmp_path / "nowhere" / "Adobe DNG Converter.exe"
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "")
    monkeypatch.setattr(develop_mod, "HE_DNG_CONVERTER_DEFAULT_PATHS", (str(probe),))
    _stub_libraw_probe(monkeypatch, unpack_exc=rawpy.LibRawDataError("Data error"))
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )

    argv = (
        f"photos astro {' '.join(str(p) for p in lights)} --darks {dark} "
        f"--output-dir {tmp_path / 'out'}"
    )
    assert _run(argv) == 1
    err = capsys.readouterr().err
    assert "4 input(s)" in err  # darks hit the identical decode path, so they count too
    assert "High Efficiency" in err
    assert "photos.he_dng_converter_exe" in err
    assert str(probe) in err  # names the default path that was probed and missed


def test_astro_he_gate_applies_to_triage_only(tmp_path, monkeypatch, capsys):
    """--triage-only decodes every frame too, so it gets the identical refusal."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    monkeypatch.setattr(develop_mod, "resolve_he_dng_converter", lambda: "")
    _stub_libraw_probe(monkeypatch, unpack_exc=rawpy.LibRawDataError("Data error"))
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} --triage-only") == 1
    assert "photos.he_dng_converter_exe" in capsys.readouterr().err


def _stub_astro_run(tmp_path, monkeypatch, prepared):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: prepared)
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))
    return path


def test_triage_only_names_a_defocused_card_while_the_operator_can_still_refocus(
    tmp_path, monkeypatch, capsys
):
    """A card of 90 frames shot out of focus: the pipeline computed
    ``likely_defocus`` on the first frame and printed nothing, and the cause surfaced
    only after the whole night failed to register. ``--triage-only`` is the card check
    run before committing to a night, so the verdict has to land here."""
    path = _stub_astro_run(tmp_path, monkeypatch, _shaped_prepared(3, **_DEFOCUSED))
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} --triage-only") == 0
    out = capsys.readouterr().out
    assert "3/3 frame(s) read likely_defocus" in out
    assert "REFOCUS before shooting more" in out  # the action, not just the label
    assert "3/3 usable" in out  # the existing usable/cloud/dew line is not displaced


def test_triage_only_stays_quiet_on_a_healthy_round_and_sharp_card(tmp_path, monkeypatch, capsys):
    """``indeterminate`` is the normal case: a shape line on every good night is noise
    the real one would be lost in."""
    path = _stub_astro_run(tmp_path, monkeypatch, _shaped_prepared(3, **_HEALTHY))
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} --triage-only") == 0
    out = capsys.readouterr().out
    assert "3/3 usable" in out
    for word in ("likely_defocus", "likely_trailing", "corner_aberration", "indeterminate"):
        assert word not in out


def test_a_defocused_stack_run_reports_the_verdict_before_it_registers_and_stacks(
    tmp_path, monkeypatch, capsys
):
    """A full run gets the same warning at the same moment triage produces it -- and it
    stays a report: the run still stacks, since a deliberately defocused frame must not
    be refused (``classify_shape``'s thresholds are PROVISIONAL and gate nothing)."""
    path = _stub_astro_run(tmp_path, monkeypatch, _shaped_prepared(4, **_DEFOCUSED))
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0
    out = capsys.readouterr().out
    assert "4/4 frame(s) read likely_defocus" in out
    assert out.index("likely_defocus") < out.index("registering + stacking")
    assert "stacked 3 frame(s)" in out  # reported, never refused


def test_astro_names_the_resolved_converter_and_the_work_it_will_do(tmp_path, monkeypatch, capsys):
    """A converter DID resolve: name it and the count once, so the operator sees the
    conversion engaged rather than inferring it from a slow first frame."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    probed = _stub_libraw_probe(monkeypatch)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0
    out = capsys.readouterr().out
    assert "/opt/dngc" in out and "1 input(s)" in out
    assert probed == []  # a resolved converter handles whatever these are: no extra I/O

    # A fresh cached .dng leaves no candidate at all: nothing resolves, nothing prints.
    (tmp_path / "IMG_0001.dng").write_bytes(b"dngdata")  # written after: mtime >= source
    monkeypatch.setattr(
        develop_mod, "resolve_he_dng_converter", lambda: pytest.fail("must not resolve")
    )
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0
    assert "converter" not in capsys.readouterr().out


def test_preflight_runs_before_any_decode(tmp_path, monkeypatch):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    calls = []
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(
        astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: calls.append("preflight")
    )
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda *a, **k: (calls.append("prepare"), _prepared_stub())[1],
    )
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (calls.append("stack_run"), _report_stub(o))[1],
    )
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0
    # TWO gate calls precede the decode: the run-level one that sizes the whole night off
    # its largest chunk, then this chunk's own backstop. The ordering this test has always
    # pinned -- no decode before a gate -- is unchanged and now holds a call earlier.
    assert calls == ["preflight", "preflight", "prepare", "stack_run"]

    # A shortfall must block that decode outright, as a message, not a traceback.
    def refuse(*a, **k):
        raise RuntimeError("Insufficient free RAM: 2.00 GB available, need 25.15 GB")

    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", refuse)
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 1


def test_preflight_is_told_the_runs_actual_dark_count(tmp_path, monkeypatch):
    """The gate must be given the darks the run will actually load: building the master
    dark holds every dark as a uint16 mosaic, a float32 conversion of each, and np.stack's
    copy all at once, which the estimator charges per dark. Measured with the shipped
    estimator at this body's 20.8768 MP, 4 darks add ~0.78 GB at 10, 15 and 20 frames
    alike -- omitted, the gate under-counts exactly when darks are adding the memory.
    Meteor reinsertion is NOT passed: this path never hands ``stack_run`` a ``meteors=``
    policy, so reinsertion is structurally off and claiming its RAM would over-count."""
    path = tmp_path / "IMG_0001.NEF"
    darks = [tmp_path / f"DARK_{i:04d}.NEF" for i in range(4)]
    for p in [path, *darks]:
        p.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    seen: list[dict] = []
    monkeypatch.setattr(
        astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: seen.append(k)
    )
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    out_dir = tmp_path / "out"
    joined = " ".join(str(d) for d in darks)
    assert _run(f"photos astro {path} --darks {joined} --output-dir {out_dir} --overwrite") == 0
    assert seen[-1]["dark_count"] == 4
    assert seen[-1].get("meteor_reinsertion_enabled", False) is False

    # ...and the real count, never a constant: a run with no --darks charges no dark term.
    assert _run(f"photos astro {path} --output-dir {out_dir} --overwrite") == 0
    assert seen[-1]["dark_count"] == 0


class _DeliveredStdout(io.StringIO):
    """stdout stand-in whose ``delivered`` holds only text a flush has pushed out."""

    delivered = ""

    def flush(self):
        self.delivered = self.getvalue()


def test_stage_progress_lines_reach_the_operator_before_each_slow_phase(tmp_path, monkeypatch):
    """Decode+calibrate AND register+stack each run ~1 min/frame in silence. A line
    merely WRITTEN sits in Python's block buffer whenever stdout is not a tty (tee, cron,
    capture) and lands only at exit -- the hang itself. Assert DELIVERED, not written."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    out = _DeliveredStdout()
    monkeypatch.setattr("sys.stdout", out)
    seen = {}

    def _prepare(*a, **k):
        seen["dec"] = out.delivered
        return _prepared_stub()

    monkeypatch.setattr(astro_orch_mod, "prepare_frames", _prepare)
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0
    assert "decoding + calibrating..." in seen["dec"]
    # written after the decode flush, so it only lands here if it flushes on its own
    assert "registering + stacking" in out.delivered


def test_stacked_summary_line_reports_the_below_floor_coverage_fraction(
    tmp_path, monkeypatch, capsys
):
    """An operator watching the run must see how thin the stack got, not just
    how many frames survived -- ``report.stack.coverage.below_floor_fraction`` must
    reach stdout."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: _report_stub(o, below_floor_fraction=0.25),
    )

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0

    out = capsys.readouterr().out
    assert "25.0% below coverage floor" in out


def test_the_run_names_the_viewable_preview_not_only_the_tif(tmp_path, monkeypatch, capsys):
    """An operator who cannot find the JPEG will go on rendering one by hand, so the
    chunk line says where it landed."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0

    assert "preview -> " in capsys.readouterr().out


def test_multi_chunk_run_writes_one_master_per_chunk(tmp_path, monkeypatch):
    paths = tuple(tmp_path / f"IMG_{i:04d}.NEF" for i in range(40))
    for p in paths:
        p.write_bytes(b"x")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [paths])
    monkeypatch.setattr(
        astro_sequence_mod,
        "chunk_sequence",
        lambda run, stack_len=None, max_span_s=None: [tuple(run[:20]), tuple(run[20:])],
    )
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 24.0)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())

    written = []
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (written.append(Path(o)), _report_stub(o))[1],
    )

    argv = f"photos astro {' '.join(str(p) for p in paths)} --output-dir {tmp_path / 'out'}"
    assert _run(argv) == 0
    assert len(written) == 2
    assert len(set(written)) == 2, "each chunk must land at its own output path"


def test_linear_only_routes_artifact_targets_and_aggregate_preflight_before_decode(
    tmp_path, monkeypatch
):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    events = []
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(
        astro_preflight_mod,
        "preflight_astro_run_disk",
        lambda *a, **k: events.append(("disk", k)),
    )
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda *a, **k: (events.append(("decode", {})), _prepared_stub())[1],
    )
    stack_calls = []

    def _stack(prepared, out_path, **kwargs):
        stack_calls.append((out_path, kwargs))
        report = _report_stub(out_path or tmp_path / "unused.tif")
        report.image_path = None
        report.linear_artifact_path = kwargs["linear_artifact_path"]
        return report

    monkeypatch.setattr(astro_orch_mod, "stack_run", _stack)

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} --linear-only") == 0

    assert events[0][0] == "disk" and events[1][0] == "decode"
    assert events[0][1] == {
        "chunk_count": 1,
        "megapixels": 24.0,
        "keep_linear": True,
        "write_tiffs": False,
    }
    out_path, kwargs = stack_calls[0]
    assert out_path is None
    assert kwargs["linear_only"] is True
    assert kwargs["linear_artifact_path"].suffix == ".linear"


def test_existing_linear_artifact_refuses_before_decode_even_with_overwrite(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    out_dir = tmp_path / "out"
    target = out_dir / photos_handler._astro_output_name(
        (path,), (), None, "astro_stack", ".linear"
    )
    target.mkdir(parents=True)
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )

    assert _run(f"photos astro {path} --output-dir {out_dir} --keep-linear --overwrite") == 1
    assert str(target) in capsys.readouterr().err


def test_rerun_over_the_same_frames_refuses_to_overwrite_the_earlier_master(
    tmp_path, monkeypatch, capsys
):
    """The name is keyed on the chunk's own frame identity, not the invocation-local
    ordinal, so a re-run over the SAME inputs lands on the SAME target and refuses rather
    than replacing a stacked master -- before prepare_frames, so no decode is wasted."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    calls = []
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (calls.append(Path(o)), _report_stub(o))[1],
    )
    argv = f"photos astro {path} --output-dir {tmp_path / 'out'}"

    assert _run(argv) == 0
    assert len(calls) == 1
    target = calls[0]

    # stack_run is mocked above, so nothing was actually written to `target`;
    # write it now to stand in for the real master the first run produced.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already stacked")
    calls.clear()
    capsys.readouterr()  # discard the first run's output

    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )
    assert _run(argv) == 1
    assert calls == []
    err = capsys.readouterr().err
    assert str(target) in err and "--overwrite" in err


def test_overwrite_flag_lets_a_rerun_replace_the_existing_master(tmp_path, monkeypatch, capsys):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    calls = []
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (calls.append(Path(o)), _report_stub(o))[1],
    )
    argv = f"photos astro {path} --output-dir {tmp_path / 'out'}"
    assert _run(argv) == 0
    target = calls[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already stacked")
    calls.clear()

    assert _run(f"{argv} --overwrite") == 0
    assert calls == [target]


def test_a_later_chunk_collision_refuses_before_any_chunk_decodes(tmp_path, monkeypatch):
    """A collision on chunk 3 of 3 must refuse before ANY chunk decodes: every target is
    computable from `chunks` up front, so waiting until its turn burns earlier chunks."""
    paths = tuple(tmp_path / f"IMG_{i:04d}.NEF" for i in range(9))
    for p in paths:
        p.write_bytes(b"x")
    chunk_groups = [paths[0:3], paths[3:6], paths[6:9]]
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [paths])
    monkeypatch.setattr(
        astro_sequence_mod,
        "chunk_sequence",
        lambda run, stack_len=None, max_span_s=None: chunk_groups,
    )
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 24.0)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target3 = out_dir / photos_handler._astro_output_name(chunk_groups[2], (), None)
    target3.write_bytes(b"already stacked")

    events = []
    monkeypatch.setattr(
        astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: events.append("preflight")
    )
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda chunk, **k: (events.append(f"DECODE:{Path(chunk[0]).stem}"), _prepared_stub())[1],
    )
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (events.append(f"WRITE:{Path(o).name}"), _report_stub(o))[1],
    )

    argv = f"photos astro {' '.join(str(p) for p in paths)} --output-dir {out_dir}"
    assert _run(argv) == 1
    assert events == [], f"chunks 1-2 must not decode/stack before the chunk-3 refusal: {events}"


def test_same_stem_different_directory_produces_different_output_names(tmp_path):
    """The fingerprint includes the frames' resolved paths, not bare stems
    -- two cards with identical default filenames must not collide on one name."""
    card1 = tmp_path / "card1" / "DCIM" / "100ND850"
    card2 = tmp_path / "card2" / "DCIM" / "100ND850"
    card1.mkdir(parents=True)
    card2.mkdir(parents=True)
    names = ["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"]
    chunk1 = tuple(card1 / n for n in names)
    chunk2 = tuple(card2 / n for n in names)
    name1 = photos_handler._astro_output_name(chunk1, (), None)
    name2 = photos_handler._astro_output_name(chunk2, (), None)
    assert name1 != name2


def test_different_darks_set_produces_a_different_output_name(tmp_path):
    """--darks is a render-affecting knob too: a different dark set changes the name."""
    chunk = tuple(tmp_path / n for n in ["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"])
    name1 = photos_handler._astro_output_name(chunk, (tmp_path / "DARK_0001.NEF",), None)
    name2 = photos_handler._astro_output_name(chunk, (tmp_path / "DARK_0002.NEF",), None)
    assert name1 != name2


def test_darks_spelled_differently_still_produce_the_same_output_name(tmp_path, monkeypatch):
    """One darks file spelled absolute vs relative resolves to one identity, so a re-run
    spelling --darks differently still trips the overwrite guard."""
    chunk = tuple(tmp_path / n for n in ["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"])
    (tmp_path / "darks").mkdir()
    name_abs = photos_handler._astro_output_name(chunk, (tmp_path / "darks/DARK.NEF",), None)
    monkeypatch.chdir(tmp_path)
    name_rel = photos_handler._astro_output_name(chunk, (Path("darks/DARK.NEF"),), None)
    assert name_abs == name_rel


def test_two_chunks_of_one_invocation_never_share_a_target_even_with_overwrite(
    tmp_path, monkeypatch, capsys
):
    """--overwrite is consent to replace an OLD master from a past run, not to let this
    run's own second chunk clobber the master its first chunk just wrote."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [(path, path)])
    monkeypatch.setattr(
        astro_sequence_mod,
        "chunk_sequence",
        lambda run, stack_len=None, max_span_s=None: [(path,), (path,)],
    )
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 24.0)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )
    argv = f"photos astro {path} --output-dir {tmp_path / 'out'} --overwrite"
    assert _run(argv) == 1
    assert "--overwrite" in capsys.readouterr().err


def test_a_genuinely_different_parameter_set_produces_a_different_output_name(
    tmp_path, monkeypatch
):
    """A different --horizon changes the render, so it must change the target name."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(photos_handler, "_astro_frame_shape", lambda p: (200, 100))
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    calls = []
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (calls.append(Path(o)), _report_stub(o))[1],
    )
    out_dir = tmp_path / "out"
    assert _run(f"photos astro {path} --output-dir {out_dir} --horizon 0.3") == 0
    assert _run(f"photos astro {path} --output-dir {out_dir} --horizon 0.6") == 0
    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_astro_orchestrator_error_is_reported_cleanly(tmp_path, monkeypatch, capsys):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)

    def boom(*a, **k):
        raise astro_orch_mod.AstroOrchestratorError("only 1 usable of 1 decoded (3 required)")

    monkeypatch.setattr(astro_orch_mod, "prepare_frames", boom)
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 1
    assert "only 1 usable of 1 decoded" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error_cls, message",
    [
        (AstroCalibrationError, "build_master_dark: no dark frames supplied"),
        (AstroHighEfficiencyError, "he.nef: raw stream undecodable and no DNG converter"),
    ],
)
def test_calibration_failure_is_reported_cleanly_not_a_traceback(
    tmp_path, monkeypatch, capsys, error_cls, message
):
    """Both typed calibration errors (open failure, HE-no-mosaic) reach the
    operator as a clean exit 1 + stderr message, never an unhandled raise."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)

    def boom(*a, **k):
        raise error_cls(message)

    monkeypatch.setattr(astro_orch_mod, "prepare_frames", boom)
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 1
    assert message in capsys.readouterr().err


def test_horizon_flag_reaches_prepare_frames_as_a_real_sky_mask(tmp_path, monkeypatch):
    """--horizon must reach triage (prepare_frames), not just the final blend
    (stack_run already received it via horizon_frac before this fix)."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(photos_handler, "_astro_frame_shape", lambda p: (200, 100))
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)

    seen = {}

    def _prepare(chunk, *, dark_paths=(), horizon_mask=None, **kwargs):
        seen["mask"] = horizon_mask
        return _prepared_stub()

    monkeypatch.setattr(astro_orch_mod, "prepare_frames", _prepare)
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} --horizon 0.5") == 0
    mask = seen["mask"]
    assert mask is not None
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (200, 100)
    # Sky-side-up, not just "some array of the right shape": row 0 (top) is
    # usable sky, the last row (bottom) is not -- catches an inverted threshold
    # that would mask off the sky and measure stars in the ground instead.
    assert mask[0, 0] and not mask[-1, 0]


def _fake_rawpy_imread(monkeypatch, *, height, width, flip):
    """Stub ``rawpy.imread`` to yield a header-only ``raw.sizes`` (no unpack/postprocess),
    the same boundary ``_astro_frame_megapixels``/``_astro_frame_shape`` read."""
    import rawpy

    sizes = SimpleNamespace(height=height, width=width, flip=flip)

    class _FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    raw = _FakeRaw()
    raw.sizes = sizes
    monkeypatch.setattr(rawpy, "imread", lambda _path: raw)


def test_astro_frame_shape_accounts_for_the_libraw_rotation(monkeypatch):
    """LibRaw applies the camera's flip during ``postprocess`` (``user_flip`` left at
    its default -- see develop.py's ``_decode_raw``), so a quarter-turn flip (5/6)
    makes the decoded/luma array the *transpose* of ``raw.sizes`` (height, width),
    which is the pre-rotation visible-sensor size. A mask built from the untransposed
    shape doesn't line up with triage's luma array on a portrait sequence and crashes
    with an IndexError instead of stacking."""
    _fake_rawpy_imread(monkeypatch, height=4000, width=6000, flip=6)
    assert photos_handler._astro_frame_shape(Path("IMG_0001.NEF")) == (6000, 4000)

    _fake_rawpy_imread(monkeypatch, height=4000, width=6000, flip=5)
    assert photos_handler._astro_frame_shape(Path("IMG_0001.NEF")) == (6000, 4000)

    _fake_rawpy_imread(monkeypatch, height=4000, width=6000, flip=0)
    assert photos_handler._astro_frame_shape(Path("IMG_0001.NEF")) == (4000, 6000)

    # flip=3 is a 180 rotation (both axes reversed, not swapped) -- shape is unchanged.
    _fake_rawpy_imread(monkeypatch, height=4000, width=6000, flip=3)
    assert photos_handler._astro_frame_shape(Path("IMG_0001.NEF")) == (4000, 6000)


def test_no_horizon_flag_leaves_prepare_frames_mask_none(tmp_path, monkeypatch):
    """Without --horizon, behavior is unchanged: prepare_frames still sees
    horizon_mask=None and auto mode runs downstream."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, path)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)

    seen = {}

    def _prepare(chunk, *, dark_paths=(), horizon_mask=None, **kwargs):
        seen["mask"] = horizon_mask
        return _prepared_stub()

    monkeypatch.setattr(astro_orch_mod, "prepare_frames", _prepare)
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 0
    assert seen["mask"] is None


def test_invalid_horizon_exits_cleanly_not_a_traceback(tmp_path, capsys):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} --horizon 1.5") == 1
    err = capsys.readouterr().err
    assert "--horizon" in err and "1.5" in err


def test_megapixel_peek_on_a_non_raw_file_is_reported_cleanly_not_a_traceback(
    tmp_path, monkeypatch, capsys
):
    """A dual-format JPEG that sorts ahead of its RAW sibling (or otherwise
    slips past ``detect_sequences``) hits the pre-loop megapixel peek, which
    sits outside the per-chunk try/except -- it must not blow the CLI up
    with a raw rawpy traceback. Deliberately does NOT use ``_mock_single_chunk``
    (it stubs ``_astro_frame_megapixels``), so the real peek runs."""
    import rawpy

    path = tmp_path / "IMG_0001.JPG"
    path.write_bytes(b"x")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [(path,)])
    monkeypatch.setattr(
        astro_sequence_mod, "chunk_sequence", lambda run, stack_len=None, max_span_s=None: [run]
    )

    def _boom(p):
        raise rawpy.LibRawFileUnsupportedError("Unsupported file format or not RAW file")

    monkeypatch.setattr(rawpy, "imread", _boom)

    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'}") == 1
    err = capsys.readouterr().err
    assert "IMG_0001.JPG" in err
    assert "not a readable RAW frame" in err


def _sparse_cadence_frames(tmp_path, n=16, spacing_s=960):
    """N lights spaced ``spacing_s`` apart, same day/ISO/exposure -- a real
    intervalometer run except for a frame-gap ceiling wide enough to see it."""
    paths = tuple(tmp_path / f"IMG_{i:04d}.NEF" for i in range(n))
    for p in paths:
        p.write_bytes(b"x")
    base = datetime.datetime(2026, 8, 1, 19, 0, 0)
    timestamps = {p: base + datetime.timedelta(seconds=i * spacing_s) for i, p in enumerate(paths)}
    return paths, timestamps


def _patch_real_sequence_reader(monkeypatch, timestamps):
    """Real ``detect_sequences``/``chunk_sequence``, fed synthetic EXIF, so the
    sparse-cadence gates exercise the library's actual rule, not a stub."""
    monkeypatch.setattr(
        astro_sequence_mod, "extract_exif_capture_timestamp", lambda p: timestamps[Path(p)]
    )
    monkeypatch.setattr(
        astro_sequence_mod,
        "extract_exif_capture_metadata",
        lambda p: SimpleNamespace(iso=6400.0, exposure_s=15.0),
    )


def test_sparse_gates_thread_to_library_and_change_real_grouping(tmp_path, monkeypatch, capsys):
    """Defaults reproduce today's behavior exactly, zero chunks on a 960s-spaced
    card; the three flags reach detect_sequences/chunk_sequence with the parsed
    values, relay the sparse warning to stderr only, and admit a
    sparse night as one multi-frame chunk."""
    paths, timestamps = _sparse_cadence_frames(tmp_path)
    _patch_real_sequence_reader(monkeypatch, timestamps)
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 20.9)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    prepared_chunks = []
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda chunk, **k: (prepared_chunks.append(chunk), _prepared_stub())[1],
    )
    written = []
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (written.append(Path(o)), _report_stub(o))[1],
    )

    calls = {"detect": [], "chunk": []}  # list: the diagnosis probe also calls detect_sequences
    real_detect, real_chunk = astro_sequence_mod.detect_sequences, astro_sequence_mod.chunk_sequence
    monkeypatch.setattr(
        astro_sequence_mod,
        "detect_sequences",
        lambda ps, **kw: (calls["detect"].append(kw), real_detect(ps, **kw))[1],
    )
    monkeypatch.setattr(
        astro_sequence_mod,
        "chunk_sequence",
        lambda run, **kw: (calls["chunk"].append(kw), real_chunk(run, **kw))[1],
    )

    argv_paths = " ".join(str(p) for p in paths)
    assert _run(f"photos astro {argv_paths} --output-dir {tmp_path / 'out1'}") == 1
    assert calls["detect"][0] == {
        "max_frame_gap_s": astro_sequence_mod.MAX_FRAME_GAP_S,
        "min_sequence_frames": astro_sequence_mod.MIN_SEQUENCE_FRAMES,
    }
    assert calls["chunk"] == []
    assert prepared_chunks == []
    capsys.readouterr()  # discard: only the sparse run below is asserted

    calls["detect"].clear()
    calls["chunk"].clear()
    sparse_argv = (
        f"photos astro {argv_paths} --output-dir {tmp_path / 'out2'} "
        "--max-frame-gap 1000 --max-chunk-span 20000 --min-sequence-frames 3"
    )
    assert _run(sparse_argv) == 0
    assert calls["detect"] == [{"max_frame_gap_s": 1000.0, "min_sequence_frames": 3}]
    assert calls["chunk"] == [
        {"stack_len": astro_sequence_mod.STACK_LEN_DEFAULT, "max_span_s": 20000.0}
    ]
    assert len(written) == 1
    assert len(prepared_chunks) == 1 and len(prepared_chunks[0]) == len(paths)
    out, err = capsys.readouterr()
    assert "sparse-cadence sequence gates in use" in err
    assert "sparse-cadence sequence gates in use" not in out
    assert "sequence gates: --max-frame-gap 1000.0s" in out


def test_diagnose_names_both_gates_and_probe_never_warns(monkeypatch, caplog):
    """3 frames at 960s refuse BOTH default gates (plus the span they'd also
    need) in one message; the always-sparse relaxed probe never leaks a WARNING."""
    base = datetime.datetime(2026, 8, 1, 19, 0, 0)
    paths = [Path(f"IMG_{i:04d}.NEF") for i in range(3)]
    times = {p: base + datetime.timedelta(seconds=i * 960) for i, p in enumerate(paths)}
    monkeypatch.setattr(exif_gps_mod, "extract_exif_capture_timestamp", lambda p: times[p])
    seq_logger = logging.getLogger("gomoveshift.pipeline.photos.astro_sequence")

    def _relaxed(ps, *, max_frame_gap_s, min_sequence_frames):
        seq_logger.warning("sparse-cadence (probe stand-in)")
        return [tuple(ps)]

    with caplog.at_level("WARNING", logger="gomoveshift.pipeline.photos.astro_sequence"):
        reason = photos_handler._diagnose_no_sequence(paths, 60.0, 330.0, 5, _relaxed)
    assert caplog.records == []
    assert "--max-frame-gap" in reason and "960.0" in reason
    assert "--max-chunk-span 1920.0" in reason  # whole-run floor, not just the one gap
    assert "--min-sequence-frames 5 is" in reason and "only 3 frame(s)" in reason


def test_contradictory_gates_exit_1_with_actionable_message(tmp_path, capsys):
    """A chunk span below the frame gap is refused outright, both flags named."""
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    argv = (
        f"photos astro {path} --output-dir {tmp_path / 'out'} "
        "--max-frame-gap 1000 --max-chunk-span 330"
    )
    assert _run(argv) == 1
    err = capsys.readouterr().err
    assert "--max-chunk-span" in err and "330" in err
    assert "--max-frame-gap" in err and "1000" in err


@pytest.mark.parametrize(
    "flag, value",
    [("--max-frame-gap", "0"), ("--max-frame-gap", "-5"), ("--max-chunk-span", "0")],
)
def test_non_positive_gate_rejected_cleanly(tmp_path, capsys, flag, value):
    path = tmp_path / "IMG_0001.NEF"
    path.write_bytes(b"x")
    assert _run(f"photos astro {path} --output-dir {tmp_path / 'out'} {flag} {value}") == 1
    err = capsys.readouterr().err
    assert flag in err and "positive" in err


def test_low_min_sequence_frames_dead_end_names_the_chunker_floor(tmp_path, monkeypatch, capsys):
    """A 4-frame run admitted by --min-sequence-frames 3 still hits chunk_sequence's
    own fixed 5-frame floor -- the failure must name the length and the floor."""
    paths, timestamps = _sparse_cadence_frames(tmp_path, n=4)
    _patch_real_sequence_reader(monkeypatch, timestamps)
    argv = (
        f"photos astro {' '.join(str(p) for p in paths)} --output-dir {tmp_path / 'out'} "
        "--max-frame-gap 1000 --max-chunk-span 20000 --min-sequence-frames 3"
    )
    assert _run(argv) == 1
    err = capsys.readouterr().err
    assert "run length(s) 4" in err and "5-frame" in err and "chunk_sequence" in err


# --- the run-level RAM gate ------------------------------------------------------------
#
# Every number below is measured with the SHIPPED estimator at a typical 20.8768 MP
# body with no darks: 12.52 GB for a 5-frame chunk, 15.44 GB for 10, 16.02 GB for 11,
# 17.19 GB for 13, 18.35 GB for 15 and 21.27 GB for 20 -- i.e. ~9.60 GB that does not move
# with frame count plus ~0.583 GB per frame. Only the machine's free-RAM READING is faked;
# the estimator, the gate and the chunker are all real.


def _steady_run(tmp_path, monkeypatch, n):
    """An n-frame intervalometer run at 30 s cadence, driven through the REAL
    ``detect_sequences``/``chunk_sequence`` over synthetic EXIF, so chunk boundaries come
    from the library's own rule (trailing-chunk merge included) and not from a stub."""
    paths, timestamps = _sparse_cadence_frames(tmp_path, n=n, spacing_s=30)
    _patch_real_sequence_reader(monkeypatch, timestamps)
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 20.8768)
    return " ".join(str(p) for p in paths)


def _ram_capped(monkeypatch, available_gb):
    """Fake the free-RAM reading at the ONE binding the gate and the suggestion share."""

    def _check_ram(min_free_gb=2.0):
        if min_free_gb > available_gb:
            raise RuntimeError(
                f"Insufficient free RAM: {available_gb:.2f} GB available, "
                f"need {min_free_gb} GB. Close other processes and retry."
            )

    monkeypatch.setattr(astro_preflight_mod, "check_ram", _check_ram)
    monkeypatch.setattr(astro_preflight_mod, "check_disk_space", lambda *a, **k: None)


def _spy_stack(monkeypatch):
    """Records every chunk decoded and every master written."""
    decoded, written = [], []
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda chunk, **k: (decoded.append(chunk), _prepared_stub())[1],
    )
    monkeypatch.setattr(
        astro_orch_mod,
        "stack_run",
        lambda p, o, **k: (written.append(Path(o)), _report_stub(o))[1],
    )
    return decoded, written


def test_darks_that_cannot_calibrate_the_lights_are_refused_before_any_decode(
    tmp_path, monkeypatch, capsys
):
    """Darks whose ISO/exposure/body disagree with the lights refuse EVERY chunk, so the
    run can only ever produce nothing -- and it currently discovers that one chunk at a
    time, after decoding each chunk's frames. Measured on a real card: 249 lights
    at ISO 3200 handed darks shot at ISO 6400 decoded all 14 chunks over 1h56m, refused
    all 14, and wrote no file. Identity is metadata-only (``read_capture_identity`` reads
    sizes/other without unpacking the image), so the whole answer is available up front
    for the cost of one read per dark. Refuse there, like the RAM and collision gates."""
    light = tmp_path / "IMG_0001.NEF"
    dark = tmp_path / "DARK_0001.NEF"
    for p in (light, dark):
        p.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, light)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda *a, **k: pytest.fail("must not stack"))

    def _identity(path):
        iso = 3200.0 if Path(path) == light else 6400.0
        return astro_calibrate_mod.CaptureIdentity(
            path=Path(path),
            camera="NIKON Z50_2",
            raw_shape=(3728, 5600),
            cfa_pattern=(0, 1, 3, 2),
            iso=iso,
            exposure_s=15.0,
            temperature_c=None,
        )

    monkeypatch.setattr(astro_calibrate_mod, "read_capture_identity", _identity)

    assert _run(f"photos astro {light} --darks {dark} --output-dir {tmp_path / 'out'}") == 1
    err = capsys.readouterr().err
    assert "before any decode" in err
    # The operator must be told WHICH field disagrees -- "darks are wrong" sends them
    # hunting; "light=3200.0 dark=6400.0" tells them which card to grab.
    assert "iso" in err and "3200.0" in err and "6400.0" in err


def test_darks_mismatched_only_against_a_later_chunk_are_refused_before_any_decode(
    tmp_path, monkeypatch, capsys
):
    """A settings change mid-night (ISO bumped after the first chunk) must not pass the
    identity gate just because the FIRST chunk's frame matched the darks -- the WHOLE
    selected input set, not one representative frame, must clear the gate before any
    chunk decodes. Two chunks: chunk 1's light matches the dark; chunk 2's does not."""
    light1 = tmp_path / "IMG_0001.NEF"
    light2 = tmp_path / "IMG_0002.NEF"
    dark = tmp_path / "DARK_0001.NEF"
    for p in (light1, light2, dark):
        p.write_bytes(b"x")
    monkeypatch.setattr(astro_sequence_mod, "detect_sequences", lambda ps, **kw: [(light1, light2)])
    monkeypatch.setattr(
        astro_sequence_mod,
        "chunk_sequence",
        lambda run, stack_len=None, max_span_s=None: [(run[0],), (run[1],)],
    )
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 24.0)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(
        astro_orch_mod, "prepare_frames", lambda *a, **k: pytest.fail("must not decode")
    )
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda *a, **k: pytest.fail("must not stack"))

    def _identity(path):
        iso = 6400.0 if Path(path) == light2 else 3200.0
        return astro_calibrate_mod.CaptureIdentity(
            path=Path(path),
            camera="NIKON Z50_2",
            raw_shape=(3728, 5600),
            cfa_pattern=(0, 1, 3, 2),
            iso=iso,
            exposure_s=15.0,
            temperature_c=None,
        )

    monkeypatch.setattr(astro_calibrate_mod, "read_capture_identity", _identity)

    assert (
        _run(f"photos astro {light1} {light2} --darks {dark} --output-dir {tmp_path / 'out'}") == 1
    )
    err = capsys.readouterr().err
    assert "before any decode" in err
    assert str(light2) in err
    assert "3200.0" in err and "6400.0" in err


def test_matching_darks_pass_the_identity_gate_and_the_run_proceeds(tmp_path, monkeypatch):
    """The gate must not cost a good run its darks: same body/ISO/exposure sails through,
    and a field either side could not read is skipped rather than refused (the identity
    reader returns None for an unwritten tag)."""
    light = tmp_path / "IMG_0001.NEF"
    darks = [tmp_path / f"DARK_{i:04d}.NEF" for i in range(3)]
    for p in (light, *darks):
        p.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, light)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    monkeypatch.setattr(
        astro_calibrate_mod,
        "read_capture_identity",
        lambda path: astro_calibrate_mod.CaptureIdentity(
            path=Path(path),
            camera="NIKON Z50_2",
            raw_shape=(3728, 5600),
            cfa_pattern=(0, 1, 3, 2),
            iso=3200.0,
            exposure_s=15.0,
            temperature_c=None,  # unwritten on both sides -- skipped, never a refusal
        ),
    )

    joined = " ".join(str(d) for d in darks)
    assert _run(f"photos astro {light} --darks {joined} --output-dir {tmp_path / 'out'}") == 0


def test_an_unreadable_dark_does_not_block_the_run_at_the_identity_gate(tmp_path, monkeypatch):
    """A dark this preflight cannot open is the DECODE path's failure to report, not a
    preflight's -- the same posture ``_astro_frame_megapixels``' own gate takes. Refusing
    here would turn one unreadable calibration file into a refusal of the whole night."""
    light = tmp_path / "IMG_0001.NEF"
    dark = tmp_path / "DARK_0001.NEF"
    for p in (light, dark):
        p.write_bytes(b"x")
    _mock_single_chunk(monkeypatch, light)
    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(astro_orch_mod, "prepare_frames", lambda *a, **k: _prepared_stub())
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))

    def _unreadable(path):
        raise AstroCalibrationError(f"{path}: could not open RAW file")

    monkeypatch.setattr(astro_calibrate_mod, "read_capture_identity", _unreadable)

    assert _run(f"photos astro {light} --darks {dark} --output-dir {tmp_path / 'out'}") == 0


def test_astro_first_artifact_deadline_s_scales_with_chunk_size_and_workers():
    """A supervisor watching a long run needs a first-artifact deadline it can compare
    against elapsed time -- it must scale with the first chunk's frame count and shrink
    as more decode workers are thrown at it."""
    ten_frames = [tuple(range(10))]
    one_worker = photos_handler._astro_first_artifact_deadline_s(ten_frames, decode_workers=1)
    two_workers = photos_handler._astro_first_artifact_deadline_s(ten_frames, decode_workers=2)
    assert one_worker > 0
    assert one_worker == pytest.approx(2 * two_workers)

    five_frames = [tuple(range(5))]
    assert photos_handler._astro_first_artifact_deadline_s(
        five_frames, decode_workers=1
    ) == pytest.approx(one_worker / 2)

    assert photos_handler._astro_first_artifact_deadline_s([], decode_workers=1) == 0.0


def test_first_artifact_deadline_reported_before_any_decode(tmp_path, monkeypatch, capsys, caplog):
    """The deadline estimate must be visible to a supervisor (log) and to a caller
    inspecting the run (return value from the helper), emitted before the run starts
    decoding/stacking -- not only a final duration reported after the fact."""
    paths, timestamps = _sparse_cadence_frames(tmp_path, n=10, spacing_s=30)
    _patch_real_sequence_reader(monkeypatch, timestamps)
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 20.8768)
    monkeypatch.setattr(astro_preflight_mod, "check_ram", lambda min_free_gb=2.0: None)
    monkeypatch.setattr(astro_preflight_mod, "check_disk_space", lambda *a, **k: None)
    decoded, written = _spy_stack(monkeypatch)

    argv = " ".join(str(p) for p in paths)
    with caplog.at_level(logging.INFO, logger="gomoveshift.cli.handlers.photos"):
        assert _run(f"photos astro {argv} --decode-workers 2 --output-dir {tmp_path / 'out'}") == 0

    out = capsys.readouterr().out
    assert "first artifact expected within" in out
    assert "first artifact expected within" in caplog.text
    assert len(decoded) == 1  # a 10-frame, 30s-cadence run stays a single chunk


def test_a_run_too_big_for_ram_is_refused_before_any_frame_decodes(tmp_path, monkeypatch, capsys):
    """The whole run is sized ONCE, up front. A 30-frame night at --stack-len 20 chunks
    into 20 + 10 frames; at 16.00 GB free the 20-frame chunk needs 21.27 GB and does not
    fit, so nothing may decode -- previously chunk 1 stacked (~1 min/frame) and only THEN
    did chunk 2 refuse and kill the run. The refusal must also answer the operator's next
    question, "then what should I have run?", with a --stack-len that genuinely fits."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=30)
    _ram_capped(monkeypatch, 16.00)
    decoded, written = _spy_stack(monkeypatch)

    # --decode-workers 1 pins the arithmetic quoted above to SERIAL decode, which is what
    # those GB figures were derived for. Concurrency's own effect on the gate is covered by
    # test_decode_workers_reaches_the_ram_gate below and by the estimator's unit tests.
    base = f"photos astro {argv_paths} --decode-workers 1 --max-chunk-span 20000 --output-dir "
    assert _run(base + str(tmp_path / "out1")) == 1
    err = capsys.readouterr().err
    assert decoded == [] and written == []  # not one frame decoded
    assert "before any decode" in err
    assert "20 frame(s)" in err and "21.2" in err
    assert "--stack-len 10" in err

    # ...and the suggested value is one that actually fits: 3 chunks of 10 all stack.
    assert _run(base + str(tmp_path / "out2") + " --stack-len 10") == 0
    assert [len(c) for c in decoded] == [10, 10, 10] and len(written) == 3


def test_the_suggested_stack_len_accounts_for_the_trailing_chunk_merge(
    tmp_path, monkeypatch, capsys
):
    """``chunk_sequence`` merges a trailing chunk shorter than its 5-frame floor into the
    predecessor, so on a 15-frame run EVERY --stack-len from 11 to 15 collapses to one
    15-frame chunk. At 16.70 GB free a 12-frame chunk (16.60 GB) is the largest that fits,
    so a suggestion taken from the estimator alone would say --stack-len 12 -- which still
    merges to 15 frames (18.35 GB) and still refuses. The suggestion must be re-derived by
    re-chunking, and land on 10."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=15)
    _ram_capped(monkeypatch, 16.70)
    decoded, written = _spy_stack(monkeypatch)

    # Serial, for the same reason as the test above: 16.60/18.35 GB are serial figures.
    base = f"photos astro {argv_paths} --decode-workers 1 --max-chunk-span 20000 --output-dir "
    assert _run(base + str(tmp_path / "out1")) == 1
    err = capsys.readouterr().err
    assert decoded == []
    assert "--stack-len 10" in err and "--stack-len 12" not in err

    # The naive answer really would have failed, and the suggested one really works.
    assert _run(base + str(tmp_path / "out2") + " --stack-len 12") == 1
    assert decoded == []
    assert _run(base + str(tmp_path / "out3") + " --stack-len 10") == 0
    assert [len(c) for c in decoded] == [10, 5] and len(written) == 2


def test_decode_workers_reaches_the_ram_gate_not_just_the_decode(tmp_path, monkeypatch, capsys):
    """The concurrency knob has to reach BOTH sides. A run sized for serial decode that then
    decodes on 4 threads holds three more frames than the gate allowed for, and meets the
    difference as an OOM past a preflight that passed -- the failure mode this whole term
    exists to prevent.

    Same 30-frame run and the same free RAM in both halves; only --decode-workers differs, so
    a gate that ignored the flag would give the same verdict twice.
    """
    argv_paths = _steady_run(tmp_path, monkeypatch, n=30)
    # 22.00 GB free sits BETWEEN the two estimates below -- the only cap at which the two
    # halves can disagree, so the test cannot pass by accident on either side.
    _ram_capped(monkeypatch, 22.00)
    decoded, written = _spy_stack(monkeypatch)

    base = f"photos astro {argv_paths} --stack-len 20 --max-chunk-span 20000 --output-dir "

    # 20 frames at the fixture's 20.8768 MP: 21.27 GB serial, fits under 22.00.
    assert _run(base + str(tmp_path / "serial") + " --decode-workers 1") == 0
    assert [len(c) for c in decoded] == [20, 10]

    decoded.clear()
    written.clear()
    # 22.67 GB at 4 workers -- three extra frames in flight -- and it no longer fits.
    assert _run(base + str(tmp_path / "parallel") + " --decode-workers 4") == 1
    assert decoded == [] and written == []  # refused BEFORE any decode, as designed
    assert "before any decode" in capsys.readouterr().err


def _ram_that_drops(monkeypatch, first_gb, then_gb):
    """``_ram_capped``, but the box gets busier after the run's own upfront gate has
    passed -- the real mid-run shape, where CI or another session takes the memory."""
    calls = []

    def _check_ram(min_free_gb=2.0):
        calls.append(1)
        available = first_gb if len(calls) == 1 else then_gb
        if min_free_gb > available:
            raise RuntimeError(
                f"Insufficient free RAM: {available:.2f} GB available, "
                f"need {min_free_gb} GB. Close other processes and retry."
            )

    monkeypatch.setattr(astro_preflight_mod, "check_ram", _check_ram)
    monkeypatch.setattr(astro_preflight_mod, "check_disk_space", lambda *a, **k: None)


def test_a_chunk_that_no_longer_fits_at_full_concurrency_decodes_serially_instead(
    tmp_path, monkeypatch, capsys
):
    """The mid-run backstop used to discard a chunk the box could still have stacked. On a
    real 25-chunk night free RAM fell to 14.7 GB while CI ran, every chunk
    needed 18.20 GB at the default 4 decode workers, and 24 of 25 were skipped -- but the
    same chunks needed only 16.80 GB serially and would have fitted the whole time.

    Concurrency is a SPEED choice; the frames are the product. So a chunk that no longer
    fits at the requested worker count drops to the LARGEST count that does fit -- giving
    up only as much speed as the shortfall actually costs -- rather than being thrown away,
    and only a chunk too big even for serial decode is skipped. The gate and the decode
    still agree: prepare_frames is handed the same reduced count the gate approved, which
    is the invariant ``test_decode_workers_reaches_the_ram_gate_not_just_the_decode`` pins.
    """
    argv_paths = _steady_run(tmp_path, monkeypatch, n=20)
    # 10 frames at the fixture's 20.8768 MP: 16.836 GB at 4 workers, 15.903 at 2, 15.436
    # at 1. The first reading clears all three; every later one clears 2 and 1 -- so
    # landing on 2 is what proves it stops at the shortfall instead of dropping to serial.
    _ram_that_drops(monkeypatch, first_gb=17.00, then_gb=16.00)
    decoded, written = _spy_stack(monkeypatch)
    seen_workers = []
    real_prepare = astro_orch_mod.prepare_frames
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda chunk, **k: (seen_workers.append(k.get("decode_workers")), real_prepare(chunk, **k))[
            1
        ],
    )

    code = _run(
        f"photos astro {argv_paths} --stack-len 10 --decode-workers 4 "
        f"--max-chunk-span 20000 --output-dir {tmp_path / 'out'}"
    )

    assert code == 0
    assert [len(c) for c in decoded] == [10, 10]  # nothing discarded
    assert len(written) == 2
    assert seen_workers == [2, 2]  # the decode honours what the gate approved
    assert "decode workers 4 -> 2" in capsys.readouterr().out


def test_a_chunk_too_big_even_for_serial_decode_is_still_skipped(tmp_path, monkeypatch, capsys):
    """Degrading is not a licence to run a chunk that cannot fit. Below the serial figure
    there is nothing left to give up, so the backstop still skips the chunk and keeps the
    run alive for the others -- the behaviour that existed before degrading did."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=20)
    # 15.436 GB serial is above 15.00, so no worker count saves this chunk.
    _ram_that_drops(monkeypatch, first_gb=17.00, then_gb=15.00)
    decoded, written = _spy_stack(monkeypatch)

    code = _run(
        f"photos astro {argv_paths} --stack-len 10 --decode-workers 4 "
        f"--max-chunk-span 20000 --output-dir {tmp_path / 'out'}"
    )

    assert code == 1
    assert decoded == [] and written == []
    assert "skipping this chunk" in capsys.readouterr().err


def test_decode_workers_below_one_is_refused_upfront(tmp_path, monkeypatch, capsys):
    """Rejected beside the other astro knobs rather than inside ThreadPoolExecutor, which
    raises on max_workers < 1 only once the first chunk is already being decoded."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=10)
    _spy_stack(monkeypatch)

    argv = f"photos astro {argv_paths} --decode-workers 0 --output-dir {tmp_path / 'out'}"
    assert _run(argv) == 1
    assert "--decode-workers must be >= 1" in capsys.readouterr().err


def _gate_failing_on_call(monkeypatch, *, nth):
    """Stub the gate so call ``nth`` refuses (call 1 is the up-front run-level gate)."""
    calls = []

    def _gate(*a, **k):
        calls.append(k.get("frame_count"))
        if len(calls) in nth:
            raise RuntimeError("Insufficient free RAM: 8.00 GB available, need 15.44 GB.")

    monkeypatch.setattr(astro_preflight_mod, "preflight_astro_stack", _gate)
    return calls


def test_a_mid_run_shortfall_skips_that_chunk_and_keeps_the_finished_masters(
    tmp_path, monkeypatch, capsys
):
    """RAM can genuinely fall mid-run on a shared box. When it does, the chunks already
    stacked must survive: the backstop skips its own chunk exactly as a failed chunk does,
    never kills the run. Chunk 2 of 3 refuses; 1 and 3 still ship."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=30)
    decoded, written = _spy_stack(monkeypatch)
    _gate_failing_on_call(monkeypatch, nth={3})  # 1 = up front, 2 = chunk 1, 3 = chunk 2

    argv = (
        f"photos astro {argv_paths} --max-chunk-span 20000 --stack-len 10 "
        f"--output-dir {tmp_path / 'out'}"
    )
    # Non-zero: the masters survive (below), but 2 of 3 chunks is not a clean night and
    # the exit status is all an unattended caller reads.
    assert _run(argv) == 1
    out, err = capsys.readouterr()
    assert len(decoded) == 2 and len(written) == 2  # chunk 2 skipped, not decoded
    assert "chunk 2/3" in err and "skipping" in err
    assert "chunk 1/3 (10 frames): stacked" in out and "chunk 3/3 (10 frames): stacked" in out


def test_an_incomplete_run_exits_non_zero_and_names_what_it_lost(tmp_path, monkeypatch, capsys):
    """A run that skipped chunks used to exit 0 as long as ONE master landed -- a real
    night lost 13 of 14 chunks to the RAM backstop and reported success, so a caller
    could not tell a complete night from a near-total failure. The masters that did
    stack still survive; only the verdict on the run changes."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=30)
    decoded, written = _spy_stack(monkeypatch)
    _gate_failing_on_call(monkeypatch, nth={3, 4})  # chunks 2 and 3 refuse, chunk 1 stacks

    argv = (
        f"photos astro {argv_paths} --max-chunk-span 20000 --stack-len 10 "
        f"--output-dir {tmp_path / 'out'}"
    )
    assert _run(argv) == 1
    out, err = capsys.readouterr()
    assert len(written) == 1  # chunk 1's master is kept, not discarded
    assert "chunk 1/3 (10 frames): stacked" in out
    assert "2 of 3 chunk(s) produced no stacked master" in err


def test_a_run_whose_every_chunk_is_skipped_still_exits_non_zero(tmp_path, monkeypatch, capsys):
    """Skipping instead of aborting must not turn a produced-nothing run green -- the
    written == 0 check is what carries the exit status now."""
    argv_paths = _steady_run(tmp_path, monkeypatch, n=30)
    decoded, written = _spy_stack(monkeypatch)
    _gate_failing_on_call(monkeypatch, nth={2, 3, 4})

    argv = (
        f"photos astro {argv_paths} --max-chunk-span 20000 --stack-len 10 "
        f"--output-dir {tmp_path / 'out'}"
    )
    assert _run(argv) == 1
    assert decoded == [] and written == []
    assert "No chunk produced a stacked master" in capsys.readouterr().err


def _spy_prepare_kwargs(monkeypatch):
    """Records the kwargs the handler hands prepare_frames."""
    seen = {}
    monkeypatch.setattr(
        astro_orch_mod,
        "prepare_frames",
        lambda chunk, **k: (seen.update(k), _prepared_stub())[1],
    )
    monkeypatch.setattr(astro_orch_mod, "stack_run", lambda p, o, **k: _report_stub(o))
    return seen


class TestMeteorDetectionFlag:
    """The streak scan is ~92% of triage and this command passes stack_run no meteor
    policy, so it must be OFF unless asked for -- and the sidecar must say "not searched"
    rather than "searched, found none"."""

    def test_the_scan_is_off_unless_the_flag_is_passed(self, tmp_path, monkeypatch):
        argv_paths = _steady_run(tmp_path, monkeypatch, n=10)
        _ram_capped(monkeypatch, 64.00)
        seen = _spy_prepare_kwargs(monkeypatch)
        rc = _run(f"photos astro {argv_paths} --output-dir {tmp_path} --stack-len 10")
        assert rc == 0
        assert seen["detect_meteors"] is False

    def test_the_flag_turns_the_scan_on(self, tmp_path, monkeypatch):
        argv_paths = _steady_run(tmp_path, monkeypatch, n=10)
        _ram_capped(monkeypatch, 64.00)
        seen = _spy_prepare_kwargs(monkeypatch)
        rc = _run(
            f"photos astro {argv_paths} --output-dir {tmp_path} --stack-len 10 --detect-meteors"
        )
        assert rc == 0
        assert seen["detect_meteors"] is True
