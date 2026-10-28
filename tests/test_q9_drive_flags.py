"""Q9d: the opt-in ``--short-reframe``, ``--output-cadence`` and ``--insert-still-motion`` flags.

The library options (tracked reframe, source cadence, still push-in) are covered in
``test_reframe.py``, ``test_source_cadence.py`` and ``test_insert_motion.py``. This file proves
the plumbing: parse -> config mirror -> registry -> orchestrator -> each call site, and that
with every flag absent each call site receives exactly what it receives today.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

FLAGS = {
    # dest: (choices, default, value, config field, config default)
    "short_reframe": (["center", "tracked"], "center", "tracked"),
    "output_cadence": (["fixed", "source"], "fixed", "source"),
    "insert_still_motion": (["none", "push_in"], "none", "push_in"),
}
PKG = Path(__file__).resolve().parent.parent / "gomoveshift"


def _flag(dest):
    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    sub = next(
        a for a in parser._actions if hasattr(a, "choices") and a.choices and "drive" in a.choices
    )
    return parser, next(a for a in sub.choices["drive"]._actions if a.dest == dest)


# -- parse -----------------------------------------------------------------------------


@pytest.mark.parametrize("dest", FLAGS)
def test_flag_parses_with_a_sentinel_default(dest):
    _, action = _flag(dest)
    assert action.choices == FLAGS[dest][0]
    assert action.default is None
    assert action.option_strings == ["--" + dest.replace("_", "-")]


def test_flags_parse_values_and_absent_is_none():
    parser, _ = _flag("short_reframe")
    args = parser.parse_args(
        ["drive", "/src", "--short-reframe", "tracked", "--output-cadence", "source",
         "--insert-still-motion", "push_in"]
    )  # fmt: skip
    assert (args.short_reframe, args.output_cadence, args.insert_still_motion) == (
        "tracked",
        "source",
        "push_in",
    )
    bare = parser.parse_args(["drive", "/src"])
    assert (bare.short_reframe, bare.output_cadence, bare.insert_still_motion) == (None,) * 3
    with pytest.raises(SystemExit):
        parser.parse_args(["drive", "/src", "--output-cadence", "24"])


# -- config ----------------------------------------------------------------------------


def test_config_defaults_keep_todays_behaviour_and_validate():
    from gomoveshift.config import DriveConfig

    d = DriveConfig()
    assert (d.short_reframe, d.output_cadence, d.insert_still_motion) == (
        "center",
        "fixed",
        "none",
    )
    for field, bad in (
        ("short_reframe", "follow"),
        ("output_cadence", "60"),
        ("insert_still_motion", "zoom"),
    ):
        with pytest.raises(ValueError, match=field):
            DriveConfig(**{field: bad})
    assert DriveConfig(output_cadence="source").output_cadence == "source"


# -- registry --------------------------------------------------------------------------


def test_every_flag_is_a_registered_sentinel_with_a_config_mirror():
    from gomoveshift.pipeline.flag_registry import FLAG_REGISTRY, Semantic

    rules = {r.state_field: r for r in FLAG_REGISTRY}
    for field in FLAGS:
        assert rules[field].cli_dest == field
        assert rules[field].semantic is Semantic.SENTINEL_CLI_WINS
        assert rules[field].config_mirror == field


def test_resume_without_the_flags_keeps_the_saved_choice():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import apply_cli_to_state, merge_on_resume

    first = Config()
    state = PipelineState()
    apply_cli_to_state(state, {k: v[2] for k, v in FLAGS.items()}, first)
    saved = state.to_dict()
    cfg = Config()
    resumed = PipelineState()
    apply_cli_to_state(resumed, {k: None for k in FLAGS}, cfg)
    merge_on_resume(resumed, saved, cfg)
    for field, (_c, _d, value) in FLAGS.items():
        assert getattr(resumed, field) == value
        assert getattr(cfg.drive, field) == value


def test_a_differing_flag_beats_the_saved_choice():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import apply_cli_to_state, merge_on_resume

    saved = PipelineState()
    apply_cli_to_state(saved, {k: v[2] for k, v in FLAGS.items()}, Config())
    cfg = Config()
    state = PipelineState()
    apply_cli_to_state(state, {"output_cadence": "fixed"}, cfg)
    merge_on_resume(state, saved.to_dict(), cfg)
    assert state.output_cadence == "fixed" and cfg.drive.output_cadence == "fixed"
    assert state.short_reframe == "tracked"  # untouched flags keep the saved value


# -- handler -> orchestrator.run -> _build_initial_state ------------------------------


def _call_keywords(path: Path, func_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == func_name:
                out |= {k.arg for k in node.keywords if k.arg}
    return out


def test_handler_threads_the_flags_to_the_orchestrator():
    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator as Orch

    handler = PKG / "cli" / "handlers" / "drive.py"
    assert set(FLAGS) <= _call_keywords(handler, "run")
    for dest in FLAGS:
        assert f'getattr(args, "{dest}", None)' in handler.read_text()
    assert set(FLAGS) <= set(inspect.signature(Orch.run).parameters)
    assert set(FLAGS) <= set(inspect.signature(Orch._build_initial_state).parameters)
    orch_src = PKG / "pipeline" / "drive_orchestrator.py"
    assert set(FLAGS) <= _call_keywords(orch_src, "_build_initial_state")


def test_build_initial_state_applies_the_flags(tmp_path):
    from test_flag_registry import _fresh_state, _make_orch

    orch = _make_orch()
    state = _fresh_state(orch, tmp_path, **{k: v[2] for k, v in FLAGS.items()})
    for field, (_c, _d, value) in FLAGS.items():
        assert state.get(field) == value
        assert getattr(orch.config.drive, field) == value
    plain = _make_orch()
    _fresh_state(plain, tmp_path)
    assert plain.config.drive.short_reframe == "center"
    assert plain.config.drive.output_cadence == "fixed"
    assert plain.config.drive.insert_still_motion == "none"


# -- call sites ------------------------------------------------------------------------


def _drive_config(**kw):
    from gomoveshift.config import Config

    cfg = Config()
    for k, v in kw.items():
        setattr(cfg.drive, k, v)
    return cfg


def test_reframe_kwargs_default_adds_nothing():
    from gomoveshift.pipeline.shorts_extractor import reframe_kwargs

    assert reframe_kwargs(_drive_config()) == {}
    assert reframe_kwargs(None) == {}
    assert reframe_kwargs(_drive_config(short_reframe="tracked")) == {"reframe": "tracked"}


def _run_batched(tmp_path, **kw):
    from gomoveshift.pipeline import shorts_extractor as se

    calls: list[dict] = []

    def fake_render(**kwargs):
        calls.append(kwargs)
        return True

    spec = {
        "index": 1,
        "start_time": 5.0,
        "duration": 4.0,
        "location": "Synthetic Place",
        "subscribe_text": "Subscribe",
        "output_path": str(tmp_path / "short_01.mp4"),
    }
    with (
        patch.object(se, "_probe_video_stream_duration", return_value=0.0),
        patch.object(se, "_probe_source_keyframes", return_value=None),
        patch.object(se, "extract_segment_streamcopy", return_value=True),
        patch.object(se, "render_short_ffmpeg", side_effect=fake_render),
    ):
        res = se.render_shorts_batched(
            source_video=tmp_path / "lf.mp4",
            shorts_specs=[spec],
            output_dir=tmp_path,
            src_width=3840,
            src_height=2160,
            max_workers=1,
            **kw,
        )
    assert res == {1: True}
    return calls


def test_batched_shorts_default_call_is_unchanged(tmp_path):
    (call,) = _run_batched(tmp_path)
    assert "reframe" not in call
    assert set(call) == {
        "source_video", "output_path", "start_time", "duration", "location",
        "subscribe_text", "src_width", "src_height", "use_nvenc", "skip_audio",
    }  # fmt: skip


def test_batched_shorts_receive_the_reframe(tmp_path):
    (call,) = _run_batched(tmp_path, reframe="tracked")
    assert call["reframe"] == "tracked"


def _call_has_reframe_kwargs(path: Path, callee: str) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == callee:
                if any(
                    k.arg is None
                    and isinstance(k.value, ast.Call)
                    and getattr(k.value.func, "id", "") == "reframe_kwargs"
                    for k in node.keywords
                ):
                    return True
    return False


@pytest.mark.parametrize(
    "path, callee",
    [
        ("pipeline/shorts_extractor.py", "render_shorts_batched"),
        ("pipeline/shorts_extractor.py", "render_short_ffmpeg"),
        ("pipeline/day_pipeline.py", "_render_day_shorts"),
    ],
)
def test_short_call_sites_splat_the_configured_reframe(path, callee):
    assert _call_has_reframe_kwargs(PKG / path, callee)


def test_day_short_render_forwards_the_reframe():
    from gomoveshift.pipeline import day_pipeline

    params = inspect.signature(day_pipeline._render_day_shorts).parameters
    assert params["reframe"].default == "center"
    src = inspect.getsource(day_pipeline._render_day_shorts)
    assert "reframe_kwargs" not in src and "**({} if reframe" in src


def test_cadence_kwargs_and_resolver():
    from gomoveshift.pipeline.color_filters import (
        OUTPUT_FPS,
        cadence_kwargs,
        resolve_output_fps,
    )

    assert cadence_kwargs(_drive_config()) == {}
    assert cadence_kwargs(None) == {}
    assert cadence_kwargs(_drive_config(output_cadence="source")) == {"output_fps": "source"}
    assert resolve_output_fps(OUTPUT_FPS, 59.94) == OUTPUT_FPS  # default never consults the source
    assert resolve_output_fps("source", 59.94) == "60000/1001"
    assert resolve_output_fps("source", 30000 / 1001) == "30000/1001"
    with pytest.raises(ValueError, match="no supported output rate"):
        resolve_output_fps("source", 24.0)


def test_encoder_args_follow_the_resolved_source_rate():
    from gomoveshift.pipeline import ost_applier
    from gomoveshift.pipeline.color_filters import OUTPUT_FPS

    info = {"width": 1920, "height": 1080, "fps": 60000 / 1001, "bitrate_kbps": 20000}
    with patch("gomoveshift.pipeline.color.detect_nvenc", return_value=False):
        default = ost_applier._resolve_encoding_args(info, False, "x.mp4")
        explicit = ost_applier._resolve_encoding_args(info, False, "x.mp4", OUTPUT_FPS)
        source = ost_applier._resolve_encoding_args(info, False, "x.mp4", "source")
        direct = ost_applier._resolve_encoding_args(info, False, "x.mp4", "60000/1001")
        assert default == explicit
        assert source == direct != default
        with pytest.raises(ValueError, match="no supported output rate"):
            ost_applier._resolve_encoding_args({**info, "fps": 24.0}, False, "x.mp4", "source")


@pytest.mark.parametrize(
    "path, callee",
    [
        ("pipeline/stages/single_pass_encode.py", "process"),
        ("pipeline/day_master_encoder.py", "process"),
        ("pipeline/day_pipeline.py", "apply_ost"),
    ],
)
def test_encode_call_sites_splat_the_configured_cadence(path, callee):
    tree = ast.parse((PKG / path).read_text())
    found = [
        k
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == callee
        for k in n.keywords
        if k.arg is None
        and isinstance(k.value, ast.Call)
        and getattr(k.value.func, "id", "") == "cadence_kwargs"
    ]
    assert found, f"{path}: no {callee}(**cadence_kwargs(...)) call"


def test_process_resolves_source_to_the_probed_rate(tmp_path):
    """Source cadence reaches the filter and the encoder; 'source' never leaks past the resolve."""
    from gomoveshift.config import Config
    from gomoveshift.pipeline.single_pass import SinglePassProcessor

    proc = SinglePassProcessor(Config())
    info = {"duration": 5.0, "height": 1080, "width": 1920, "fps": 24.0, "bitrate_kbps": 9000}
    with pytest.raises(ValueError, match="no supported output rate"):
        proc.process(tmp_path / "in.mp4", tmp_path / "out.mp4", "X", video_info=info,
                     output_fps="source")  # fmt: skip


def _plan_calls(config, tmp_path):
    from gomoveshift.pipeline import inserts

    seen: list[dict] = []

    def fake_plan(*args, **kwargs):
        seen.append(kwargs)
        return []

    def fake_apply(lf, plans, out):
        out.write_bytes(b"x")
        return "result"

    lf = tmp_path / "lf.mp4"
    lf.write_bytes(b"x")
    inserts.splice_lf_inserts(
        lf,
        [],
        config=config,
        label=None,
        profile=None,
        work_dir=lf.parent / "w",
        plan=fake_plan,
        apply=fake_apply,
    )
    return seen


@pytest.mark.parametrize("motion, expect", [("none", None), ("push_in", "push_in")])
def test_splice_passes_still_motion_only_when_enabled(tmp_path, motion, expect):
    cfg = _drive_config(insert_still_motion=motion)
    (kw,) = _plan_calls(cfg, tmp_path)
    assert kw.get("still_motion") == expect
    if expect is None:
        assert "still_motion" not in kw
        assert set(kw) == {"profile", "rendition", "work_dir", "gate"}


# ── source cadence vs stream-copy joins ───────────────────────────────────────


@pytest.mark.parametrize(
    "kw,named",
    [
        ({"insert_photos": "a.jpg", "trail": False}, "--insert-photos"),
        ({"insert_clips": ["c.mp4"], "trail": False}, "--insert-clip"),
        ({"trail": None}, "--no-trail"),
        ({"trail": True}, "--no-trail"),
    ],
)
def test_source_cadence_is_refused_up_front_for_conflicting_stages(kw, named):
    from gomoveshift.pipeline.drive_orchestrator import refuse_source_cadence_conflicts
    from gomoveshift.pipeline.exceptions import PipelineValidationError

    cfg = _drive_config(output_cadence="source", **kw)
    with pytest.raises(PipelineValidationError, match=named):
        refuse_source_cadence_conflicts(cfg)


def test_source_cadence_passes_without_conflicting_stages_and_fixed_never_refuses():
    from gomoveshift.pipeline.drive_orchestrator import refuse_source_cadence_conflicts

    refuse_source_cadence_conflicts(_drive_config(output_cadence="source", trail=False))
    refuse_source_cadence_conflicts(_drive_config(trail=None, insert_photos="a.jpg"))


def test_a_saved_source_cadence_is_refused_too():
    from types import SimpleNamespace

    from gomoveshift.pipeline.drive_orchestrator import refuse_source_cadence_conflicts
    from gomoveshift.pipeline.exceptions import PipelineValidationError

    with pytest.raises(PipelineValidationError):
        refuse_source_cadence_conflicts(
            _drive_config(trail=None), SimpleNamespace(output_cadence="source")
        )


def test_run_refuses_after_the_resume_merge_and_before_any_phase():
    src = inspect.getsource(
        __import__(
            "gomoveshift.pipeline.drive_orchestrator", fromlist=["x"]
        ).DrivePipelineOrchestrator.run
    )
    assert (
        src.index("reset_ost_opt_ins_after_resume(")
        < src.index("refuse_source_cadence_conflicts(")
        < src.index("invalidate_phase_d_if_schedule_region_stale(")
    )


def test_repair_encode_args_default_unchanged_and_follow_the_source_rate():
    from gomoveshift.pipeline import montage_builder as mb

    default = mb.lf_reencode_args()
    assert default[default.index("-r") + 1] == "30000/1001"
    assert mb.lf_reencode_args("30000/1001") == default
    fast = mb.lf_reencode_args("60000/1001")
    assert fast[fast.index("-r") + 1] == "60000/1001" and fast != default
    assert mb.reencode_args_for("unused.mp4") == default
    with patch.object(mb, "probe_source_fps", return_value=60000 / 1001):
        assert mb.reencode_args_for("m.mp4", "source") == fast


def test_extract_and_prepend_thread_the_cadence():
    tree = ast.parse(Path("gomoveshift/pipeline/day_pipeline.py").read_text())
    for callee in ("extract_and_concat_segments", "prepend_montage"):
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == callee
        ]
        assert calls, callee
        for c in calls:
            splats = [k.value for k in c.keywords if k.arg is None]
            assert any(
                isinstance(v, ast.Call) and getattr(v.func, "id", "") == "cadence_kwargs"
                for v in splats
            ) or any(getattr(v, "id", "") == "_extract_kw" for v in splats), callee


def test_short_fingerprint_default_is_byte_identical_and_reframe_changes_it():
    import hashlib

    from gomoveshift.pipeline.shorts_extractor import _short_ost_fingerprint as fp

    old = hashlib.sha256("Loc\x00Sub\x001.0000:2.0000".encode()).hexdigest()
    assert fp("Loc", "Sub", 1.0, 2.0) == old
    assert fp("Loc", "Sub", 1.0, 2.0, reframe="center") == old
    assert fp("Loc", "Sub", 1.0, 2.0, reframe="tracked") != old


def test_shorts_cache_prefilter_and_writers_pass_the_reframe():
    src = Path("gomoveshift/pipeline/shorts_extractor.py").read_text()
    assert src.count("**reframe_kwargs(self.config)") >= 5
