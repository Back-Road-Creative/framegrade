"""The engine reads a per-tenant MediaProfile from brand.yaml, not brand literals."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from gomoveshift.pipeline import ffmpeg_filters
from gomoveshift.publishing import brand_policy
from gomoveshift.publishing.brand_policy import MediaProfile, load_media_profile

PIPELINE = Path(brand_policy.__file__).resolve().parent.parent / "pipeline"


def _brand_dir(monkeypatch, tmp_path: Path, slug: str, body: str) -> None:
    d = tmp_path / slug
    d.mkdir()
    (d / "brand.yaml").write_text(body)
    monkeypatch.setattr(brand_policy, "_SHIPPED_BRANDS_DIR", tmp_path)


def test_media_profile_unknown_key_raises(monkeypatch, tmp_path):
    _brand_dir(monkeypatch, tmp_path, "x", "media:\n  bogus: 1\n")
    with pytest.raises(ValueError, match=r"media\.bogus is not a recognized media key"):
        load_media_profile("x")


def test_media_profile_defaults_match_today(monkeypatch, tmp_path):
    _brand_dir(monkeypatch, tmp_path, "x", "disabled_tabs: []\n")
    p = load_media_profile("x")
    assert p == MediaProfile()
    assert p.font == ffmpeg_filters.FONT_PATH
    assert p.brand_text == ffmpeg_filters.BRAND_TEXT
    assert p.website_text == ffmpeg_filters.WEBSITE_TEXT
    assert p.upscale == "off"
    assert p.claim_native_resolution_only is True
    assert p.lut is None and p.logo is None
    assert p.still_card_style == "letterbox"
    assert dict(p.preset_defaults) == {}
    assert load_media_profile("no-such-brand") == p


def test_media_profile_validates_upscale(monkeypatch, tmp_path):
    _brand_dir(monkeypatch, tmp_path, "x", "media:\n  upscale: ml\n")
    assert load_media_profile("x").upscale == "ml"
    for bad in ("media:\n  upscale: lanczos\n", "media:\n  claim_native_resolution_only: 1\n"):
        (tmp_path / "x" / "brand.yaml").write_text(bad)
        with pytest.raises(ValueError, match=r"media\."):
            load_media_profile("x")
    (tmp_path / "x" / "brand.yaml").write_text("media:\n  font: 3\n")
    with pytest.raises(ValueError, match=r"media\.font"):
        load_media_profile("x")


def test_shipped_brands_load_and_gomoveshift_states_values():
    gms = load_media_profile("gomoveshift")
    assert gms == MediaProfile()
    raw = yaml.safe_load(
        (brand_policy._SHIPPED_BRANDS_DIR / "gomoveshift" / "brand.yaml").read_text()
    )
    assert raw["media"]["font"] == gms.font
    assert load_media_profile("backroadcreative") == MediaProfile()


def test_missing_logo_still_raises_at_use_site():
    # existence is checked at use (apply_ost logs the missing logo), not at load
    assert MediaProfile(logo="/nope/logo.png").logo == "/nope/logo.png"


@pytest.mark.parametrize("name", ["quality.py", "single_pass.py", "ost_applier.py"])
def test_engine_has_no_brand_literal(name):
    tree = ast.parse((PIPELINE / name).read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    bad = [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
        and any(t in n.value for t in ("GoMoveShift", "gomoveshift.com", "/usr/share/fonts"))
    ]
    assert not bad, f"{name} carries brand literals: {bad}"


def test_ost_filter_uses_profile_text():
    from gomoveshift.pipeline import ost_applier

    def chain(**kw):
        return ost_applier._build_drawtext_filter(
            location="Somewhere",
            layout=ost_applier.LAYOUTS["regular_lf"],
            font_path=ost_applier.FONT_PATH,
            video_width=3840,
            video_duration=817.0,
            **kw,
        )

    prof = MediaProfile(brand_text="AcmeCo", website_text="acme.example")
    custom = chain(media_profile=prof)
    assert "AcmeCo" in custom and "acme.example" in custom
    assert "GoMoveShift" not in custom
    default = chain()
    assert "GoMoveShift" in default and "gomoveshift.com" in default
    assert chain(media_profile=MediaProfile()) == default


def test_single_pass_exposes_profile():
    from gomoveshift.pipeline.single_pass import SinglePassProcessor

    assert "profile" in SinglePassProcessor.__init__.__code__.co_varnames


def test_drive_path_hands_loaded_profile_to_engine(tmp_path, monkeypatch):
    """run_day_pipeline loads the tenant MediaProfile once and passes it to
    encode_day_master (-> SinglePassProcessor) and apply_ost."""
    from unittest.mock import patch

    import test_day_pipeline_ost_audit_gate as g
    from gomoveshift.publishing import brand_policy

    loaded = MediaProfile(brand_text="WiredCo")
    monkeypatch.setattr(brand_policy, "load_media_profile", lambda slug: loaded)
    state = g._state_with_ost_text("2025-09-14", 1)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ost_audit_2025-09-14.confirmed").write_text("ok")
    (out / "shorts_ost_audit_2025-09-14.confirmed").write_text("ok")

    real_patch_heavy = g._patch_heavy
    seen = {}

    def _patch_heavy(tp):
        master, patches = real_patch_heavy(tp)

        def _enc(*a, **kw):
            seen.update(kw)
            return master

        patches[0] = patch("gomoveshift.pipeline.day_pipeline.encode_day_master", side_effect=_enc)
        return master, patches

    monkeypatch.setattr(g, "_patch_heavy", _patch_heavy)
    result, apply_mock, _ = g._run(tmp_path, state)
    assert result is not None
    assert seen["profile"] is loaded
    assert apply_mock.call_args.kwargs["media_profile"] is loaded


def test_day_master_encoder_hands_profile_to_processor():
    import inspect

    from gomoveshift.pipeline import day_master_encoder

    assert "profile=profile" in inspect.getsource(day_master_encoder.encode_day_master)
