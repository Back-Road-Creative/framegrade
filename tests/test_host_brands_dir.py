"""``GMS_BRANDS_DIR``: a host brand-templates directory that overlays, per
slug, the templates shipped inside the package."""

from pathlib import Path

import pytest

from gomoveshift.console import settings as console_settings
from gomoveshift.publishing import brand_policy


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    shipped = tmp_path / "shipped"
    host = tmp_path / "host"
    shipped.mkdir()
    host.mkdir()
    monkeypatch.setattr(brand_policy, "_SHIPPED_BRANDS_DIR", shipped)
    monkeypatch.setattr(console_settings, "_SHIPPED_BRANDS_DIR", shipped)
    monkeypatch.setenv("GMS_BRANDS_DIR", str(host))
    return shipped, host


def _brand(root: Path, slug: str, **files: str) -> Path:
    d = root / slug
    d.mkdir()
    for name, body in files.items():
        (d / name).write_text(body)
    return d


def test_host_dir_wins_for_a_slug_in_both(dirs):
    shipped, host = dirs
    _brand(shipped, "examplebrand")
    _brand(host, "examplebrand")
    assert brand_policy.brand_dir("examplebrand") == host / "examplebrand"


def test_slug_missing_from_host_falls_back_to_shipped(dirs):
    shipped, host = dirs
    _brand(shipped, "examplebrand")
    assert brand_policy.brand_dir("examplebrand") == shipped / "examplebrand"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_or_unset_gives_shipped(dirs, monkeypatch, value):
    shipped, host = dirs
    _brand(host, "examplebrand")
    if value is None:
        monkeypatch.delenv("GMS_BRANDS_DIR")
    else:
        monkeypatch.setenv("GMS_BRANDS_DIR", value)
    assert brand_policy.brand_dir("examplebrand") == shipped / "examplebrand"


def test_overlay_is_whole_slug_not_per_file(dirs):
    shipped, host = dirs
    _brand(shipped, "examplebrand", **{"brand.yaml": "disabled_tabs: [drives]\n"})
    _brand(host, "examplebrand", **{"settings.yaml": "{}\n"})
    assert console_settings.brand_disabled_tabs("examplebrand") == set()


def test_known_brands_includes_host_only_slugs(dirs, monkeypatch, tmp_path):
    shipped, host = dirs
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "data"))
    _brand(shipped, "shippedbrand")
    _brand(host, "hostbrand")
    assert console_settings.known_brands() == ["hostbrand", "shippedbrand"]


def test_brand_disabled_tabs_and_kit_read_host_brand_yaml(dirs):
    _, host = dirs
    _brand(
        host,
        "examplebrand",
        **{"brand.yaml": "disabled_tabs:\n  - drives\nkit:\n  colors:\n    primary: '#112233'\n"},
    )
    assert console_settings.brand_disabled_tabs("examplebrand") == {"drives"}
    assert console_settings.brand_kit("examplebrand")["colors"]["primary"] == "#112233"


def test_load_brand_settings_seeds_from_host_template(dirs, tmp_path):
    shipped, host = dirs
    _brand(shipped, "examplebrand", **{"settings.yaml": "GMS_OUTPUT_DIR: /shipped\n"})
    _brand(host, "examplebrand", **{"settings.yaml": "GMS_OUTPUT_DIR: /host\n"})
    values = console_settings.load_brand_settings("examplebrand", data_dir=tmp_path / "d")
    assert values["GMS_OUTPUT_DIR"] == "/host"


def test_brand_policy_and_media_profile_read_host_dir(dirs):
    _, host = dirs
    _brand(
        host,
        "examplebrand",
        **{"brand.yaml": "policy:\n  per_location_cap: 3\nmedia:\n  brand_text: Example\n"},
    )
    assert brand_policy.load_brand_policy("examplebrand").per_location_cap == 3
    assert brand_policy.load_media_profile("examplebrand").brand_text == "Example"


def test_audio_profile_reads_host_brand_dir(dirs):
    from gomoveshift.pipeline.audio_profile import load_audio_profile

    _, host = dirs
    _brand(host, "examplebrand", **{"brand.yaml": "audio:\n  enabled: true\n  gain_db: -9\n"})
    profile = load_audio_profile(
        "examplebrand", brands_dir=brand_policy.brand_dir("examplebrand").parent
    )
    assert profile.enabled is True and profile.gain_db == -9.0
