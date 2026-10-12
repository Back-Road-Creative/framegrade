"""Settings registry (every GMS_* from the README) + brand round-trip."""

import re
from pathlib import Path

from gomoveshift.console import settings as console_settings

_README = Path(__file__).parent.parent / "README.md"


def test_registry_has_every_readme_gms_name():
    registry_names = {s.name for s in console_settings.load_registry()}
    readme_names = set(re.findall(r"`(GMS_[A-Z_]+)`", _README.read_text()))
    assert readme_names, "sanity: the README should mention at least one GMS_* var"
    assert not (readme_names - registry_names)


def test_registry_has_publishing_route_rows():
    # Only the registry rows are asserted here, not a round trip through
    # brand_launch_env.
    registry_names = {s.name for s in console_settings.load_registry()}
    assert {
        "GMS_PUBLISHING_ROUTE_YOUTUBE",
        "GMS_PUBLISHING_ROUTE_INSTAGRAM",
        "GMS_PUBLISHING_ROUTE_FACEBOOK",
        "GMS_PUBLISHING_ROUTE_TIKTOK",
    } <= registry_names


def test_console_data_dir_defaults_under_staging(monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_DATA_DIR", raising=False)
    monkeypatch.setenv("GMS_STAGING_DIR", "/tmp/gms-staging-test")
    assert console_settings.console_data_dir() == Path("/tmp/gms-staging-test") / "_console"


def test_console_data_dir_honors_explicit_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console-data"))
    assert console_settings.console_data_dir() == tmp_path / "console-data"


def test_brand_settings_round_trips(shipped_brands, tmp_path):
    values = console_settings.load_brand_settings("gomoveshift", data_dir=tmp_path)
    assert values["GMS_STAGING_DIR"] is None  # seeded from the shipped template

    values["GMS_OUTPUT_DIR"] = "/custom/output"
    console_settings.save_brand_settings("gomoveshift", values, data_dir=tmp_path)

    reloaded = console_settings.load_brand_settings("gomoveshift", data_dir=tmp_path)
    assert reloaded["GMS_OUTPUT_DIR"] == "/custom/output"
    assert reloaded["GMS_STAGING_DIR"] is None


def test_known_brands_includes_a_data_dir_brand_not_shipped(shipped_brands, monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    (tmp_path / "brands" / "customer-x").mkdir(parents=True)

    brands = console_settings.known_brands()

    assert "customer-x" in brands
    assert "gomoveshift" in brands  # shipped template, still present
    assert brands == sorted(set(brands))  # sorted and deduped


def test_brand_launch_env_merges_registry_overrides_over_os_environ(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "gomoveshift")
    monkeypatch.setenv("GMS_OUTPUT_DIR", "/env/default")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "untouched")

    console_settings.save_brand_settings(
        "gomoveshift", {"GMS_OUTPUT_DIR": "/brand/override"}, data_dir=tmp_path
    )

    env = console_settings.brand_launch_env()

    assert env["GMS_OUTPUT_DIR"] == "/brand/override"
    assert env["SOME_UNRELATED_VAR"] == "untouched"


def test_brand_launch_env_accepts_explicit_slug(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "gomoveshift")  # current brand -- must be ignored
    console_settings.save_brand_settings(
        "examplebrand", {"GMS_OUTPUT_DIR": "/brand/override"}, data_dir=tmp_path
    )

    env = console_settings.brand_launch_env("examplebrand")

    assert env["GMS_OUTPUT_DIR"] == "/brand/override"
