"""Brand policy ``shorts_per_drive``: planner-only per-drive Shorts cap."""

import pytest

import gomoveshift.publishing.brand_policy as brand_policy_mod
from gomoveshift.publishing import title_kit
from gomoveshift.publishing.brand_policy import load_brand_policy


def _brand(tmp_path, monkeypatch, yaml_text):
    slug_dir = tmp_path / "brands" / "b"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text(yaml_text)
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", tmp_path / "brands")
    monkeypatch.delenv("GMS_BRANDS_DIR", raising=False)
    monkeypatch.setattr("gomoveshift.console.settings.current_brand", lambda: "b")


def test_key_is_parsed_and_left_out_of_the_batch_cap(tmp_path, monkeypatch):
    _brand(tmp_path, monkeypatch, "policy:\n  shorts_per_drive: 1\n")
    policy = load_brand_policy("b")
    assert policy.shorts_per_drive == 1
    assert policy.per_location_cap is None


@pytest.mark.parametrize(
    "yaml_text, expected",
    [
        ("policy:\n  shorts_per_drive: 1\n", 1),
        ("policy:\n  per_location_cap: 3\n", 3),
        ("policy:\n  per_location_cap: 3\n  shorts_per_drive: 2\n", 2),
        ("policy:\n  per_location_cap: 1\n  shorts_per_drive: 2\n", 1),
        ("policy: {}\n", None),
    ],
)
def test_planner_cap_is_the_smaller_of_the_two(tmp_path, monkeypatch, yaml_text, expected):
    _brand(tmp_path, monkeypatch, yaml_text)
    assert title_kit._per_location_cap() == expected


def test_bad_value_raises_naming_the_key(tmp_path, monkeypatch):
    _brand(tmp_path, monkeypatch, "policy:\n  shorts_per_drive: 0\n")
    with pytest.raises(ValueError, match="policy.shorts_per_drive"):
        load_brand_policy("b")


def test_shipped_gomoveshift_brand_plans_one_short_per_drive(monkeypatch):
    monkeypatch.delenv("GMS_BRANDS_DIR", raising=False)
    policy = load_brand_policy("gomoveshift")
    assert policy.shorts_per_drive == 1
    assert policy.per_location_cap is None
