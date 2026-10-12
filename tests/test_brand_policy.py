"""Per-brand policy block in brand.yaml."""

import pytest

from gomoveshift.publishing.brand_policy import BrandPolicy, load_brand_policy


def _write_policy(tmp_path, monkeypatch, slug, yaml_text):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / slug
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text(yaml_text)
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)


def test_policy_block_sets_per_location_cap(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "capped", "policy:\n  per_location_cap: 5\n")
    policy = load_brand_policy("capped")
    assert policy.per_location_cap == 5


def test_brand_yaml_without_policy_block_falls_back_to_no_cap(tmp_path, monkeypatch):
    # A brand.yaml that only hides tabs has no `policy:` block.
    _write_policy(tmp_path, monkeypatch, "examplebrand", "disabled_tabs:\n  - drives\n")
    policy = load_brand_policy("examplebrand")
    assert policy.per_location_cap is None


def test_unknown_brand_falls_back_to_no_cap():
    policy = load_brand_policy("no-such-brand")
    assert policy.per_location_cap is None


def test_missing_policy_key_falls_back_to_none(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "nokey"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("disabled_tabs: [drives]\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    policy = load_brand_policy("nokey")
    assert policy.per_location_cap is None


def test_bad_per_location_cap_raises_naming_the_key(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "bad"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("policy:\n  per_location_cap: 0\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    with pytest.raises(ValueError, match="policy.per_location_cap"):
        load_brand_policy("bad")


def test_non_int_per_location_cap_raises(tmp_path, monkeypatch):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "bad2"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("policy:\n  per_location_cap: not-a-number\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    with pytest.raises(ValueError, match="policy.per_location_cap"):
        load_brand_policy("bad2")


def test_brand_policy_is_a_dataclass_with_defaults():
    policy = BrandPolicy()
    assert policy.per_location_cap is None


def test_brand_yaml_policy_block_does_not_disable_any_tab(shipped_brands, monkeypatch):
    """Adding `policy:` to a brand.yaml must not change console behaviour --
    `disabled_tabs` absent is still "every tab enabled"."""
    import gomoveshift.publishing.brand_policy as brand_policy_mod
    from gomoveshift.console import settings as console_settings

    (shipped_brands / "gomoveshift" / "brand.yaml").write_text("policy:\n  per_location_cap: 4\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", shipped_brands)

    assert load_brand_policy("gomoveshift").per_location_cap == 4
    assert console_settings.brand_disabled_tabs("gomoveshift") == set()


# --- weekly_ceiling, daily_ladder, stagger_hours, region_weight ----------


def test_full_policy_block_sets_all_extended_keys(tmp_path, monkeypatch):
    _write_policy(
        tmp_path,
        monkeypatch,
        "full",
        "policy:\n"
        "  per_location_cap: 5\n"
        "  weekly_ceiling: 10\n"
        "  daily_ladder: [1, 2, 3]\n"
        "  stagger_hours: 24\n"
        "  region_weight:\n"
        "    north: 3\n"
        "    south: 1\n",
    )
    policy = load_brand_policy("full")
    assert policy.per_location_cap == 5
    assert policy.weekly_ceiling == 10
    assert policy.daily_ladder == (1, 2, 3)
    assert policy.stagger_hours == 24
    assert dict(policy.region_weight) == {"north": 3, "south": 1}


def test_brand_yaml_without_policy_block_leaves_extended_keys_none(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "examplebrand", "disabled_tabs:\n  - drives\n")
    policy = load_brand_policy("examplebrand")
    assert policy.weekly_ceiling is None
    assert policy.daily_ladder is None
    assert policy.stagger_hours is None
    assert policy.region_weight is None


def test_brand_policy_dataclass_defaults_all_extended_keys_none():
    policy = BrandPolicy()
    assert policy.weekly_ceiling is None
    assert policy.daily_ladder is None
    assert policy.stagger_hours is None
    assert policy.region_weight is None


def test_weekly_ceiling_missing_falls_back_to_none(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "wc1", "policy:\n  per_location_cap: 3\n")
    assert load_brand_policy("wc1").weekly_ceiling is None


def test_weekly_ceiling_valid(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "wc2", "policy:\n  weekly_ceiling: 7\n")
    assert load_brand_policy("wc2").weekly_ceiling == 7


def test_weekly_ceiling_invalid_raises_naming_the_key(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "wc3", "policy:\n  weekly_ceiling: 0\n")
    with pytest.raises(ValueError, match="policy.weekly_ceiling"):
        load_brand_policy("wc3")


def test_weekly_ceiling_bool_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "wc4", "policy:\n  weekly_ceiling: true\n")
    with pytest.raises(ValueError, match="policy.weekly_ceiling"):
        load_brand_policy("wc4")


def test_daily_ladder_missing_falls_back_to_none(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "dl1", "policy:\n  per_location_cap: 3\n")
    assert load_brand_policy("dl1").daily_ladder is None


def test_daily_ladder_valid_becomes_tuple(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "dl2", "policy:\n  daily_ladder: [1, 2, 3]\n")
    assert load_brand_policy("dl2").daily_ladder == (1, 2, 3)


def test_daily_ladder_empty_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "dl3", "policy:\n  daily_ladder: []\n")
    with pytest.raises(ValueError, match="policy.daily_ladder"):
        load_brand_policy("dl3")


def test_daily_ladder_not_increasing_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "dl4", "policy:\n  daily_ladder: [2, 2, 3]\n")
    with pytest.raises(ValueError, match="policy.daily_ladder"):
        load_brand_policy("dl4")


def test_daily_ladder_non_int_entry_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "dl5", "policy:\n  daily_ladder: [1, two, 3]\n")
    with pytest.raises(ValueError, match="policy.daily_ladder"):
        load_brand_policy("dl5")


def test_stagger_hours_missing_falls_back_to_none(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "sh1", "policy:\n  per_location_cap: 3\n")
    assert load_brand_policy("sh1").stagger_hours is None


def test_stagger_hours_zero_is_valid(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "sh2", "policy:\n  stagger_hours: 0\n")
    assert load_brand_policy("sh2").stagger_hours == 0


def test_stagger_hours_negative_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "sh3", "policy:\n  stagger_hours: -1\n")
    with pytest.raises(ValueError, match="policy.stagger_hours"):
        load_brand_policy("sh3")


def test_stagger_hours_bool_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "sh4", "policy:\n  stagger_hours: false\n")
    with pytest.raises(ValueError, match="policy.stagger_hours"):
        load_brand_policy("sh4")


def test_region_weight_missing_falls_back_to_none(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "rw1", "policy:\n  per_location_cap: 3\n")
    assert load_brand_policy("rw1").region_weight is None


def test_region_weight_valid(tmp_path, monkeypatch):
    _write_policy(
        tmp_path, monkeypatch, "rw2", "policy:\n  region_weight:\n    ireland: 3\n    usa: 1.5\n"
    )
    weight = load_brand_policy("rw2").region_weight
    assert dict(weight) == {"ireland": 3, "usa": 1.5}


def test_region_weight_empty_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "rw3", "policy:\n  region_weight: {}\n")
    with pytest.raises(ValueError, match="policy.region_weight"):
        load_brand_policy("rw3")


def test_region_weight_uppercase_key_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "rw4", "policy:\n  region_weight:\n    Ireland: 3\n")
    with pytest.raises(ValueError, match="policy.region_weight"):
        load_brand_policy("rw4")


def test_region_weight_non_positive_value_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "rw5", "policy:\n  region_weight:\n    ireland: 0\n")
    with pytest.raises(ValueError, match="policy.region_weight"):
        load_brand_policy("rw5")


def test_region_weight_bool_value_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "rw6", "policy:\n  region_weight:\n    ireland: true\n")
    with pytest.raises(ValueError, match="policy.region_weight"):
        load_brand_policy("rw6")


def test_unknown_policy_key_raises_naming_the_key(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "uk1", "policy:\n  bogus_key: 1\n")
    with pytest.raises(ValueError, match="policy.bogus_key"):
        load_brand_policy("uk1")


# --- blackout, quiet_after_alert_hours ------------------------------------


def test_blackout_missing_falls_back_to_empty_tuple(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "bo1", "policy:\n  per_location_cap: 3\n")
    assert load_brand_policy("bo1").blackout == ()


def test_blackout_valid_iso_dates_become_a_tuple_of_dates(tmp_path, monkeypatch):
    import datetime

    _write_policy(
        tmp_path, monkeypatch, "bo2", "policy:\n  blackout:\n    - 2026-12-25\n    - 2027-01-01\n"
    )
    assert load_brand_policy("bo2").blackout == (
        datetime.date(2026, 12, 25),
        datetime.date(2027, 1, 1),
    )


def test_blackout_not_a_list_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "bo3", "policy:\n  blackout: 2026-12-25\n")
    with pytest.raises(ValueError, match="policy.blackout"):
        load_brand_policy("bo3")


def test_blackout_bad_date_string_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "bo4", "policy:\n  blackout:\n    - not-a-date\n")
    with pytest.raises(ValueError, match="policy.blackout"):
        load_brand_policy("bo4")


def test_quiet_after_alert_hours_missing_falls_back_to_none(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "qa1", "policy:\n  per_location_cap: 3\n")
    assert load_brand_policy("qa1").quiet_after_alert_hours is None


def test_quiet_after_alert_hours_zero_is_valid(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "qa2", "policy:\n  quiet_after_alert_hours: 0\n")
    assert load_brand_policy("qa2").quiet_after_alert_hours == 0


def test_quiet_after_alert_hours_negative_raises(tmp_path, monkeypatch):
    _write_policy(tmp_path, monkeypatch, "qa3", "policy:\n  quiet_after_alert_hours: -1\n")
    with pytest.raises(ValueError, match="policy.quiet_after_alert_hours"):
        load_brand_policy("qa3")


def test_brand_policy_dataclass_defaults_blackout_and_quiet(tmp_path):
    policy = BrandPolicy()
    assert policy.blackout == ()
    assert policy.quiet_after_alert_hours is None
