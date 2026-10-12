"""Tests for DriveConfig default values — tuning."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gomoveshift.config import Config, DriveConfig


def _load_with_user_drive(tmp_path, monkeypatch, drive):
    """``Config.load()`` with ``drive`` as the user's ~/.config override."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    (home / ".config" / "gomoveshift").mkdir(parents=True)
    (home / ".config" / "gomoveshift" / "config.yaml").write_text(
        yaml.safe_dump({"drive": drive}), encoding="utf-8"
    )
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return Config.load()


class TestBestMomentsPercentileDefault:
    """best_moments_percentile defaults to 60.0."""

    def test_default_is_60(self):
        cfg = DriveConfig()
        assert cfg.best_moments_percentile == 60.0, (
            f"Expected default 60.0, got {cfg.best_moments_percentile}"
        )

    def test_explicit_override_still_works(self):
        cfg = DriveConfig(best_moments_percentile=75.0)
        assert cfg.best_moments_percentile == 75.0


class TestBlurEnabledDefault:
    """Privacy blur is ON by default — operators opt out per run with --skip-blur.

    Locks the activation flip so a future edit can't silently revert faces/plates
    to shipping unblurred. The blur gate only protects blur-ENABLED runs, so the
    default being on is what makes the protection apply.
    """

    def test_default_is_on(self):
        assert DriveConfig().blur_enabled is True

    def test_explicit_opt_out_still_works(self):
        assert DriveConfig(blur_enabled=False).blur_enabled is False


class TestPublishingRouteDefaults:
    """Per-platform publishing route flags default to ``off`` — a config that
    names no route publishes nowhere, so nothing (a fresh checkout, the test
    suite's blank HOME) enters Phase D's publish-plan sidecar by accident.
    Metricool is retired and is no longer a valid route value; the operator
    opts each platform in with an explicit ``first_party``."""

    def test_youtube_defaults_off(self):
        assert DriveConfig().publishing_route_youtube == "off"

    def test_instagram_defaults_off(self):
        assert DriveConfig().publishing_route_instagram == "off"

    def test_facebook_defaults_off(self):
        assert DriveConfig().publishing_route_facebook == "off"

    def test_tiktok_defaults_off(self):
        assert DriveConfig().publishing_route_tiktok == "off"

    def test_first_party_is_accepted(self):
        cfg = DriveConfig(publishing_route_youtube="first_party")
        assert cfg.publishing_route_youtube == "first_party"

    def test_off_is_accepted(self):
        cfg = DriveConfig(publishing_route_youtube="off")
        assert cfg.publishing_route_youtube == "off"

    @pytest.mark.parametrize(
        "field",
        [
            "publishing_route_youtube",
            "publishing_route_instagram",
            "publishing_route_facebook",
            "publishing_route_tiktok",
        ],
    )
    def test_unknown_route_raises_at_construction(self, field):
        with pytest.raises(ValidationError):
            DriveConfig(**{field: "carrier_pigeon"})

    @pytest.mark.parametrize(
        "field",
        [
            "publishing_route_youtube",
            "publishing_route_instagram",
            "publishing_route_facebook",
            "publishing_route_tiktok",
        ],
    )
    def test_retired_metricool_route_coerces_to_off_with_one_warning(self, field, caplog):
        # The operator's live ``~/.config/gomoveshift/config.yaml`` carried
        # ``publishing_route_tiktok: metricool`` after the retirement; a value
        # a run can no longer honor must never brick ``Config.load()`` (which
        # ``gomoveshift.pipeline`` calls at import). It publishes NOTHING on
        # that platform until the operator writes ``first_party`` explicitly.
        import logging

        with caplog.at_level(logging.WARNING):
            cfg = DriveConfig(publish_tiktok=True, **{field: "metricool"})
        assert getattr(cfg, field) == "off"
        hits = [rec.message for rec in caplog.records if "retired" in rec.message]
        assert len(hits) == 1 and field in hits[0]

    def test_unknown_route_value_is_refused_naming_the_valid_values(self):
        with pytest.raises(ValidationError, match="first_party.*off|off.*first_party"):
            DriveConfig(publishing_route_youtube="buffer")

    def test_user_config_with_retired_tiktok_route_still_loads(self, tmp_path, monkeypatch):
        # A user config that pins TikTok to the retired route with
        # publish_tiktok false must still load, not make the whole package
        # unimportable.
        cfg = _load_with_user_drive(
            tmp_path, monkeypatch, {"publishing_route_tiktok": "metricool", "publish_tiktok": False}
        )
        assert cfg.drive.publishing_route_tiktok == "off"
        assert cfg.drive.publish_tiktok is False

    def test_publish_tiktok_false_keeps_first_party_tiktok_route(self):
        # publish_tiktok gates only the shorts CSV's TikTok column. Forcing
        # the route off from it left every queued TikTok
        # delivery with no handler, so none was ever sent.
        cfg = DriveConfig(publish_tiktok=False, publishing_route_tiktok="first_party")
        assert cfg.publishing_route_tiktok == "first_party"

    def test_user_config_tiktok_inbox_mode_stays_routed(self, tmp_path, monkeypatch):
        # The live user config's TikTok inbox mode: the first-party route on,
        # TikTok kept off the shorts CSV.
        cfg = _load_with_user_drive(
            tmp_path,
            monkeypatch,
            {"publishing_route_tiktok": "first_party", "publish_tiktok": False},
        )
        assert cfg.drive.publishing_route_tiktok == "first_party"
        assert cfg.drive.publish_tiktok is False

    def test_publish_tiktok_true_leaves_tiktok_route_alone(self):
        cfg = DriveConfig(publish_tiktok=True, publishing_route_tiktok="off")
        assert cfg.publishing_route_tiktok == "off"

    def test_legacy_rollback_key_is_ignored_with_a_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            DriveConfig(rollback="metricool")
        assert sum("rollback" in rec.message for rec in caplog.records) == 1

    def test_tiktok_audit_approved_defaults_false(self):
        assert DriveConfig().publishing_tiktok_audit_approved is False

    def test_tiktok_audit_approved_explicit_true(self):
        assert (
            DriveConfig(publishing_tiktok_audit_approved=True).publishing_tiktok_audit_approved
            is True
        )

    def test_tiktok_privacy_level_defaults_to_self_only(self):
        # The audience of the FIRST audited TikTok run. SELF_ONLY keeps that
        # run a canary; the operator opts into a public audience explicitly.
        assert DriveConfig().publishing_tiktok_privacy_level == "SELF_ONLY"

    @pytest.mark.parametrize(
        "level",
        ["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "SELF_ONLY"],
    )
    def test_every_documented_privacy_level_is_accepted(self, level):
        cfg = DriveConfig(publishing_tiktok_privacy_level=level)
        assert cfg.publishing_tiktok_privacy_level == level

    def test_privacy_level_tiktok_does_not_define_raises_at_construction(self):
        # "PUBLIC" is the plausible typo; TikTok's value is PUBLIC_TO_EVERYONE.
        with pytest.raises(ValidationError):
            DriveConfig(publishing_tiktok_privacy_level="PUBLIC")
