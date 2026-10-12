"""Per-platform publishing routes are env-overridable via GMS_PUBLISHING_ROUTE_*
so the console Settings tab (one input per registry row) can flip a route per
brand at job launch. Env supplies the DEFAULT only -- an explicit config.yaml
(constructor kwarg) value still wins, same precedent as GMS_GEOCODING_CACHE_DIR.
"""

import pytest
from pydantic import ValidationError

from gomoveshift.config import DriveConfig

_ENV_TO_FIELD = {
    "GMS_PUBLISHING_ROUTE_YOUTUBE": "publishing_route_youtube",
    "GMS_PUBLISHING_ROUTE_INSTAGRAM": "publishing_route_instagram",
    "GMS_PUBLISHING_ROUTE_FACEBOOK": "publishing_route_facebook",
    "GMS_PUBLISHING_ROUTE_TIKTOK": "publishing_route_tiktok",
}


@pytest.mark.parametrize("env_name,field", list(_ENV_TO_FIELD.items()))
def test_env_first_party_overrides_default(monkeypatch, env_name, field):
    monkeypatch.setenv(env_name, "first_party")
    assert getattr(DriveConfig(), field) == "first_party"


@pytest.mark.parametrize("env_name,field", list(_ENV_TO_FIELD.items()))
def test_env_bogus_value_raises_at_construction(monkeypatch, env_name, field):
    monkeypatch.setenv(env_name, "carrier_pigeon")
    with pytest.raises(ValidationError):
        DriveConfig()


@pytest.mark.parametrize("env_name,field", list(_ENV_TO_FIELD.items()))
def test_env_unset_defaults_to_off(monkeypatch, env_name, field):
    monkeypatch.delenv(env_name, raising=False)
    assert getattr(DriveConfig(), field) == "off"


@pytest.mark.parametrize("env_name,field", list(_ENV_TO_FIELD.items()))
def test_explicit_kwarg_beats_env(monkeypatch, env_name, field):
    monkeypatch.setenv(env_name, "off")
    cfg = DriveConfig(**{field: "off"})
    assert getattr(cfg, field) == "off"
