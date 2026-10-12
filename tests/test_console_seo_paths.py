"""Console container SEO paths -- the override that makes drives startable
in the console image, plus the compose mounts that keep its data live."""

from pathlib import Path

import yaml

from gomoveshift.config import SEOConfig

_SERVICE_ROOT = Path(__file__).parent.parent
_OVERRIDE_PATH = _SERVICE_ROOT / "docker" / "console-config-override.yaml"
_COMPOSE_PATH = _SERVICE_ROOT / "docker" / "console-compose.yaml"


def _load_override_seo():
    return yaml.safe_load(_OVERRIDE_PATH.read_text())["seo"]


def test_override_seo_paths_are_all_under_app():
    seo = _load_override_seo()
    for key in ("storage_path", "credentials_dir", "manager_path"):
        assert seo[key].startswith("/app/"), seo


def test_override_seo_paths_pass_validate_paths_exist(tmp_path):
    """Mirrors the in-image directory shape under tmp_path (manager_path is a
    real dir, COPYed at build time; storage_path's parent lives under it) --
    a green result here proves the override doesn't reintroduce the startup
    ConfigError that hits every console-launched drive without it."""
    seo = _load_override_seo()
    app_root = tmp_path / "app"
    manager_rel = Path(seo["manager_path"]).relative_to("/app")
    storage_rel = Path(seo["storage_path"]).relative_to("/app")
    creds_rel = Path(seo["credentials_dir"]).relative_to("/app")

    manager_dir = app_root / manager_rel
    manager_dir.mkdir(parents=True)
    (app_root / storage_rel).parent.mkdir(parents=True, exist_ok=True)

    config = SEOConfig(
        storage_path=str(app_root / storage_rel),
        credentials_dir=str(app_root / creds_rel),
        manager_path=str(manager_dir),
    )
    config.validate_paths_exist()


def _load_compose():
    return yaml.safe_load(_COMPOSE_PATH.read_text())


def test_compose_mounts_seo_storage_credentials_and_token_under_app():
    svc = _load_compose()["services"]["console-base"]
    volumes = [str(v) for v in svc["volumes"]]
    assert any(v.endswith(":/app/waypost/data/seo") for v in volumes), volumes
    assert any(v.endswith(":/app/waypost/credentials:ro") for v in volumes), volumes
    assert any(v.endswith(":/app/waypost/youtube_upload_token.json.enc:ro") for v in volumes), (
        volumes
    )


def test_compose_seo_mount_sources_are_required_variables_without_defaults():
    """The host side of each SEO mount is operator data; a guessed default would
    be auto-created as an empty directory, so each must be a required variable."""
    svc = _load_compose()["services"]["console-base"]
    sources = {
        "GMS_SEO_STORAGE_DIR": ":/app/waypost/data/seo",
        "GMS_SEO_CREDENTIALS_DIR": ":/app/waypost/credentials:ro",
        "GMS_SEO_YT_TOKEN_ENC": ":/app/waypost/youtube_upload_token.json.enc:ro",
    }
    for var, suffix in sources.items():
        (entry,) = [str(v) for v in svc["volumes"] if str(v).endswith(suffix)]
        assert entry.startswith("${" + var + ":?"), entry


def test_override_manager_path_matches_the_image_copy():
    """The Dockerfile COPYs the waypost checkout to the path the override names."""
    dockerfile = (_SERVICE_ROOT / "docker" / "console.Dockerfile").read_text()
    assert "COPY waypost " + _load_override_seo()["manager_path"] in dockerfile


def test_corrections_path_follows_the_data_home_set_after_import(tmp_path, monkeypatch):
    """The console writer and the pipeline reader must resolve the same ledger file,
    including when the data home changes after the console module was imported."""
    from gomoveshift.console import seo as console_seo
    from gomoveshift.pipeline.validation.place_corrections import default_corrections_path

    monkeypatch.delenv("GMS_CORRECTIONS_PATH", raising=False)
    first, second = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("GMS_DATA_HOME", str(first))
    assert console_seo.corrections_path() == first / "place_corrections.yaml"
    monkeypatch.setenv("GMS_DATA_HOME", str(second))
    assert console_seo.corrections_path() == second / "place_corrections.yaml"
    assert console_seo.corrections_path() == default_corrections_path()
    override = tmp_path / "override.yaml"
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(override))
    assert console_seo.corrections_path() == override
