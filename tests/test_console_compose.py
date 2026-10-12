"""The console docker lane must be able to publish and tick: ``sops`` must be
installed in the image, every credential env path the compose file passes must
actually be bind-mounted, and the ticker must be enabled with a healthcheck so
an operator sees a live container.

Every host path in the compose file is a REQUIRED variable (``${VAR:?message}``)
with no default, because Docker auto-creates a missing bind source as an empty
root-owned directory; a guessed default would stand an empty folder in for a
key or a config file. The tests below pin that contract, and that an operator
override moves only the HOST side of a mount, never where it lands in the
container.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from _compose_helpers import env_key as _env_key
from _compose_helpers import resolved_service
from _compose_helpers import volume_target as _target

DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker"

#: Host-side variables that have no default: compose refuses to render without them.
REQUIRED_HOST_VARS = (
    "GMS_STAGING_DIR",
    "GMS_OUTPUT_DIR",
    "GMS_DRIVE_CREDENTIALS_ENC",
    "GMS_DRIVE_TOKEN_ENC",
    "GMS_AGE_KEY_FILE",
    "GMS_SOPS_CONFIG",
    "GMS_SEO_STORAGE_DIR",
    "GMS_SEO_CREDENTIALS_DIR",
    "GMS_SEO_YT_TOKEN_ENC",
)

#: Container-side targets of the credential mounts (fixed literals) and the env
#: var the app reads inside the container to find each one.
SECRET_TARGETS = {
    "GMS_DRIVE_CREDENTIALS_ENC": "/secrets/drive_credentials.json.enc",
    "GMS_DRIVE_TOKEN_ENC": "/secrets/drive_token.json.enc",
    "GMS_AGE_KEY_FILE": "/secrets/age.key",
    "GMS_SOPS_CONFIG": "/secrets/.sops.yaml",
}

#: The container-side path the image's baked ``seo.*`` override points at, and
#: therefore where ``social_queue.sqlite3`` lives inside the container.
QUEUE_TARGET = "/app/waypost/data/seo"
SEO_CREDENTIALS_TARGET = "/app/waypost/credentials"
SEO_TOKEN_TARGET = "/app/waypost/youtube_upload_token.json.enc"  # noqa: S105 - a path, not a secret

#: The media library reads a site checkout and the night tab reads camera-card
#: roots (console/library.py, console/night.py); both are an optional overlay.
SITE_TARGET = "/site"
NIGHT_TARGET = "/night-source"
SITE_ENV = ("GMS_SITE_PATH", "GMS_NIGHT_SOURCE_ROOTS")

#: A second output root, an optional overlay: mounted read-write at the SAME
#: path inside the container, because queue rows carry absolute sidecar paths.
EXTRA_OUTPUT_VAR = "GMS_VID_OUTPUT_DIR"

_VOLUME = re.compile(r"^(?P<host>\$\{[^}]*\}|[^:]+):(?P<rest>.+)$")


def _dockerfile_text() -> str:
    return (DOCKER_DIR / "console.Dockerfile").read_text()


def _compose() -> dict:
    return yaml.safe_load((DOCKER_DIR / "console-compose.yaml").read_text())


def _site_overlay() -> dict:
    return yaml.safe_load((DOCKER_DIR / "console-site.yaml").read_text())


def _extra_output_overlay() -> dict:
    return yaml.safe_load((DOCKER_DIR / "console-extra-output.yaml").read_text())


def _service():
    return resolved_service("gomoveshift-console", _compose())


def _env(service) -> dict:
    return dict(item.split("=", 1) for item in service["environment"])


def _host_for_target(volumes: list[str], target: str) -> str:
    """The host side of the bind mount landing on ``target``."""
    for entry in volumes:
        match = _VOLUME.match(entry)
        if match and _target(entry) == target:
            return match.group("host")
    raise AssertionError(f"no mount onto {target} among: {volumes!r}")


def test_dockerfile_installs_sops():
    assert "sops" in _dockerfile_text().lower()


def test_credential_env_paths_are_mounted():
    service = _service()
    env = _env(service)
    targets = {_target(v) for v in service["volumes"]}
    for key, target in SECRET_TARGETS.items():
        assert env.get(key) == target, f"{key} must be the fixed container path {target}"
        assert target in targets, f"{key}: {target} is not bind-mounted"


def test_every_host_bind_source_is_a_required_variable_with_no_default():
    """A bind source with a default would be created as an empty directory when
    the default does not exist. Every host side is ``${VAR:?message}``."""
    host_sides = {}
    for entry in _service()["volumes"]:
        if entry.startswith("console-data:"):
            continue  # the named volume, not a host path
        match = _VOLUME.match(entry)
        assert match, entry
        host = match.group("host")
        assert re.fullmatch(r"\$\{(\w+):\?[^}:]+\}", host), (
            f"host side {host!r} is not a required variable: {entry!r}"
        )
        host_sides[re.match(r"\$\{(\w+)", host).group(1)] = entry
    assert set(host_sides) == set(REQUIRED_HOST_VARS)


def test_compose_files_carry_no_machine_specific_absolute_path_default():
    for name in ("console-compose.yaml", "console-site.yaml", "console-extra-output.yaml"):
        text = (DOCKER_DIR / name).read_text()
        # Defaults inside the container's own /data volume are fine; a default
        # naming any other absolute path would be a host location.
        assert not re.search(r"\$\{\w+:-/(?!data/)", text), f"{name} has a host-path default"
        for prefix in ("/home/", "/mnt/", "/opt/"):
            assert prefix not in text, f"{name} names a {prefix} location"


def test_container_side_targets_are_fixed_literals():
    """An env override must move only the HOST source. The app looks these
    paths up at fixed locations inside the container, so keying the container
    side off the same variable would let an override move the target away from
    where the app reads it."""
    for entry in _compose()["services"]["console-base"]["volumes"]:
        target = _target(entry)
        if target.startswith("${"):
            # staging/output are bound at the SAME path inside the container on
            # purpose: queue rows hold absolute paths under them.
            assert re.fullmatch(r"\$\{GMS_(STAGING|OUTPUT)_DIR\}", target), entry
        else:
            assert "${" not in target


def test_seo_storage_credentials_and_token_are_mounted_under_the_image_copy():
    volumes = _service()["volumes"]
    assert "GMS_SEO_STORAGE_DIR" in _host_for_target(volumes, QUEUE_TARGET)
    assert "GMS_SEO_CREDENTIALS_DIR" in _host_for_target(volumes, SEO_CREDENTIALS_TARGET)
    assert "GMS_SEO_YT_TOKEN_ENC" in _host_for_target(volumes, SEO_TOKEN_TARGET)
    read_only = {_target(v) for v in volumes if v.endswith(":ro")}
    assert {SEO_CREDENTIALS_TARGET, SEO_TOKEN_TARGET, *SECRET_TARGETS.values()} <= read_only


def test_ticker_enabled():
    assert _env(_service()).get("GMS_CONSOLE_TICKER") == "1"


def test_supervised_env_present():
    """The container's restart policy + the console job store are the drive
    pipeline's supervisor -- assert it so --no-tmux runs proceed."""
    assert _env(_service()).get("GMS_SUPERVISED") == "1"


def test_console_token_env_present():
    assert "GMS_CONSOLE_TOKEN" in _env(_service())


def test_healthcheck_present():
    service = _service()
    assert "healthcheck" in service
    assert "/healthz" in " ".join(service["healthcheck"]["test"])


def test_runtime_state_defaults_live_in_the_data_volume():
    """The GPU lock and comments ledger default to the persistent /data volume,
    not a host path that does not exist in the container."""
    env = _env(_service())
    assert env["GMS_GPU_LOCK_PATH"] == "${GMS_GPU_LOCK_PATH:-/data/gpu.lock}"
    assert env["GMS_COMMENTS_PATH"] == "${GMS_COMMENTS_PATH:-/data/comments.jsonl}"


def test_host_data_group_defaults_to_a_generic_gid_and_can_be_overridden():
    base = _compose()["services"]["console-base"]
    assert [str(g) for g in base["group_add"]] == ["${GMS_HOST_GID:-1000}"]


def test_main_volume_is_pinned_to_the_live_volume_name():
    # The compose project name (derived from the directory it's run from)
    # must never change which volume gets mounted -- pin it explicitly.
    assert _compose()["volumes"]["console-data"]["name"] == "docker_console-data"


def test_main_service_has_a_stable_container_name():
    assert _service().get("container_name") == "gomoveshift-console"


def test_image_is_built_from_the_three_sibling_checkouts():
    build = _compose()["services"]["console-base"]["build"]
    assert build["context"] == "../.."
    assert build["dockerfile"] == "framegrade/docker/console.Dockerfile"
    text = _dockerfile_text()
    for sibling in ("gps-journey", "waypost", "framegrade"):
        assert f"COPY {sibling} /app/{sibling}" in text


# --- rendered config (skipped where the docker CLI is unusable) ---------------


def _render_env(tmp_path, **overrides):
    """A clean environment with every required host variable set to a path
    under ``tmp_path`` -- and no inherited GMS_* values from the test session."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GMS_")}
    host = {
        "GMS_STAGING_DIR": tmp_path / "staging",
        "GMS_OUTPUT_DIR": tmp_path / "output",
        "GMS_DRIVE_CREDENTIALS_ENC": tmp_path / "creds" / "drive_credentials.json.enc",
        "GMS_DRIVE_TOKEN_ENC": tmp_path / "creds" / "drive_token.json.enc",
        "GMS_AGE_KEY_FILE": tmp_path / "creds" / "age.key",
        "GMS_SOPS_CONFIG": tmp_path / "creds" / ".sops.yaml",
        "GMS_SEO_STORAGE_DIR": tmp_path / "seo" / "data",
        "GMS_SEO_CREDENTIALS_DIR": tmp_path / "seo" / "credentials",
        "GMS_SEO_YT_TOKEN_ENC": tmp_path / "seo" / "youtube_upload_token.json.enc",
        "GMS_SITE_HOST_PATH": tmp_path / "site",
        "GMS_NIGHT_HOST_ROOT": tmp_path / "cards",
        EXTRA_OUTPUT_VAR: tmp_path / "output2",
    }
    env.update({k: str(v) for k, v in host.items()})
    env.update(overrides)
    return env


def _run_compose_config(files, env):
    return subprocess.run(  # noqa: S603
        ["docker", "compose"]  # noqa: S607
        + [arg for f in files for arg in ("-f", str(DOCKER_DIR / f))]
        + ["config"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _rendered_config(tmp_path, *files, **overrides):
    result = _run_compose_config(files, _render_env(tmp_path, **overrides))
    if result.returncode != 0:
        pytest.skip(f"docker compose not usable here: {result.stderr}")
    return yaml.safe_load(result.stdout)


@pytest.mark.parametrize("missing", REQUIRED_HOST_VARS)
def test_compose_refuses_to_render_when_a_required_host_path_is_unset(tmp_path, missing):
    env = _render_env(tmp_path)
    probe = _run_compose_config(["console-compose.yaml"], env)
    if probe.returncode != 0:
        pytest.skip(f"docker compose not usable here: {probe.stderr}")
    del env[missing]
    result = _run_compose_config(["console-compose.yaml"], env)
    assert result.returncode != 0, f"{missing} unset yet compose rendered a config"
    assert missing in result.stderr, result.stderr


def test_main_console_port_defaults_to_loopback_host_bind(tmp_path):
    """A default install must not publish the unauthenticated console on every
    host interface -- the published address must default to loopback, not the
    wildcard that a bare ``"8787:8787"`` mapping produces."""
    rendered = _rendered_config(tmp_path, "console-compose.yaml")
    service = rendered["services"]["gomoveshift-console"]
    for port in service["ports"]:
        assert port.get("host_ip") == "127.0.0.1", f"expected loopback host bind, got {port!r}"


def test_console_bind_override_moves_the_published_address(tmp_path):
    rendered = _rendered_config(tmp_path, "console-compose.yaml", GMS_CONSOLE_BIND="0.0.0.0")  # noqa: S104
    for port in rendered["services"]["gomoveshift-console"]["ports"]:
        assert port.get("host_ip") == "0.0.0.0"  # noqa: S104


def test_rendered_mounts_bind_the_operator_paths_at_fixed_container_targets(tmp_path):
    rendered = _rendered_config(tmp_path, "console-compose.yaml")["services"]["gomoveshift-console"]
    by_target = {v["target"]: v["source"] for v in rendered["volumes"]}
    env = _render_env(tmp_path)
    for var, target in SECRET_TARGETS.items():
        assert by_target[target] == env[var]
        assert rendered["environment"][var] == target
    assert by_target[QUEUE_TARGET] == env["GMS_SEO_STORAGE_DIR"]
    assert by_target[SEO_CREDENTIALS_TARGET] == env["GMS_SEO_CREDENTIALS_DIR"]
    assert by_target[SEO_TOKEN_TARGET] == env["GMS_SEO_YT_TOKEN_ENC"]
    assert by_target[env["GMS_STAGING_DIR"]] == env["GMS_STAGING_DIR"]
    assert by_target[env["GMS_OUTPUT_DIR"]] == env["GMS_OUTPUT_DIR"]


def test_operator_overrides_move_only_the_host_side_of_the_secret_mounts(tmp_path):
    """Regression: when both sides of a mount were keyed off one env var, an
    operator override moved the container-side target too, so the app, which
    reads these paths at fixed locations inside the container, could no longer
    find its credentials. Targets and the
    in-container env values are fixed literals; only the HOST source moves."""
    overrides = {
        "GMS_DRIVE_CREDENTIALS_ENC": str(tmp_path / "custom" / "drive_credentials.json.enc"),
        "GMS_DRIVE_TOKEN_ENC": str(tmp_path / "custom" / "drive_token.json.enc"),
        "GMS_AGE_KEY_FILE": str(tmp_path / "custom" / "age.key"),
        "GMS_SOPS_CONFIG": str(tmp_path / "custom" / ".sops.yaml"),
    }
    rendered = _rendered_config(tmp_path, "console-compose.yaml", **overrides)
    service = rendered["services"]["gomoveshift-console"]
    by_target: dict[str, list[str]] = {}
    for v in service["volumes"]:
        by_target.setdefault(v["target"], []).append(v["source"])
    for var, target in SECRET_TARGETS.items():
        assert service["environment"][var] == target
        assert by_target[target] == [overrides[var]], (
            f"{target}: an env override must not add a second mount or move the target"
        )


def test_site_overlay_adds_the_site_and_card_root_read_only(tmp_path):
    rendered = _rendered_config(tmp_path, "console-compose.yaml", "console-site.yaml")
    service = rendered["services"]["gomoveshift-console"]
    env = _render_env(tmp_path)
    by_target = {v["target"]: v for v in service["volumes"]}
    assert by_target[SITE_TARGET]["source"] == env["GMS_SITE_HOST_PATH"]
    assert by_target[NIGHT_TARGET]["source"] == env["GMS_NIGHT_HOST_ROOT"]
    assert by_target[SITE_TARGET].get("read_only") is True
    assert by_target[NIGHT_TARGET].get("read_only") is True
    assert service["environment"]["GMS_SITE_PATH"] == SITE_TARGET
    assert service["environment"]["GMS_NIGHT_SOURCE_ROOTS"] == NIGHT_TARGET


def test_without_the_overlay_the_site_and_card_root_are_not_mounted(tmp_path):
    rendered = _rendered_config(tmp_path, "console-compose.yaml")
    service = rendered["services"]["gomoveshift-console"]
    assert {SITE_TARGET, NIGHT_TARGET}.isdisjoint(v["target"] for v in service["volumes"])
    assert set(SITE_ENV).isdisjoint(service["environment"])


@pytest.mark.parametrize("missing", ["GMS_SITE_HOST_PATH", "GMS_NIGHT_HOST_ROOT"])
def test_site_overlay_requires_both_of_its_host_paths(tmp_path, missing):
    env = _render_env(tmp_path)
    probe = _run_compose_config(["console-compose.yaml", "console-site.yaml"], env)
    if probe.returncode != 0:
        pytest.skip(f"docker compose not usable here: {probe.stderr}")
    del env[missing]
    result = _run_compose_config(["console-compose.yaml", "console-site.yaml"], env)
    assert result.returncode != 0
    assert missing in result.stderr


def test_extra_output_overlay_mounts_the_second_root_at_the_same_path(tmp_path):
    rendered = _rendered_config(tmp_path, "console-compose.yaml", "console-extra-output.yaml")
    service = rendered["services"]["gomoveshift-console"]
    root = _render_env(tmp_path)[EXTRA_OUTPUT_VAR]
    by_target = {v["target"]: v for v in service["volumes"]}
    assert by_target[root]["source"] == root
    assert not by_target[root].get("read_only")


def test_extra_output_override_moves_both_sides_to_the_new_path(tmp_path):
    """The variable still overrides: the operator's path is mounted at itself."""
    custom = str(tmp_path / "elsewhere")
    rendered = _rendered_config(
        tmp_path, "console-compose.yaml", "console-extra-output.yaml", **{EXTRA_OUTPUT_VAR: custom}
    )
    volumes = rendered["services"]["gomoveshift-console"]["volumes"]
    assert any(v["source"] == custom and v["target"] == custom for v in volumes)


def test_without_the_extra_output_overlay_the_variable_mounts_nothing(tmp_path):
    root = _render_env(tmp_path)[EXTRA_OUTPUT_VAR]
    rendered = _rendered_config(tmp_path, "console-compose.yaml")
    volumes = rendered["services"]["gomoveshift-console"]["volumes"]
    assert root not in {v["target"] for v in volumes}


def test_extra_output_overlay_requires_its_host_path(tmp_path):
    env = _render_env(tmp_path)
    files = ["console-compose.yaml", "console-extra-output.yaml"]
    probe = _run_compose_config(files, env)
    if probe.returncode != 0:
        pytest.skip(f"docker compose not usable here: {probe.stderr}")
    del env[EXTRA_OUTPUT_VAR]
    result = _run_compose_config(files, env)
    assert result.returncode != 0
    assert EXTRA_OUTPUT_VAR in result.stderr


def test_service_helper_matches_the_rendered_config(tmp_path):
    rendered = _rendered_config(tmp_path, "console-compose.yaml")["services"]
    main = rendered["gomoveshift-console"]
    ours = _service()["volumes"]
    # A ${VAR}-keyed target follows whatever the environment sets, so compare
    # the fixed targets and the count.
    fixed = {_target(v) for v in ours if "${" not in _target(v)}
    assert fixed <= {v["target"] for v in main["volumes"]}
    assert len(main["volumes"]) == len(ours)
    assert set(main["environment"]) == {_env_key(e) for e in _service()["environment"]}


# --- static checks that hold even where docker is unusable --------------------


def test_site_overlay_text_is_read_only_and_variable_driven():
    volumes = _site_overlay()["services"]["gomoveshift-console"]["volumes"]
    assert {_target(v) for v in volumes} == {SITE_TARGET, NIGHT_TARGET}
    assert all(v.endswith(":ro") for v in volumes)
    assert "GMS_SITE_HOST_PATH" in _host_for_target(volumes, SITE_TARGET)
    assert "GMS_NIGHT_HOST_ROOT" in _host_for_target(volumes, NIGHT_TARGET)
    env = _site_overlay()["services"]["gomoveshift-console"]["environment"]
    assert sorted(e.split("=", 1)[0] for e in env) == sorted(SITE_ENV)


def test_extra_output_overlay_text_is_a_required_same_path_read_write_mount():
    volumes = _extra_output_overlay()["services"]["gomoveshift-console"]["volumes"]
    assert len(volumes) == 1
    (entry,) = volumes
    host = _VOLUME.match(entry).group("host")
    assert re.fullmatch(rf"\$\{{{EXTRA_OUTPUT_VAR}:\?[^}}:]+\}}", host), entry
    assert _target(entry) == "${" + EXTRA_OUTPUT_VAR + "}"
    assert not entry.endswith(":ro")


def test_base_service_carries_no_second_output_root():
    compose = _compose()["services"]
    for name in ("console-base", "gomoveshift-console"):
        text = " ".join(compose[name].get("volumes", []) + compose[name].get("environment", []))
        assert EXTRA_OUTPUT_VAR not in text


def test_base_service_carries_no_site_or_card_root():
    """The library and night tabs are unconfigured by default: neither the
    shared base nor the concrete service may carry the mounts or the env."""
    compose = _compose()["services"]
    for name in ("console-base", "gomoveshift-console"):
        service = compose[name]
        assert {SITE_TARGET, NIGHT_TARGET}.isdisjoint(
            _target(v) for v in service.get("volumes", [])
        )
        assert set(SITE_ENV).isdisjoint(e.split("=", 1)[0] for e in service.get("environment", []))


def _port_mapping_lines(name: str) -> list[str]:
    """Every published-port mapping in the file, commented lines included.

    The rendered-config tests above skip when the docker CLI is unusable, so
    on a box without docker they cannot fail -- and a commented block is
    invisible to ``docker compose config`` even where it is. This reads the
    file as text so neither hole hides a wildcard bind.
    """
    lines = []
    for raw in (DOCKER_DIR / name).read_text().splitlines():
        stripped = raw.lstrip().lstrip("#").strip()
        if stripped.startswith('- "') and stripped.endswith('"') and ":8787" in stripped:
            lines.append(stripped)
    return lines


def test_no_port_mapping_omits_a_host_bind_even_in_comments():
    """A commented-out mapping is a template someone will uncomment: a bare
    ``- "8788:8787"`` publishes on every interface."""
    mappings = _port_mapping_lines("console-compose.yaml")
    assert mappings, "no port mapping found in console-compose.yaml -- the scan is broken"
    for mapping in mappings:
        assert mapping.count(":") >= 2, (
            f"port mapping {mapping} has no host address, so it publishes on every "
            "interface; use ${GMS_CONSOLE_BIND:-127.0.0.1}:<host>:<container>"
        )
