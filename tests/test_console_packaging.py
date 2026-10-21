"""Console Docker packaging -- compose shape, no secrets in the image, backup
script sanity, and a drift gate between the compose env passthrough and the settings
registry (the same registry `gomoveshift/console/settings.py` reads at runtime)."""

import re
import subprocess
from pathlib import Path

import yaml
from _compose_helpers import resolved_service

from gomoveshift.console import settings as console_settings

_SERVICE_ROOT = Path(__file__).parent.parent
_DOCKER_DIR = _SERVICE_ROOT / "docker"
_COMPOSE_PATH = _DOCKER_DIR / "console-compose.yaml"
_DOCKERFILE_PATH = _DOCKER_DIR / "console.Dockerfile"
_BACKUP_SCRIPT = _DOCKER_DIR / "console-backup.sh"
_RESTORE_SCRIPT = _DOCKER_DIR / "console-restore.sh"

_SECRET_LOOKALIKE = re.compile(
    r"(?i)(password|secret|api[_-]?key|token)\s*[:=]\s*['\"]?[A-Za-z0-9]"
)


def _load_compose():
    return yaml.safe_load(_COMPOSE_PATH.read_text())


def _gomoveshift_service():
    return resolved_service("gomoveshift-console", _load_compose())


def test_compose_parses_and_has_gomoveshift_service():
    compose = _load_compose()
    assert "gomoveshift-console" in compose["services"]


def test_gomoveshift_service_restarts_and_exposes_8787():
    svc = _gomoveshift_service()
    assert svc["restart"] == "unless-stopped"
    assert any("8787" in str(p) for p in svc["ports"])


def test_gomoveshift_service_uses_named_console_data_volume():
    compose = _load_compose()
    assert "console-data" in compose.get("volumes", {})
    svc = _gomoveshift_service()
    named_mounts = [v for v in svc["volumes"] if str(v).startswith("console-data:")]
    assert named_mounts, svc["volumes"]


#: Container-only plumbing (data dir) and the auth secret: neither is a user
#: setting, and the token must never appear in the settings registry the
#: /settings tab renders (it would set the bearer gate in tests and is a secret).
_NOT_A_SETTING = {"GMS_CONSOLE_DATA_DIR", "GMS_CONSOLE_TOKEN", "GMS_HOST_SPOOL_DIR"}


def test_gomoveshift_service_env_passthrough_matches_registry():
    """Drift gate: every GMS_* var named in the compose passthrough must exist in the
    registry the console itself reads (gomoveshift/console/registry/gms_settings.yaml)."""
    svc = _gomoveshift_service()
    registry_names = {s.name for s in console_settings.load_registry()}
    env = svc.get("environment", [])
    passthrough_names = set()
    for entry in env:
        name = entry.split("=", 1)[0].split(":", 1)[0] if isinstance(entry, str) else None
        if name and name.startswith("GMS_") and name not in _NOT_A_SETTING:
            passthrough_names.add(name)
    assert passthrough_names, "sanity: compose should pass through at least one GMS_* var"
    assert not (passthrough_names - registry_names)


def test_gomoveshift_service_passes_through_notify_env():
    """GMS_CONSOLE_NOTIFY_URL / GMS_CONSOLE_URL are read by
    gomoveshift/console/notify.py, so they must be wired through the compose
    environment or the failure webhook is a silent no-op in the container."""
    svc = _gomoveshift_service()
    env = svc.get("environment", [])
    names = {entry.split("=", 1)[0] for entry in env if isinstance(entry, str)}
    assert "GMS_CONSOLE_NOTIFY_URL" in names
    assert "GMS_CONSOLE_URL" in names


_INSTALL_MD = _SERVICE_ROOT / "docs" / "install.md"
_DOCKER_README = _DOCKER_DIR / "README.md"

#: Matches a `docker compose ... up ...` invocation up to the end of the line
#: (or a closing backtick, for inline-code mentions).
_COMPOSE_UP_LINE = re.compile(r"docker compose\b.*\bup\b[^`\n]*")

#: `up`'s own flags -- if the last token on the line is one of these (or the
#: elided `...`), no service was named.
_FLAG_TOKENS = {"-d", "--build", "--force-recreate", "..."}


def test_docs_never_run_a_bare_compose_up():
    """console-compose.yaml also defines console-base, a shared template never
    meant to run standalone -- a bare `docker compose ... up` (no service
    names) would start it too: no ports, but it mounts console-data and runs
    a second background ticker against the same job store. Every documented
    `up` invocation must name a service explicitly."""
    for doc_path in (_INSTALL_MD, _DOCKER_README):
        text = doc_path.read_text()
        for line in _COMPOSE_UP_LINE.findall(text):
            trailing = line.rsplit("up", 1)[-1].strip()
            assert trailing, f"{doc_path}: bare `up` with nothing after it: {line!r}"
            last_token = trailing.rstrip("`").split()[-1]
            assert last_token not in _FLAG_TOKENS and not last_token.startswith("-"), (
                f"{doc_path}: `up` line names no service (ends in {last_token!r}): {line!r}"
            )


def test_install_md_documents_auth_token_and_notify_url():
    """docs/install.md must document how to set the auth token
    (gomoveshift/console/auth.py) and the notify URL."""
    text = _INSTALL_MD.read_text()
    assert "future auth gate" not in text
    assert "GMS_CONSOLE_NOTIFY_URL" in text
    assert "?token=" in text


def test_dockerfile_has_no_secret_lookalikes():
    text = _DOCKERFILE_PATH.read_text()
    assert not _SECRET_LOOKALIKE.search(text)


def test_dockerfile_pins_pip_retries_and_timeout_before_installing():
    # A single PyPI read timeout under pip's 15 s default would fail the whole
    # image build. The build must set a patient timeout and more retries BEFORE
    # the first pip install, so a network blip retries instead of failing.
    text = _DOCKERFILE_PATH.read_text()
    # body = the ENV line plus every backslash-continued line after it
    env = re.search(r"^ENV\s+(?P<body>PIP_(?:[^\n]*\\\n)*[^\n]*)$", text, re.M)
    assert env, "no ENV line setting PIP_* before pip install"
    body = env.group("body")
    timeout = re.search(r"PIP_DEFAULT_TIMEOUT=(\d+)", body)
    retries = re.search(r"PIP_RETRIES=(\d+)", body)
    assert timeout and int(timeout.group(1)) >= 60
    assert retries and int(retries.group(1)) >= 10
    first_pip = text.find(next((s for s in _run_steps(text) if "pip install" in s), "\0"))
    assert first_pip >= 0, "no RUN ... pip install step"
    assert env.start() < first_pip


def _run_steps(text):
    """Each RUN instruction with its backslash-continued lines joined."""
    return [m.group(0) for m in re.finditer(r"^RUN\b(?:[^\n]*\\\n)*[^\n]*$", text, re.M)]


def test_dockerfile_pip_installs_reuse_a_buildkit_cache():
    # The source COPY sits above the pip step, so every code change rebuilds it;
    # with --no-cache-dir each rebuild re-downloaded every wheel (onnxruntime-gpu
    # alone is ~283 MB) and a flaky network failed the build. A BuildKit cache
    # mount keeps the wheels on the host between builds without putting them in
    # the image.
    text = _DOCKERFILE_PATH.read_text()
    pip_steps = [step for step in _run_steps(text) if "pip install" in step]
    assert pip_steps, "no RUN ... pip install step"
    for step in pip_steps:
        assert "--no-cache-dir" not in step, step
        assert re.search(r"--mount=type=cache,[^\s]*target=/root/\.cache/pip", step), step


def test_dockerfile_installs_ffmpeg_and_runs_non_root():
    text = _DOCKERFILE_PATH.read_text()
    assert "ffmpeg" in text
    assert "USER " in text and "USER root" not in text.split("USER ", 1)[-1].split("\n")[0]


def test_backup_and_restore_scripts_are_valid_bash():
    for script in (_BACKUP_SCRIPT, _RESTORE_SCRIPT):
        result = subprocess.run(  # noqa: S603
            ["bash", "-n", str(script)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_backup_script_has_no_secret_lookalikes():
    text = _BACKUP_SCRIPT.read_text()
    assert not _SECRET_LOOKALIKE.search(text)


def test_console_base_joins_the_host_data_group():
    """Bind mounts are host dirs owned by a host group (mode 2775) while the
    image runs as its own system user (uid 999), so staging and SEO storage are
    not writable from inside the container by default. ``group_add`` gives the
    container user the host gid (``GMS_HOST_GID``, default 1000), so
    group-writable mounts work without chowning the host tree."""
    base = _load_compose()["services"]["console-base"]
    groups = [str(g) for g in base.get("group_add", [])]
    assert "${GMS_HOST_GID:-1000}" in groups, groups
