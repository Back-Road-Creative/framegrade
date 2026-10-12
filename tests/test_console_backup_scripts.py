"""Console backup/restore must target the real compose-created volume name
(docker_console-data), never a stale project-prefixed default that docker
silently auto-creates as an empty volume. The backup directory defaults to the
per-user data home and can be overridden; extra volumes are opt-in."""

import os
import subprocess
import tempfile
from pathlib import Path

_SERVICE_ROOT = Path(__file__).parent.parent
_DOCKER_DIR = _SERVICE_ROOT / "docker"
_BACKUP_SCRIPT = _DOCKER_DIR / "console-backup.sh"
_RESTORE_SCRIPT = _DOCKER_DIR / "console-restore.sh"

_STALE_DEFAULT = "framegrade_console-data"


def _bash_syntax_ok(script: Path) -> None:
    result = subprocess.run(  # noqa: S603
        ["bash", "-n", str(script)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{script}: {result.stderr}"


def test_backup_and_restore_scripts_are_valid_bash():
    _bash_syntax_ok(_BACKUP_SCRIPT)
    _bash_syntax_ok(_RESTORE_SCRIPT)


def test_stale_wrong_volume_default_is_gone():
    for script in (_BACKUP_SCRIPT, _RESTORE_SCRIPT):
        text = script.read_text()
        assert _STALE_DEFAULT not in text, f"{script} still defaults to the wrong volume"


def test_backup_default_targets_real_compose_volume_name():
    text = _BACKUP_SCRIPT.read_text()
    assert "docker_console-data" in text


def test_restore_default_targets_real_compose_volume_name():
    text = _RESTORE_SCRIPT.read_text()
    assert "docker_console-data" in text


def test_scripts_never_let_docker_auto_create_the_volume():
    for script in (_BACKUP_SCRIPT, _RESTORE_SCRIPT):
        text = script.read_text()
        assert "docker volume create" not in text, f"{script} still auto-creates"
        assert "docker volume inspect" in text, f"{script} is missing an existence check"


def test_optional_volumes_are_opt_in_not_hardcoded():
    text = _BACKUP_SCRIPT.read_text()
    assert "CONSOLE_OPTIONAL_VOLUME_NAMES" in text
    # No second-instance volume name is baked in as a default.
    assert "console-data-" not in text


def test_restore_requires_explicit_volume_name_argument():
    text = _RESTORE_SCRIPT.read_text()
    # restore must no longer fall back silently to a bare default; it should
    # accept the target volume as an explicit CLI argument.
    assert "$2" in text


def _fake_docker_bin(tmp_path, missing=()):
    """A PATH-shadowing fake `docker`: `volume inspect V` fails for V in
    `missing`; `volume ls -q` lists an unrelated volume; `run ...` just logs
    to $FAKE_DOCKER_LOG and succeeds -- no real container is started."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    script = bindir / "docker"
    missing_list = " ".join(missing)
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'MISSING="{missing_list}"\n'
        'if [ "$1" = "volume" ] && [ "$2" = "inspect" ]; then\n'
        '  for m in $MISSING; do [ "$m" = "$3" ] && exit 1; done\n'
        "  exit 0\n"
        'elif [ "$1" = "volume" ] && [ "$2" = "ls" ]; then\n'
        '  echo "${FAKE_DOCKER_VOLUMES:-unrelated-volume}" | tr " " "\\n"\n'
        "  exit 0\n"
        'elif [ "$1" = "run" ]; then\n'
        '  echo "docker run $*" >> "${FAKE_DOCKER_LOG:-/dev/null}"\n'
        "  exit 0\n"
        "else\n"
        "  exit 0\n"
        "fi\n"
    )
    script.chmod(0o755)
    return bindir


def _run_console_backup(tmp_path, missing=(), extra_env=None):
    bindir = _fake_docker_bin(tmp_path, missing=missing)
    backup_dir = tmp_path / "backups"
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "CONSOLE_BACKUP_DIR": str(backup_dir),
        "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
        **(extra_env or {}),
    }
    return subprocess.run(  # noqa: S603
        ["bash", str(_BACKUP_SCRIPT)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_missing_required_volume_fails_loudly():
    # Today the script `continue`s past a missing volume and exits 0 -- a
    # false-success backup that captured nothing for the required volume.
    # A bare non-zero check is not enough: the script also has a latent
    # set -e/pipefail crash elsewhere (retention pruning on an empty glob)
    # that exits non-zero for an unrelated reason on every invocation under
    # this fake-docker harness, which would make this assertion pass on the
    # pre-fix script for the wrong cause. Require BOTH the specific exit
    # code this failure path uses AND that the output actually names the
    # missing volume -- a crash elsewhere cannot produce that message.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(tmp_path, missing=("docker_console-data",))
    assert result.returncode == 3, (
        f"missing required volume must exit 3 (the required-volume-missing path), "
        f"not just non-zero; got {result.returncode}. stdout={result.stdout} "
        f"stderr={result.stderr}"
    )
    assert "REQUIRED volume missing: docker_console-data" in result.stderr, (
        f"failure output must name the missing required volume. stderr={result.stderr}"
    )


_OPTIONAL = {"CONSOLE_OPTIONAL_VOLUME_NAMES": "extra-data"}


def test_missing_optional_volume_still_succeeds():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(tmp_path, missing=("extra-data",), extra_env=_OPTIONAL)
    assert result.returncode == 0, (
        f"a missing OPTIONAL volume must not fail the run. stdout={result.stdout} "
        f"stderr={result.stderr}"
    )
    assert "skipping optional extra-data" in result.stderr


def test_required_and_optional_volumes_are_both_backed_up():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(tmp_path, missing=(), extra_env=_OPTIONAL)
        assert result.returncode == 0
        log = (tmp_path / "docker.log").read_text() if (tmp_path / "docker.log").exists() else ""
        assert "docker_console-data" in log
        assert "extra-data" in log


def test_without_optional_volumes_only_the_required_one_is_backed_up():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(tmp_path, missing=())
        assert result.returncode == 0
        log = (tmp_path / "docker.log").read_text()
        assert "docker_console-data" in log
        assert "extra-data" not in log
        assert log.count("docker run") == 1


def test_unlisted_console_data_volume_is_reported_not_silently_skipped():
    """Another console-data volume on the host with no optional list configured
    is named on stderr; the run itself still succeeds."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(
            tmp_path,
            extra_env={"FAKE_DOCKER_VOLUMES": "docker_console-data other_console-data"},
        )
    assert result.returncode == 0, result.stderr
    assert "other_console-data" in result.stderr
    assert "CONSOLE_OPTIONAL_VOLUME_NAMES" in result.stderr


def test_no_unlisted_volume_note_when_the_optional_list_is_set():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(
            tmp_path,
            extra_env={
                "FAKE_DOCKER_VOLUMES": "docker_console-data other_console-data",
                "CONSOLE_OPTIONAL_VOLUME_NAMES": "other_console-data",
            },
        )
    assert result.returncode == 0, result.stderr
    assert "not backed up" not in result.stderr


def test_no_unlisted_volume_note_for_the_single_required_volume():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = _run_console_backup(
            tmp_path, extra_env={"FAKE_DOCKER_VOLUMES": "docker_console-data"}
        )
    assert result.returncode == 0, result.stderr
    assert "not backed up" not in result.stderr


def _drill_env(tmp_path, **overrides):
    env = {k: v for k, v in os.environ.items() if k not in _DATA_HOME_VARS | {"CONSOLE_BACKUP_DIR"}}
    env["PATH"] = f"{_fake_docker_bin(tmp_path, missing=())}:{os.environ['PATH']}"
    env.update(overrides)
    return env


_DATA_HOME_VARS = {"GMS_DATA_HOME", "XDG_DATA_HOME"}


def _drill(tmp_path, **overrides):
    return subprocess.run(  # noqa: S603
        ["bash", str(_BACKUP_SCRIPT), "--restore-drill"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        env=_drill_env(tmp_path, **overrides),
    )


def test_backup_dir_defaults_under_home_data_dir(tmp_path):
    """Neither CONSOLE_BACKUP_DIR nor a data-home variable set: the default is
    ~/.local/share/gomoveshift/console-backup, not a machine-specific mount."""
    result = _drill(tmp_path, HOME=str(tmp_path))
    expected = tmp_path / ".local" / "share" / "gomoveshift" / "console-backup"
    assert f"no archive in {expected}" in result.stderr, result.stderr


def test_backup_dir_default_follows_xdg_data_home(tmp_path):
    xdg = tmp_path / "xdg"
    result = _drill(tmp_path, HOME=str(tmp_path / "home"), XDG_DATA_HOME=str(xdg))
    assert f"no archive in {xdg / 'gomoveshift' / 'console-backup'}" in result.stderr, result.stderr


def test_backup_dir_default_follows_gms_data_home(tmp_path):
    data = tmp_path / "gms-data"
    result = _drill(
        tmp_path,
        HOME=str(tmp_path / "home"),
        XDG_DATA_HOME=str(tmp_path / "xdg"),
        GMS_DATA_HOME=str(data),
    )
    assert f"no archive in {data / 'console-backup'}" in result.stderr, result.stderr


def test_console_backup_dir_override_beats_every_default(tmp_path):
    override = tmp_path / "elsewhere"
    result = _drill(
        tmp_path,
        HOME=str(tmp_path / "home"),
        GMS_DATA_HOME=str(tmp_path / "gms-data"),
        CONSOLE_BACKUP_DIR=str(override),
    )
    assert f"no archive in {override}" in result.stderr, result.stderr


def _make_fixture_archive(backup_dir, volume, sqlite_bytes=None, corrupt=False):
    """Build a docker/console-backup.sh-shaped archive (members `./name`,
    like `tar -czf ... -C /data .`) containing a real (or corrupt) sqlite3
    database, mirroring what console-backup.sh actually produces."""
    import sqlite3
    import tarfile

    backup_dir.mkdir(parents=True, exist_ok=True)
    stage = backup_dir / "_stage"
    stage.mkdir(exist_ok=True)
    db_path = stage / "jobs.sqlite3"
    if corrupt:
        db_path.write_bytes(b"not a real sqlite database, just garbage bytes")
    else:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO jobs (name) VALUES ('drill-fixture')")
        conn.commit()
        conn.close()
    archive = backup_dir / f"{volume}-2000-01-01-030000.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(db_path, arcname="./jobs.sqlite3")
    return archive


def _fake_sqlite3_bin(tmp_path, result="ok"):
    """A PATH-shadowing fake `sqlite3` that answers a fixed
    `PRAGMA integrity_check` result. Needed because a host/CI runner may have
    no real `sqlite3` binary, which would otherwise force every drill down the
    INCOMPLETE path and never actually exercise OK vs CORRUPT."""
    bindir = tmp_path / "fakesqlitebin"
    bindir.mkdir()
    script = bindir / "sqlite3"
    script.write_text(f'#!/usr/bin/env bash\necho "{result}"\n')
    script.chmod(0o755)
    return bindir


def _run_console_backup_drill(tmp_path, backup_dir, sqlite3_result=None):
    bindir = _fake_docker_bin(tmp_path, missing=())
    path = f"{bindir}:{os.environ['PATH']}"
    if sqlite3_result is not None:
        sqbindir = _fake_sqlite3_bin(tmp_path, result=sqlite3_result)
        path = f"{sqbindir}:{path}"
    env = {
        **os.environ,
        "PATH": path,
        "CONSOLE_BACKUP_DIR": str(backup_dir),
        "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
    }
    return subprocess.run(  # noqa: S603
        ["bash", str(_BACKUP_SCRIPT), "--restore-drill"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_restore_drill_over_real_archive_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        backup_dir = tmp_path / "backups"
        _make_fixture_archive(backup_dir, "docker_console-data")
        result = _run_console_backup_drill(tmp_path, backup_dir, sqlite3_result="ok")
    assert "restore-drill OK" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0


def test_restore_drill_empty_backup_dir_fails_and_names_the_volume():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        result = _run_console_backup_drill(tmp_path, backup_dir)
    assert result.returncode != 0
    assert "docker_console-data" in (result.stdout + result.stderr)


def test_restore_drill_corrupt_database_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        backup_dir = tmp_path / "backups"
        _make_fixture_archive(backup_dir, "docker_console-data", corrupt=True)
        result = _run_console_backup_drill(tmp_path, backup_dir, sqlite3_result="corrupt")
    assert result.returncode != 0
    assert "CORRUPT" in (result.stdout + result.stderr)


def test_restore_drill_runs_no_docker():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        backup_dir = tmp_path / "backups"
        _make_fixture_archive(backup_dir, "docker_console-data")
        _run_console_backup_drill(tmp_path, backup_dir)
        log = tmp_path / "docker.log"
        assert not log.exists() or log.read_text() == ""
