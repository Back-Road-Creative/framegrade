"""Config.load() deep-merges user overrides on top of the repo config.

Precedence (lowest → highest):
  1. Pydantic defaults
  2. ``./config.yaml`` (or ``./config.yml``) — repo / project defaults
  3. ``~/.config/gomoveshift/config.yaml`` — operator overrides

Nested mappings merge key-by-key; non-mapping values (including lists)
replace wholesale. Rationale: a partial user file that only sets
``drive.shorts_publish_times`` must NOT wipe sibling keys like
``drive.gps_quality`` — that was the prior bug that shadowed the
user override entirely and forced operators to edit tracked config.
"""

from pathlib import Path

import yaml

from gomoveshift.config import Config


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_user_override_beats_project_config(tmp_path, monkeypatch):
    """A field set in ~/.config wins over the same field in ./config.yaml."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()

    _write(project / "config.yaml", {"drive": {"shorts_publish_times": ["09:00"]}})
    _write(
        home / ".config" / "gomoveshift" / "config.yaml",
        {"drive": {"shorts_publish_times": ["10:00"]}},
    )

    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    # Force Path.home() to re-resolve.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load()
    assert cfg.drive.shorts_publish_times == ["10:00"]


def test_user_override_preserves_project_siblings(tmp_path, monkeypatch):
    """Partial user override must NOT wipe sibling keys under the same section.

    This is the regression guard: a user file that only flips
    ``drive.shorts_publish_times`` to 10:00 must keep the project's
    ``drive.ost_title_crossfade_sec`` value (and any other sibling
    fields). A shallow dict replace would reset them to defaults.
    """
    project = tmp_path / "project"
    home = tmp_path / "home"

    _write(
        project / "config.yaml",
        {
            "drive": {
                "shorts_publish_times": ["09:00"],
                "ost_title_crossfade_sec": 0.75,
                "montage_max_duration_sec": 15.0,
            }
        },
    )
    _write(
        home / ".config" / "gomoveshift" / "config.yaml",
        {"drive": {"shorts_publish_times": ["10:00"]}},
    )

    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load()
    assert cfg.drive.shorts_publish_times == ["10:00"]
    assert cfg.drive.ost_title_crossfade_sec == 0.75
    assert cfg.drive.montage_max_duration_sec == 15.0


def test_no_user_file_falls_back_to_project(tmp_path, monkeypatch):
    """Without a user file, project config.yaml still applies cleanly."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    home.mkdir()

    _write(project / "config.yaml", {"drive": {"shorts_publish_times": ["09:00"]}})

    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load()
    assert cfg.drive.shorts_publish_times == ["09:00"]


def test_no_project_file_uses_user_alone(tmp_path, monkeypatch):
    """Without a project file, user config alone is applied on top of defaults."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()

    _write(
        home / ".config" / "gomoveshift" / "config.yaml",
        {"drive": {"shorts_publish_times": ["10:00"]}},
    )

    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load()
    assert cfg.drive.shorts_publish_times == ["10:00"]


def test_lists_replace_wholesale_not_appended(tmp_path, monkeypatch):
    """Lists are replaced, not concatenated — list-merging has no clear semantics."""
    project = tmp_path / "project"
    home = tmp_path / "home"

    _write(project / "config.yaml", {"drive": {"shorts_publish_times": ["08:00", "09:00"]}})
    _write(
        home / ".config" / "gomoveshift" / "config.yaml",
        {"drive": {"shorts_publish_times": ["10:00"]}},
    )

    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load()
    assert cfg.drive.shorts_publish_times == ["10:00"]


def test_explicit_config_path_skips_user_merge(tmp_path, monkeypatch):
    """``Config.load(path)`` honors exactly ``path`` — no implicit user merge.

    Callers that specify an exact file are asserting they know what
    they want; don't surprise them by layering an override.
    """
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()

    explicit = project / "custom.yaml"
    _write(explicit, {"drive": {"shorts_publish_times": ["07:00"]}})
    _write(
        home / ".config" / "gomoveshift" / "config.yaml",
        {"drive": {"shorts_publish_times": ["10:00"]}},
    )

    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load(explicit)
    assert cfg.drive.shorts_publish_times == ["07:00"]


def test_empty_user_file_is_noop(tmp_path, monkeypatch):
    """An empty user config (possibly after yaml.safe_load → None) is a no-op."""
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()

    _write(project / "config.yaml", {"drive": {"shorts_publish_times": ["09:00"]}})
    user_file = home / ".config" / "gomoveshift" / "config.yaml"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text("")  # empty → safe_load returns None

    monkeypatch.chdir(project)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = Config.load()
    assert cfg.drive.shorts_publish_times == ["09:00"]
