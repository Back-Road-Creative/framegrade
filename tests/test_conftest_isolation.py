"""Prove the suite is blind to the operator's real ``~/.config/gomoveshift/``.

``Config.load()`` deep-merges ``Path.home() / ".config" / "gomoveshift" /
"config.yaml"`` as its highest-precedence layer (``gomoveshift/config.py``).
On a workstation that file holds live operational overrides (e.g. a
canary ``drive.publishing_route_youtube: first_party``). See
``tests/test_config_deep_merge.py`` for the merge contract itself. Without
suite-wide isolation, any test that calls ``Config.load()`` with no
explicit ``config_path`` picks up whatever the *operator* currently has
set, which is not a property of the code under test.

The ``isolate_home_from_operator_config`` fixture in ``conftest.py``
(session-scoped, autouse) points ``HOME`` at an empty temp directory for
the whole run so ``Config.load()`` can never see the real file. These
tests prove that guarantee holds even when a file exists at the exact
path the loader would look.

The operator's output root gets the same treatment: the
``isolate_output_root_from_operator`` fixture (autouse) points
``GMS_OUTPUT_DIR`` inside each test's own ``tmp_path``, so a
default-constructed ``TrackingStore`` can never write the live store.
"""

import os
from pathlib import Path

import yaml

from gomoveshift.config import Config
from gomoveshift.paths import gms_output_dir
from gomoveshift.tracking import TrackingStore


def test_home_is_the_disposable_test_home():
    """``Path.home()`` inside a test must be the disposable home the
    conftest created, never a real user's home.

    ``HOME`` is one of the things being isolated, so ``Path("~").expanduser()``
    is not an independent oracle here -- it reads the same env var. Assert
    against the conftest's own naming of the disposable directory instead,
    and that it is the same directory ``HOME`` names.
    """
    assert Path.home().name.startswith("gms-test-home-")
    assert Path(os.environ["HOME"]) == Path.home()


def test_operator_style_override_at_home_is_invisible_to_config_load():
    """A file written at the isolated HOME's config path must still be
    ignored if it isn't the one this test explicitly creates — i.e. the
    isolated HOME starts empty, so nothing an operator wrote to their
    real ``~/.config/gomoveshift/config.yaml`` can leak in.

    We write an operator-style override (``publishing_route_youtube:
    first_party``, distinct from the pydantic default ``off``) at the resolved ``Path.home()``,
    and confirm ``Config.load()`` picks it up from THIS isolated location —
    proving isolation didn't just skip reading user config altogether, it
    genuinely rehomed the lookup.
    """
    user_config = Path.home() / ".config" / "gomoveshift" / "config.yaml"
    user_config.parent.mkdir(parents=True, exist_ok=True)
    original = None
    if user_config.exists():
        original = user_config.read_text()
    try:
        user_config.write_text(
            yaml.safe_dump({"drive": {"publishing_route_youtube": "first_party"}})
        )
        cfg = Config.load()
        assert cfg.drive.publishing_route_youtube == "first_party"
    finally:
        if original is None:
            user_config.unlink(missing_ok=True)
        else:
            user_config.write_text(original)


def test_config_load_default_is_not_the_operators_route_override():
    """Absent any test-authored user config, ``Config.load()`` must resolve
    ``publishing_route_youtube`` to its pydantic default (``off``),
    never an operator's real ``first_party`` override — proving the isolated HOME
    starts empty on every test, not just this one.
    """
    cfg = Config.load()
    assert cfg.drive.publishing_route_youtube == "off"


def test_output_root_is_never_the_operators(tmp_path):
    """A default-constructed ``TrackingStore`` must root inside this test's
    temp dir, never the operator's output root.

    ``gms_output_dir()`` falls back to the output directory under the data
    home when ``GMS_OUTPUT_DIR`` is unset, and the suite may run as the
    operator. ``enqueue-plan --apply`` registers every enqueued short in the
    default store, so each ``--apply`` test with no override of its own
    wrote a ``proj`` project into the live store, which a scheduled metrics
    run then read as a real project.
    """
    assert gms_output_dir().is_relative_to(tmp_path)
    assert TrackingStore().root.is_relative_to(tmp_path)
