"""Prove ``HOME`` is blinded at conftest *import* time, before collection.

Background: ``gomoveshift.pipeline.longform_metadata`` reads ``Config.load()``
at IMPORT time (``DEFAULT_SEO_STORAGE = Path(_get_config().seo.storage_path)``),
and ``Config.load()`` deep-merges ``Path.home() / ".config" / "gomoveshift" /
"config.yaml"`` as its highest-precedence layer. The old
``isolate_home_from_operator_config`` fixture only ran at test *setup*, which
is always after pytest's collection phase has already imported every test
module and everything they import — so the import-time ``Config.load()``
call above saw the operator's REAL ``~/.config/gomoveshift/config.yaml``
during collection, before the fixture ever ran. On a workstation whose real
config held ``drive.publishing_route_tiktok: metricool`` under a branch that
rejected that value, ``Config.load()`` raised during collection, producing
``ERROR collecting tests/test_publishing_plan_io.py`` and a long run of
failing/erroring tests. Fixing this requires blinding ``HOME`` at
module-level import in ``conftest.py``, before ``import gomoveshift`` — a
fixture can never reach back into collection.

These tests capture state AT COLLECTION TIME (module level, before any test
function runs) to prove the fix actually took effect before import, not just
before the first test.
"""

import os
import pwd
from pathlib import Path

_HOME_AT_COLLECTION = Path.home()
_ENV_HOME_AT_COLLECTION = os.environ.get("HOME")

from gomoveshift.pipeline import longform_metadata  # noqa: E402


def test_home_is_fake_at_collection():
    """HOME must already be pointed at the fake dir before collection ran."""
    assert _HOME_AT_COLLECTION.name.startswith("gms-test-home-")
    assert _ENV_HOME_AT_COLLECTION == str(_HOME_AT_COLLECTION)


def test_home_at_collection_is_not_the_real_operator_home():
    """The collection-time HOME must be a disposable dir, not the real
    operator home — proving the blind took effect before any import, not
    merely before any test function. It is NOT under /tmp: product defaults
    expand ``~`` and are pinned to stay off tmpfs."""
    assert _HOME_AT_COLLECTION.name.startswith("gms-test-home-")
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert _HOME_AT_COLLECTION != real_home


def test_home_is_stable_between_collection_and_test_setup():
    """The fixture must point at the SAME dir the module-level block set,
    not a second, different fake dir — otherwise import-time and test-time
    code observe different HOMEs."""
    assert Path.home() == _HOME_AT_COLLECTION


def test_import_time_config_defaults_are_not_under_the_real_home():
    """``longform_metadata``'s import-time ``Config.load()`` reads must have
    resolved against the fake collection-time HOME, not the operator's real
    ``~/.config/gomoveshift/config.yaml`` — proving the leaked import-time
    read can no longer reach the real config."""
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert not str(longform_metadata.DEFAULT_SEO_STORAGE).startswith(str(real_home))
    assert not str(longform_metadata.DEFAULT_CREDENTIALS_DIR).startswith(str(real_home))
