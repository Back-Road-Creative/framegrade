"""``schedule_constants.apply_config`` pushes loaded config into module attrs.

Regression guard for a preview crash: a user override in
``~/.config/gomoveshift/config.yaml`` loaded into ``Config`` correctly
(``days=frozenset({4})`` via the accessor) but every downstream
lazy-import reader — state_invariants, quality_gates, scheduler —
still saw ``LONG_FORM_DAYS=frozenset({2})`` (the DriveConfig default
snapshot at module import time). Result: ``StateInvariantError:
publish_day 'fri' ... not in LONG_FORM_DAYS=[2]`` despite the override.

Fix contract: the CLI calls ``apply_config(config)`` right after
``get_config()``. Every lazy-import reader then sees the effective
values.

Tests use ``monkeypatch.setattr`` to stage the pre-apply state and rely
on monkeypatch's built-in restore to keep the module attrs clean for
adjacent test files (``test_schedule_constants.py`` asserts defaults).
"""

import pytest

from gomoveshift.pipeline import schedule_constants

from _state_helpers import make_state


def test_apply_config_overwrites_module_days(monkeypatch):
    """After ``apply_config``, the module attr matches the config override."""
    from datetime import time as dt_time

    from gomoveshift.config import Config

    # Stage the pre-apply state via monkeypatch so teardown restores
    # cleanly even if the test asserts mid-way.
    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({2}))
    monkeypatch.setattr(schedule_constants, "LONG_FORM_TIME", dt_time(11, 0, 0))

    cfg = Config()
    cfg.drive.long_form_publish_days = [4]  # Fri
    cfg.drive.long_form_publish_time = "09:30"

    schedule_constants.apply_config(cfg)

    assert schedule_constants.LONG_FORM_DAYS == frozenset({4})
    assert schedule_constants.LONG_FORM_TIME == dt_time(9, 30, 0)


def test_apply_config_is_observed_by_state_invariants(monkeypatch):
    """state_invariants reads the module attr — apply_config must update what it sees.

    The failure mode: state carries ``publish_day="fri"``,
    ``_check_publish_day`` reads module ``LONG_FORM_DAYS``, finds {2},
    raises. After ``apply_config({4})`` the same check must pass.
    """
    from datetime import time as dt_time

    from gomoveshift.config import Config
    from gomoveshift.pipeline.exceptions import StateInvariantError
    from gomoveshift.pipeline.state_invariants import assert_state_invariants

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({2}))
    monkeypatch.setattr(schedule_constants, "LONG_FORM_TIME", dt_time(11, 0, 0))

    cfg = Config()
    cfg.drive.long_form_publish_days = [4]
    schedule_constants.apply_config(cfg)

    state = make_state(publish_day="fri")
    assert_state_invariants(state)  # must not raise — the point of the fix

    # Sanity-inverse: flip back to Wed, re-apply, confirm the same
    # state NOW fails (so we know the invariant reads the post-apply
    # value, not a cached copy from earlier).
    cfg.drive.long_form_publish_days = [2]
    schedule_constants.apply_config(cfg)
    with pytest.raises(StateInvariantError):
        assert_state_invariants(state)


def test_apply_config_is_idempotent(monkeypatch):
    """Calling ``apply_config`` twice with the same config is a no-op."""
    from datetime import time as dt_time

    from gomoveshift.config import Config

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({2}))
    monkeypatch.setattr(schedule_constants, "LONG_FORM_TIME", dt_time(11, 0, 0))

    cfg = Config()
    cfg.drive.long_form_publish_days = [4]
    schedule_constants.apply_config(cfg)
    first = schedule_constants.LONG_FORM_DAYS
    schedule_constants.apply_config(cfg)
    assert schedule_constants.LONG_FORM_DAYS == first
    assert schedule_constants.LONG_FORM_DAYS == frozenset({4})


def test_monkeypatch_still_wins_after_apply_config(monkeypatch):
    """Subsequent ``monkeypatch.setattr`` wins over ``apply_config``.

    Confirms there's no shadow state — tests that monkeypatch the
    module attr after a CLI startup sequence still observe their
    override.
    """
    from datetime import time as dt_time

    from gomoveshift.config import Config

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({2}))
    monkeypatch.setattr(schedule_constants, "LONG_FORM_TIME", dt_time(11, 0, 0))

    cfg = Config()
    cfg.drive.long_form_publish_days = [4]
    schedule_constants.apply_config(cfg)

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({5}))
    assert schedule_constants.LONG_FORM_DAYS == frozenset({5})
