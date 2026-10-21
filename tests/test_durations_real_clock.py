"""pytest's own timer must read the real clock while a test freezes time.

``--durations=25`` (pyproject addopts) is this suite's only per-test profile.
pytest 9 times every phase with ``_pytest.timing.Instant()``, which reads
``perf_counter``/``time`` imported into ``_pytest.timing``. freezegun 1.5's
default ignore list names ``_pytest.runner.`` but not ``_pytest.timing``, so a
test whose fixture enters ``freeze_time`` reported a setup "duration" of
~1.79e9 s. Staging run 38292433030 (suite 2946 s) listed 25 frozen-clock
tests as its slowest instead of the slow ones.
"""

import datetime as dt
import time

from _pytest import timing
from freezegun import freeze_time


def test_pytest_timer_ignores_a_frozen_clock() -> None:
    real_now = time.time()
    start = timing.Instant()
    with freeze_time(dt.datetime(2000, 1, 1, 12, 0)):
        took = start.elapsed().seconds
        stamp = timing.time()
    assert 0 <= took < 60, took
    assert abs(stamp - real_now) < 60, stamp
