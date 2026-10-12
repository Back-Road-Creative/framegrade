"""``python -m gomoveshift`` entry point.

The console launches every job as ``[sys.executable, "-m", "gomoveshift", ...]``
(``console/jobs.py``, ``console/drives.py``, ``console/journeys.py``). Without
this module Python refuses with "No module named gomoveshift.__main__", so
every console-launched job would fail at exec.
"""

from gomoveshift.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
