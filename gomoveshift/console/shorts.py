"""Console Shorts tab helpers.

Shorts have no standalone CLI subcommand -- grep ``gomoveshift/cli/__init__.py``'s
``_COMMAND_MAP``: only ``drive`` (and the legacy single-video ``pipeline``)
produce shorts, writing the shorts Metricool CSV as one of their outputs
(``drive_phases/phase_d.py``: ``<project>_shorts_metricool.csv``). So
``build_shorts_argv`` launches the exact same argv as the Drives tab --
the console never invents a second launch path -- and this module's other
job is read-only access to the CSV a run already wrote, never regenerating it.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .drives import build_drive_argv

#: Matches ``drive_phases/phase_d.py``'s ``csv_path`` naming.
_SHORTS_CSV_GLOB = "*_shorts_metricool.csv"


def build_shorts_argv(
    directory: str,
    *,
    output_dir: str,
    region: "str | None" = None,
    dry_run: bool = True,
    confirm_plan: bool = False,
    skip_upload: bool = False,
    ost_system_review: bool = False,
    publish: "str | None" = None,
    lf_day: "str | None" = None,
    shorts_target: "int | None" = None,
    shorts_gap: "float | None" = None,
) -> list[str]:
    """Argv for a shorts-producing console launch -- identical to the
    Drives-tab argv (shorts have no separate CLI entry point). ``publish``/
    ``lf_day`` are forwarded to :func:`build_drive_argv` verbatim, which
    pairs and validates them -- a shorts launch is a real ``drive`` run and
    is just as schedule-critical as one from the Drives tab.
    ``shorts_target``/``shorts_gap`` are the only shorts-specific knobs
    ``drive`` exposes; forwarded verbatim too. Full-run safety choices use
    the same drive argv builder as the Drives tab."""
    return build_drive_argv(
        directory,
        output_dir=output_dir,
        region=region,
        dry_run=dry_run,
        confirm_plan=confirm_plan,
        skip_upload=skip_upload,
        ost_system_review=ost_system_review,
        publish=publish,
        lf_day=lf_day,
        shorts_target=shorts_target,
        shorts_gap=shorts_gap,
    )


def find_shorts_csv(output_dir: "str | Path") -> "Path | None":
    """Return the run's shorts Metricool CSV under ``output_dir``, or
    ``None`` if the run hasn't written one yet."""
    matches = sorted(Path(output_dir).glob(_SHORTS_CSV_GLOB))
    return matches[0] if matches else None


def read_shorts_csv_rows(csv_path: "str | Path") -> "tuple[list[str], list[list[str]]]":
    """Parse the shorts CSV for the preview table -- read-only, never
    regenerated; the download route serves the same file's raw bytes."""
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]
