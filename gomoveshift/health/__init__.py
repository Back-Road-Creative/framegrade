"""Pipeline health detection: detect stuck or failed states, write incident rows,
push one line per new incident. Detection only -- nothing here launches, resumes or repairs.
See ``docs/pipeline/156-health-check.md``."""

from .check import run_check

__all__ = ["run_check"]
