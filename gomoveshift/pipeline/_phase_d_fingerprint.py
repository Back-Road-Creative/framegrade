"""Phase D resume fingerprint — invalidates stale ``phase_d`` checkpoint state.

Background:
the orchestrator's ``_try_resume`` short-circuits to ``SKIP phase_d`` whenever
``"phase_d"`` appears in ``state["phases_completed"]``. That branch ignores
per-invocation CLI flags that change Phase D outputs (``--lf-day``,
``--publish``, ``--start-date``, ``--shorts-target``, ``--shorts-per-day``,
``--per-lf-location``, ``--location``, ``--lf-time``, ``--shorts-times``,
``--lf-publish-date``) and ignores changes to the Phase D output generators
themselves. Result: silent stale outputs ("Drive pipeline
completed in 0.0s") and the operator forced to pass ``--phase D --force``.

This module provides three **pure** helpers used by the orchestrator's
resume path:

* :func:`compute_phase_d_fingerprint` — sha256 of canonicalised inputs.
* :func:`extract_cli_inputs_from_args_and_state` — build the inputs dict
  from an args-like object and a state mapping. No I/O.
* :func:`diff_fingerprint_inputs` — list of changed field names between
  two inputs dicts; used by the orchestrator's invalidation log line.

Silent staleness is unrepresentable
when the resume path always recomputes the fingerprint and compares.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _normalise_per_lf_locations(value: Any) -> dict:
    """Sort a per-LF-locations mapping into a stable, JSON-serialisable shape.

    Accepts ``None``, ``dict``, or any mapping. Keys are coerced to ``int``
    when possible (the in-memory shape uses ``int`` LF indices but JSON
    state files round-trip them as strings). Empty / missing → ``{}``.
    """
    if not value:
        return {}
    if isinstance(value, Mapping):
        normalised: dict[int, str] = {}
        for k, v in value.items():
            try:
                int_key = int(k)
            except (TypeError, ValueError):
                int_key = k  # type: ignore[assignment]
            normalised[int_key] = "" if v is None else str(v)
        return dict(sorted(normalised.items(), key=lambda kv: (str(kv[0]),)))
    return {}


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_str_list_or_none(value: Any) -> list[str] | None:
    """Coerce a shorts-times value into an order-preserving ``list[str]``.

    ``None`` → ``None`` (operator did not pass the flag). Order is NOT
    sorted: ``--shorts-per-day N`` selects the first N windows, so the
    sequence is significant and a re-ordering is a real schedule change.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def compute_phase_d_fingerprint(
    cli_inputs: Mapping[str, Any],
    code_version: int,
) -> str:
    """Return sha256 hex of canonicalised Phase D inputs + code version.

    Canonicalisation: sorted JSON keys, no whitespace beyond defaults,
    ``ensure_ascii=True``. The dict is augmented with ``code_version``
    (overwriting any caller-supplied entry) so callers can't bypass the
    version invalidation by omission.
    """
    payload = dict(cli_inputs)
    payload["code_version"] = int(code_version)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def extract_cli_inputs_from_args_and_state(
    args: Any,
    *,
    project_name: str,
) -> dict:
    """Build the canonical inputs dict from an args-like object + project name.

    ``args`` is any object exposing the relevant attributes (``argparse.Namespace``,
    ``types.SimpleNamespace``, a dict-like object, or the orchestrator's own
    ad-hoc context). Missing attrs default to ``None`` (treated as
    "operator did not pass this flag").

    ``project_name`` is required and keyword-only: it is the one field the
    orchestrator derives from the source path rather than from CLI flags, so
    the caller — which owns the state — must hand it over explicitly. It is
    deliberately NOT read off a state argument: ``PipelineState`` is a
    dataclass without the attribute, so a sniffing lookup would silently
    fingerprint an empty project and mis-decide the Phase D resume.

    Returned dict shape (exactly these keys, sorted by JSON dump):

    * ``lf_day`` — resolved weekday string or ``None``
    * ``publish`` — resolved weekday string
    * ``start_date`` — ``YYYY-MM-DD`` or ``None``
    * ``shorts_target`` — int or ``None``
    * ``shorts_per_day`` — int or ``None``
    * ``per_lf_locations`` — sorted ``{int: str}`` mapping
    * ``location`` — drive-wide location string
    * ``lf_time`` — long-form publish time (``HH:MM``) or ``None``
    * ``shorts_times`` — ordered list of shorts publish times or ``None``
    * ``lf_publish_date`` — single-LF publish anchor (``YYYY-MM-DD``) or ``None``
    * ``project_name`` — the caller-supplied name, verbatim

    Note: ``code_version`` is NOT included here. It is added by
    :func:`compute_phase_d_fingerprint` from its dedicated parameter so
    that the version constant has exactly one entry point.
    """

    def _attr(name: str) -> Any:
        if isinstance(args, Mapping):
            return args.get(name)
        return getattr(args, name, None)

    # ``lf_day`` accepts either a CLI-style string ("mon"/"tue"/...) OR
    # a normalised list-of-int weekdays (the orchestrator passes the
    # latter, since ``--lf-day`` reaches it via
    # ``config.drive.long_form_publish_days``). Pass through both shapes
    # untouched so the JSON canonicalisation distinguishes ``"mon"`` from
    # ``[0]`` and a re-ordered list from a stable sorted one.
    raw_lf_day = _attr("lf_day")
    if isinstance(raw_lf_day, (list, tuple)):
        lf_day_value: Any = list(raw_lf_day)
    else:
        lf_day_value = _coerce_str_or_none(raw_lf_day)
    return {
        "lf_day": lf_day_value,
        "publish": _coerce_str_or_none(_attr("publish")),
        "start_date": _coerce_str_or_none(_attr("start_date")),
        "shorts_target": _coerce_int_or_none(_attr("shorts_target")),
        "shorts_per_day": _coerce_int_or_none(_attr("shorts_per_day")),
        "per_lf_locations": _normalise_per_lf_locations(_attr("per_lf_locations")),
        "location": "" if _attr("location") is None else str(_attr("location")),
        "lf_time": _coerce_str_or_none(_attr("lf_time")),
        "shorts_times": _coerce_str_list_or_none(_attr("shorts_times")),
        "lf_publish_date": _coerce_str_or_none(_attr("lf_publish_date")),
        "project_name": str(project_name),
    }


def diff_fingerprint_inputs(
    old_inputs: Mapping[str, Any] | None,
    new_inputs: Mapping[str, Any],
) -> list[str]:
    """Return sorted list of field names whose values differ.

    A missing ``old_inputs`` (``None`` or empty) yields every field present
    in ``new_inputs`` — surfaces the "no prior fingerprint" backward-compat
    case as "everything changed", which is the honest signal.

    ``code_version`` is included in the diff when present in either side.
    """
    old = dict(old_inputs or {})
    new = dict(new_inputs)
    keys = set(old.keys()) | set(new.keys())
    changed: list[str] = []
    for k in sorted(keys):
        if old.get(k) != new.get(k):
            changed.append(k)
    return changed
