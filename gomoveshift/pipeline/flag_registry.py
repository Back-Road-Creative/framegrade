"""Declarative CLI↔state flag-authority registry.

Every CLI flag that lands on ``PipelineState`` gets ONE row declaring (a) how the
CLI value is applied to a fresh state at build time and (b) who wins on resume.
:func:`apply_cli_to_state` and :func:`merge_on_resume` are the ONLY merge paths;
``drive_orchestrator`` must not hand-wire individual flags — each hand-wired flag
was a fresh chance for a one-directional bug.

Resume-authority semantics:

* ``SENTINEL_CLI_WINS`` — argparse-sentinel (default ``None``): an explicit CLI
  value (True OR False) beats saved state; an absent flag preserves saved state.
* ``ALWAYS_CLI`` — the current invocation's resolved value (including its
  coerced default) always wins on resume; the saved value never survives.
* ``STATE_WINS`` — resume-authoritative: persisted state
  beats a fresh CLI flag; a disagreement logs the "Resume CLI override
  conflict" warning. Carries a ``config.drive`` mirror kept in lockstep.
* ``NOT_PERSISTED`` — per-invocation only; never lands on ``PipelineState``.

The one-directional encodings (``ONE_DIRECTIONAL_TRUE``,
``STATE_WINS_SILENT``, ``Apply.IF_TRUTHY_TRUE``) no longer exist, so a
True-only write is not expressible. New persisted flags MUST be registered here (the
completeness test in ``tests/test_flag_registry.py`` fails otherwise) instead
of growing a new ad-hoc merge site.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Semantic(enum.Enum):
    SENTINEL_CLI_WINS = "sentinel-cli-wins"
    ALWAYS_CLI = "always-cli"
    STATE_WINS = "state-wins"
    NOT_PERSISTED = "not-persisted"


class Apply(enum.Enum):
    """How ``apply_cli_to_state`` writes the CLI value onto a fresh state."""

    IF_NOT_NONE = "if-not-none"  # sentinel: write only when CLI supplied it
    ALWAYS = "always"  # unconditional write of the resolved value
    CONSTRUCTOR = "constructor"  # applied by PipelineState(...) / pre-registry code


@dataclasses.dataclass(frozen=True)
class FlagRule:
    cli_dest: Optional[str]  # argparse dest on the drive subparser (None = derived)
    state_field: Optional[str]  # PipelineState field (None = NOT_PERSISTED)
    semantic: Semantic
    provenance_hint: str
    apply: Apply = Apply.IF_NOT_NONE
    coerce: Optional[Callable[[Any], Any]] = None  # value transform at apply time
    config_mirror: Optional[str] = None  # config.drive attr kept in lockstep
    merge_capture: bool = True  # participates in the resume capture/re-apply loop
    known_violation: Optional[str] = None  # migration marker; must stay empty
    unset: Optional[Callable[[Any], bool]] = None  # "CLI did not supply" test; default: is None


# fmt: off
FLAG_REGISTRY: tuple[FlagRule, ...] = (
    # ── run() constructor args (the former ``user_args`` restore block) ─────────────────────────
    FlagRule("directory", "source_dir", Semantic.ALWAYS_CLI, "constructor arg; never None at build", Apply.CONSTRUCTOR),
    FlagRule("output_dir", "output_base", Semantic.ALWAYS_CLI, "constructor arg; never None at build", Apply.CONSTRUCTOR),
    FlagRule("publish", "publish_day", Semantic.ALWAYS_CLI, "resolved pre-build (--publish > --lf-day > region default) so always non-None — always restored over saved", Apply.CONSTRUCTOR),
    FlagRule("region", "region", Semantic.SENTINEL_CLI_WINS, "argparse default None: absent flag preserves the saved region; run() resolves a still-None region to 'ireland' AFTER the merge", Apply.CONSTRUCTOR),
    FlagRule("location", "location", Semantic.SENTINEL_CLI_WINS, "parser default '' means 'not supplied': unset treats '' like None so a location-less resume preserves the saved location instead of stomping it; downstream str-typed readers never see None", Apply.CONSTRUCTOR, unset=lambda v: not v),
    FlagRule(None, "stages_base", Semantic.ALWAYS_CLI, "derived from output dir / GMS_INTERMEDIATE_DIR, not a CLI dest", Apply.CONSTRUCTOR),
    # ── schedule / per-invocation intent (the former ``schedule_overrides`` allowlist) ──────────
    FlagRule("start_date", "start_date", Semantic.SENTINEL_CLI_WINS, "first-LF publish date override"),
    FlagRule("lf_publish_date", "lf_publish_date", Semantic.SENTINEL_CLI_WINS, "rebase backstop: operator-authoritative publish anchor"),
    FlagRule("force_no_rebase", "force_no_rebase", Semantic.SENTINEL_CLI_WINS, "disables Phase D schedule rebase; sentinel so a CLI-absent resume preserves a saved True — readers treat None as False", coerce=bool),
    FlagRule("shorts_per_day", "shorts_per_day", Semantic.SENTINEL_CLI_WINS, "schedule slot count"),
    FlagRule("shorts_target", "shorts_target", Semantic.STATE_WINS, "resume-authoritative: persisted override beats a stale CLI flag; mirrored onto config for in-process planners", config_mirror="shorts_target_count"),
    FlagRule("shorts_gap", "shorts_gap", Semantic.STATE_WINS, "resume-authoritative: persisted override beats a stale CLI flag; mirrored onto config for in-process planners", config_mirror="shorts_min_gap_sec"),
    FlagRule("youtube_upload", "youtube_upload", Semantic.ALWAYS_CLI, "resolved by cmd_drive (--no-youtube-upload > --youtube-upload > --publish implies True) and coerced concrete pre-build — every invocation re-decides upload intent", Apply.ALWAYS),
    FlagRule("per_lf_locations", "per_lf_overrides", Semantic.SENTINEL_CLI_WINS, "operator per-LF labels; resume handled by preserve_per_lf_overrides_on_resume, which mirrors a non-empty CLI override onto every field in drive_state_model.PER_LF_OVERRIDE_MIRROR_FIELDS: non-empty CLI wins, empty/absent leaves saved", Apply.ALWAYS, coerce=lambda v: dict(v or {}), merge_capture=False),
    FlagRule("keep_intermediates", "keep_intermediates", Semantic.SENTINEL_CLI_WINS, "cleanup opt-out", coerce=bool),
    FlagRule("keep_masters", "keep_masters", Semantic.SENTINEL_CLI_WINS, "day-master retention opt-out", coerce=bool),
    FlagRule("force_sidecar_reset", "force_sidecar_reset", Semantic.SENTINEL_CLI_WINS, "operator sidecar reconciliation", coerce=bool),
    FlagRule("allow_seo_degraded", "allow_seo_degraded", Semantic.SENTINEL_CLI_WINS, "quality-gate escape hatch", coerce=bool),
    FlagRule("allow_short_count", "allow_short_count", Semantic.SENTINEL_CLI_WINS, "SHORTS-CONTRACT escape: accept a shorts shortfall and relax the master-drop gate", coerce=bool),
    FlagRule("skip_shorts_quality_gate", "skip_shorts_quality_gate", Semantic.SENTINEL_CLI_WINS, "--allow-bad-shorts: skips the Phase D 1080x1920/9:16 shorts dimension gate; sentinel so a CLI-absent resume preserves a saved opt-in (a False default would re-stomp it every resume)", coerce=bool),
    FlagRule("no_ledger_reconcile", "no_ledger_reconcile", Semantic.SENTINEL_CLI_WINS, "escape hatch: skip the startup YouTube↔ledger reconciliation (offline/quota-constrained runs)", coerce=bool),
    FlagRule("no_audio", "skip_audio", Semantic.SENTINEL_CLI_WINS, "operator near-silent LF: strips the day master's audio track (encoded -an) so every LF ships silent; sentinel so a CLI-absent resume preserves a saved True; config.drive mirror is the pipeline read path", coerce=bool, config_mirror="skip_audio"),
    FlagRule("sign_verify", "sign_verify", Semantic.SENTINEL_CLI_WINS, "road-sign GPS verification: enables the per-day sign OCR pass + the advisory per-LF comparison against the resolved location; sentinel so a CLI-absent resume preserves the operator's saved intent (a False default would re-stomp a saved True); config.drive mirror is the pipeline read path", coerce=bool, config_mirror="sign_verify"),
    FlagRule("sign_verify_strict", "sign_verify_strict", Semantic.SENTINEL_CLI_WINS, "escalates a corroborated sign-vs-GPS state/county conflict from a warning to a pre-encode hard fail; sentinel for the same resume reason as sign_verify; config.drive mirror is the pipeline read path", coerce=bool, config_mirror="sign_verify_strict"),
    FlagRule("superlong_enabled", "superlong_enabled", Semantic.SENTINEL_CLI_WINS, "opt-in per-day superlong part production (sl_NN.mp4); sentinel so a CLI-absent resume preserves a saved True across a multi-day run; config.drive mirror is the pipeline read path", coerce=bool, config_mirror="superlong_enabled"),
    FlagRule("superlong_publish_day", "superlong_publish_day", Semantic.SENTINEL_CLI_WINS, "per-run override for the superlong part publish weekday; sentinel so a CLI-absent resume preserves a saved value across a multi-day run; config.drive mirror is the pipeline read path (schedule_constants.resolve_superlong_publish_day)", config_mirror="superlong_publish_day"),
    FlagRule("ost_audit_confirmed", "ost_audit_confirmed", Semantic.SENTINEL_CLI_WINS, "OST audit gate bypass", coerce=bool),
    FlagRule("gate_cd_confirmed", "gate_cd_confirmed", Semantic.SENTINEL_CLI_WINS, "Gate C/D content-review bypass", coerce=bool),
    FlagRule("no_youtube_upload", "no_youtube_upload", Semantic.ALWAYS_CLI, "explicit YouTube opt-out (banner provenance); ALWAYS_CLI to mirror youtube_upload — cmd_drive folds it into resolve_youtube_upload, so both re-decide every invocation and a saved True is cleared by simply omitting the flag", Apply.ALWAYS, coerce=bool),
    # ── upload flags coerced concrete by _build_initial_state pre-registry ──────────────────────
    FlagRule("skip_upload", "skip_upload", Semantic.ALWAYS_CLI, "back-compat alias setting BOTH skips; coerced concrete pre-build so the current invocation always re-decides (--youtube-upload must clear a saved skip)", Apply.CONSTRUCTOR),
    FlagRule("skip_drive_upload", "skip_drive_upload", Semantic.ALWAYS_CLI, "split skip; composite bool(skip_drive_upload or skip_upload) pre-build", Apply.CONSTRUCTOR),
    FlagRule("skip_youtube_upload", "skip_youtube_upload", Semantic.ALWAYS_CLI, "split skip; composite bool(skip_youtube_upload or skip_upload) pre-build", Apply.CONSTRUCTOR),
    FlagRule("skip_csv", "skip_csv", Semantic.ALWAYS_CLI, "Metricool CSV opt-out; constructor arg, concrete bool — captured and re-applied on every resume so THIS invocation re-decides, like the skip-upload family", Apply.CONSTRUCTOR),
    FlagRule("fresh", "fresh", Semantic.ALWAYS_CLI, "consumed before the resume merge (--fresh purges state and early-returns), so the merged saved value is inert", Apply.CONSTRUCTOR, merge_capture=False),
    # ── per-invocation only — never persisted to PipelineState ──────────────────────────────────
    FlagRule("dry_run", None, Semantic.NOT_PERSISTED, "preflight plan print + exit"),
    FlagRule("force", None, Semantic.NOT_PERSISTED, "preflight prompt skip"),
    FlagRule("force_unlock", None, Semantic.NOT_PERSISTED, "run-lock escape hatch"),
    FlagRule("skip_invariants", None, Semantic.NOT_PERSISTED, "Layer-1 invariant break-glass (instance attr)"),
    FlagRule("list_lfs", None, Semantic.NOT_PERSISTED, "Phase A table print + exit"),
    FlagRule("label_dry_run", None, Semantic.NOT_PERSISTED, "label verification harness"),
    FlagRule("skip_stop_verify", None, Semantic.NOT_PERSISTED, "stop-removal verification gate bypass (config.drive mirror set in the drive handler)"),
    FlagRule("day_gpx", None, Semantic.NOT_PERSISTED, "day-GPX slicer input: writes <stem>.gpx sidecars before ingest; the sidecars, not the flag, are the persisted contract"),
    FlagRule("camera_utc_offset_hours", None, Semantic.NOT_PERSISTED, "day-GPX slicer camera clock offset; required with --day-gpx"),
    FlagRule("skip_blur", None, Semantic.NOT_PERSISTED, "blur-detection stage bypass (config.drive mirror set in the drive handler)"),
    FlagRule("insert_photos", None, Semantic.NOT_PERSISTED, "photo inserts: config.drive.insert_photos mirror set in the drive handler; each day's placement is recomputed (and hashed into the OST audit) every run"),
    FlagRule("insert_clip", None, Semantic.NOT_PERSISTED, "clip inserts: config.drive.insert_clips mirror set in the drive handler"),
)
# fmt: on


def apply_cli_to_state(state, cli_values: dict[str, Any], config=None) -> None:
    """Apply per-invocation CLI values onto a fresh ``PipelineState``.

    Replaces the hand-wired sentinel-write block in
    ``DrivePipelineOrchestrator._build_initial_state``. ``cli_values`` is keyed
    by ``state_field``; rows with ``Apply.CONSTRUCTOR`` (or absent keys) are
    skipped — those land via the ``PipelineState`` constructor or pre-registry
    coercions. ``config_mirror`` attributes are kept in lockstep on ``config``.
    """
    for rule in FLAG_REGISTRY:
        if rule.apply is Apply.CONSTRUCTOR or rule.state_field is None:
            continue
        if rule.state_field not in cli_values:
            continue
        val = cli_values[rule.state_field]
        if rule.apply is Apply.IF_NOT_NONE:
            if val is None:
                continue
        if rule.coerce is not None:
            val = rule.coerce(val)
        setattr(state, rule.state_field, val)
        if rule.config_mirror is not None and config is not None:
            setattr(config.drive, rule.config_mirror, val)


def merge_on_resume(state, saved: dict, config, log: Optional[logging.Logger] = None) -> None:
    """Merge a saved checkpoint into the fresh CLI-built ``state``, in place.

    Single merge path replacing ``_try_resume``'s hand-wired ``user_args`` +
    ``schedule_overrides`` allowlists and the resume-authoritative special case. Mechanism
    (behaviour-identical to the pre-registry code):

    1. Capture every registry row's supplied value from the fresh state (a
       row's ``unset`` predicate — default ``is None`` — means "operator did
       not supply it"; ``location`` also treats ``''`` as unset).
    2. WARN on resume-authoritative (``STATE_WINS``) conflicts: both a fresh CLI value and
       a persisted value present and disagreeing — state wins.
    3. Merge ``saved`` via ``PipelineState.from_dict`` coercions.
    4. Re-apply the captured CLI values so CLI wins — except ``STATE_WINS``
       rows whose key was persisted (resume-authoritative).
    5. Re-apply persisted ``STATE_WINS`` overrides onto their ``config.drive``
       mirrors for in-process planners that read config on resume-without-flag.
    """
    from .drive_state_model import PipelineState

    log = log or logger
    overrides: dict[str, Any] = {}
    for rule in FLAG_REGISTRY:
        if not rule.merge_capture or rule.state_field is None:
            continue
        val = getattr(state, rule.state_field, None)
        unset = rule.unset or (lambda v: v is None)
        if not unset(val):
            overrides[rule.state_field] = val
    for rule in FLAG_REGISTRY:
        if rule.semantic is not Semantic.STATE_WINS:
            continue
        cli_val = overrides.get(rule.state_field)
        saved_val = saved.get(rule.state_field)
        if cli_val is not None and saved_val is not None and cli_val != saved_val:
            log.warning(
                "  Resume CLI override conflict: --%s=%s passed on "
                "this invocation but checkpoint has %s=%s — state "
                "wins (resume-authoritative). To force the "
                "CLI value, delete pipeline_state.json's %s key or "
                "re-run with --fresh.",
                rule.state_field.replace("_", "-"),
                cli_val,
                rule.state_field,
                saved_val,
                rule.state_field,
            )
    # ``PipelineState.from_dict`` applies the int-key/Path/tuple/dataclass
    # coercions; copy each populated field across (matching the historical
    # ``object.__setattr__`` restore loops).
    coerced = PipelineState.from_dict(saved)
    for f in dataclasses.fields(coerced):
        val = getattr(coerced, f.name)
        if val is not None:
            object.__setattr__(state, f.name, val)
    state_wins_fields = {r.state_field for r in FLAG_REGISTRY if r.semantic is Semantic.STATE_WINS}
    for key, val in overrides.items():
        if key in state_wins_fields and saved.get(key) is not None:
            continue  # state persisted this override — honour state-wins.
        object.__setattr__(state, key, val)
    for rule in FLAG_REGISTRY:
        if rule.config_mirror is None:
            continue
        persisted = saved.get(rule.state_field)
        if persisted is not None:
            setattr(config.drive, rule.config_mirror, persisted)
