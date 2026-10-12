"""Pipeline-wide exceptions."""

from pathlib import Path


class StateInvariantError(ValueError):
    """Raised when the drive pipeline state fails Layer-1 invariants.

    Checked in ``gomoveshift.pipeline.state_invariants.assert_state_invariants``
    from inside ``save_drive_state`` BEFORE JSON serialization. A failure
    means we refuse to write the state file — the last-valid state on disk
    is left untouched so a subsequent resume doesn't pick up corrupt data.

    Operators handle this by:

    1. Reading the message (always starts with a stable grep-able prefix,
       see ``state_invariants.STATE_INVARIANT_ERROR_PREFIX``).
    2. Inspecting / editing the project's ``_stages/pipeline_state.json``
       to fix the flagged field manually.
    3. Re-running the pipeline. The write that previously raised should
       now pass invariants.

    Subclasses ``ValueError`` so callers who want the broad category can
    use ``except ValueError``.
    """


class PhaseBCExitContractError(StateInvariantError):
    """Raised at Phase BC exit when per-LF state fails the aggregate boundary
    contract — i.e. one or more required ``per_lf_*`` maps lack entries for
    LF indices that the phase rendered.

    Wraps the per-field assertion family (``_assert_per_lf_trim_ratio_complete``,
    ``_assert_per_lf_distance_complete``, etc.) into a single named error
    listing ALL violations across ALL fields in one shot — operators fix
    them together rather than chasing one ``RuntimeError`` per resume.
    """


class PhaseDEntryContractError(StateInvariantError):
    """Raised at Phase D entry when persisted state is missing per-LF maps
    that Phase D will consume. Symmetric counterpart to
    :class:`PhaseBCExitContractError`: covers the resume-from-corrupted-state
    case where BC's exit aggregator would have raised, but the state file on
    disk was written by an older revision and bypassed that gate.
    """


class StateCoherenceError(RuntimeError):
    """Raised at orchestrator startup when persisted state is internally inconsistent.

    Distinct from :class:`StateInvariantError` (Layer 1, write-time type/shape):
    this is a Layer-1.5 cross-field check that runs ONCE on resume, AFTER the
    checkpoint has been loaded into memory but BEFORE the phase loop commits
    work. It catches drift that a single-field type/shape check can't see —
    e.g. ``phases_completed`` claims ``phase_bc_day_first`` is done but
    ``per_day_completed`` doesn't cover the dates the orchestrator is about
    to process.

    The failure this prevents: the orchestrator
    skipped Phase B+C on resume because the phase was marked complete, even
    though only some of the days had been encoded. Phase D then ran on an
    incomplete batch.

    Recovery is INTENTIONALLY operator-driven, not auto-correcting: silently
    fixing the spurious phase entry would mask the upstream bug that wrote it.
    The exception message names the spurious phase and the missing dates
    so the operator can either remove the entry from the state file or wipe
    the checkpoint to force a clean resume.

    Subclasses ``RuntimeError`` (not ``ValueError``) to distinguish from the
    write-time invariant family — operators trapping ``StateInvariantError``
    should NOT silently swallow a coherence failure.
    """


class PipelineValidationError(RuntimeError):
    """Raised when the DoD validator rejects the pipeline output.

    Carries the full validation report as the message so failure traces
    include check numbers and diagnostics without requiring separate logs.
    """


class CorruptStateError(RuntimeError):
    """Raised when ``pipeline_state.json`` is unreadable/unrecoverable.

    Previous behavior returned ``None`` on JSONDecodeError (after
    ``.tmp`` sibling recovery failed) and logged a WARNING. An operator
    tailing stdout could miss the warning and lose hours of prior work
    to an unintended fresh run. This exception forces the CLI to surface
    the failure loudly (non-zero exit + banner) and require an explicit
    ``--fresh`` re-invocation to proceed.

    Carries the offending state-file path so the operator can inspect or
    repair it manually.
    """

    def __init__(self, state_file, message: str = ""):
        self.state_file = state_file
        detail = f"Corrupt pipeline state file: {state_file}"
        if message:
            detail = f"{detail} — {message}"
        super().__init__(detail)


class PhaseQualityError(ValueError):
    """Raised when Phase D's end-of-phase quality gate finds semantic
    violations in the state it's about to commit.

    Phase D produces many independent artifacts (LF metadata JSONs,
    drive_schedule, consolidated Metricool CSV, per-LF CSVs,
    thumbnails). Each one can pass its own in-phase check yet leave
    the batch semantically broken — concrete failures seen:

      * Every LF title identical (drive-wide ``--location`` leaked
        into every per-LF slot)
      * Metricool CSV's First Comment Text contained bare 11-char
        video IDs (phantom entries that survived a stale sidecar)
      * ``publish_day='sat'`` on a Wed-only channel (silent schedule
        drift after a resume)

    The gate at end-of-Phase-D accumulates EVERY violation it finds
    (not fail-on-first) and raises this exception once with the full
    list so operators fix all of them in one round. Subclasses
    ``ValueError`` so callers that trap ``ValueError`` catch it, and
    distinguishes from ``StateInvariantError`` (Layer 1, write-time
    type/shape) and ``PipelineValidationError`` (Layer 3, DoD
    rearguard).

    Escape hatch: ``--allow-seo-degraded`` (CLI → orchestrator →
    ``state["allow_seo_degraded"]``) downgrades Rule 2 (no
    ``seo_degraded=True`` LFs) to a warning-only check. Every other
    rule remains a hard fail — if SEO manager is unavailable AND a
    title is still mis-derived, the other rules still catch it.

    See ``gomoveshift/pipeline/quality_gates.py`` for the full rule list.
    """


class YouTubeQuotaError(RuntimeError):
    """Raised when a YouTube Data API HTTP 403 carries a quota-class
    ``reason`` — distinct from a permanent auth failure.

    The YouTube Data API v3 surfaces both quota exhaustion and
    permission/auth failures as HTTP 403. The two have different
    recovery profiles:

      * ``quotaExceeded`` / ``dailyLimitExceeded`` — the daily project
        quota is spent. Recovery: wait until midnight Pacific Time
        (the YouTube Data API quota resets 00:00 PT per the API
        quotas dashboard) and resume. The in-progress batch is safe
        to resume because ``upload_if_needed`` skips LFs already
        recorded in ``state["youtube_uploads"]``.
      * ``userRateLimitExceeded`` — a short-window rate limit
        (usually per-user-per-second). Recovery: back off a few
        seconds and retry. We still surface this as a
        ``YouTubeQuotaError`` so the operator sees it in the same
        banner and can make the call on whether to retry now or wait.
      * Any other 403 (``forbidden``, ``insufficientPermissions``,
        etc.) — permanent auth failure requiring credential rotation,
        NOT this exception. Those keep the original HttpError path
        unchanged.

    The exception carries:

      * ``reason`` — the raw string from
        ``HttpError.error_details[*]["reason"]`` (``quotaExceeded``,
        ``userRateLimitExceeded``, ``dailyLimitExceeded``).
      * ``status_code`` — always 403; kept as an attribute for
        forward-compat if the API ever emits 429 on the same class.
      * ``http_error`` — the original ``HttpError`` so callers with
        richer context needs (telemetry, log extracts) don't have to
        re-parse.

    Recovery guidance is baked into ``__str__`` so a bare ``logger.error``
    or raised-stack traceback surfaces the operator instructions. See
    ``drive_phases/youtube_uploader._execute_resumable`` for the raise
    site and ``drive_phases/phase_d.py`` for the Phase D banner /
    clean-exit handling — saved state checkpoint + propagate so the run
    exits cleanly for resume after the quota window resets.

    Subclasses ``RuntimeError`` (not the ``googleapiclient`` ``HttpError``
    — we want callers to trap quota distinctly from all other HTTP paths)
    and specifically NOT ``StateInvariantError`` / ``PipelineValidationError``
    — those carry different operator guidance.
    """

    def __init__(
        self,
        reason: str,
        message: str = "",
        status_code: int = 403,
        http_error=None,
    ):
        self.reason = reason
        self.status_code = status_code
        self.http_error = http_error
        # Build a self-contained, operator-friendly message. This is the
        # banner the YouTube batch loop falls back to when there's no
        # richer handler — must stand alone as actionable guidance.
        hint = ""
        if reason in {"quotaExceeded", "dailyLimitExceeded"}:
            hint = (
                " — daily YouTube Data API quota exhausted; "
                "retry after midnight Pacific Time when the daily quota window "
                "resets. Already-uploaded LFs are safe; "
                "``upload_if_needed`` skips them on resume."
            )
        elif reason == "userRateLimitExceeded":
            hint = (
                " — YouTube Data API per-user rate limit hit; "
                "retry after a short backoff. Already-uploaded LFs are "
                "safe; ``upload_if_needed`` skips them on resume."
            )
        detail = f"YouTube quota error (reason={reason!r}, HTTP {status_code}){hint}"
        if message:
            detail = f"{detail}: {message}"
        super().__init__(detail)


class ProbeFailedError(RuntimeError):
    """Raised when ffprobe cannot return usable video metadata.

    Replaces the legacy hardcoded fallback (``width=3840, height=2160,
    duration=300.0``) inside
    :func:`gomoveshift.pipeline.ost_applier._probe_video_info`. The old
    behaviour looked correct on a fresh 4K input but silently accepted
    truncated / half-written outputs on resume, fabricating a duration
    and dimensions that downstream encode and validation steps treated
    as ground truth. Probing must report what is on disk, never
    invented defaults.

    Callers must decide explicitly:

      * **Hard fail / abort** — let the exception propagate (or trap and
        return a clear failure to the orchestrator). This is the right
        choice for any path where the probe value drives encoder
        arguments or downstream "is this artifact complete" checks.
      * **Soft fallback** — trap at the call site and substitute an
        explicit, named default. Required only when the caller has
        independent evidence that the file is intact and ffprobe is the
        problem (e.g. an isolated diagnostic tool).

    Carries the offending video path so the operator can inspect or
    repair it manually.
    """

    def __init__(self, video_path, message: str = ""):
        self.video_path = video_path
        detail = f"ffprobe failed for {video_path}"
        if message:
            detail = f"{detail} — {message}"
        super().__init__(detail)


class SidecarIntegrityError(ValueError):
    """Raised when ``youtube_uploads.json`` and the current drive_plan
    disagree on LF slot count / indices.

    The sidecar at ``<output_base>/youtube_uploads.json`` is a local
    record of prior upload attempts — NOT a live mirror of YouTube. It
    can drift from reality when:

      * an operator deletes drafts via YouTube Studio;
      * a prior run aborted after writing sidecar entries;
      * a plan was re-generated with a different LF count.

    When the drift is silent, Phase D treats stale entries as
    authoritative and the Metricool CSV's First Comment Text ends up
    with dead video IDs (phantom entries baked into a live CSV).

    This exception is raised at Phase D entry by
    :func:`gomoveshift.pipeline.youtube_sidecar.assert_sidecar_integrity`.
    It carries:

      * ``sidecar_path``: absolute path to the offending JSON file
      * ``phantom_indices``: LF indices in sidecar but NOT in drive_plan
      * ``missing_indices``: LF indices in drive_plan but NOT in sidecar

    The operator then decides whether to:

      1. wipe the sidecar and re-upload (``--force-sidecar-reset``), or
      2. reconcile manually (edit the file or delete stale YouTube drafts).

    API-level reconciliation against YouTube is intentionally NOT
    attempted here — the gate is strictly local state/plan agreement.
    The YouTube sidecar is not authoritative.

    Subclasses ``ValueError`` so callers that already trap generic
    value-errors still see it; cast to ``SidecarIntegrityError`` for
    operator-friendly messaging.
    """

    def __init__(
        self,
        sidecar_path,
        phantom_indices,
        missing_indices,
        message: str = "",
    ):
        self.sidecar_path = sidecar_path
        self.phantom_indices = list(phantom_indices)
        self.missing_indices = list(missing_indices)
        # Name whichever sidecar actually failed (``youtube_uploads.json``
        # or ``superlong_uploads.json``) instead of hardcoding the LF one.
        filename = Path(sidecar_path).name or "youtube_uploads.json"
        detail = f"{filename} integrity check failed at {sidecar_path}"
        if message:
            detail = f"{detail} — {message}"
        super().__init__(detail)


class WeatherMeasurementError(RuntimeError):
    """Raised when ``WeatherDetector`` returns degraded data (fewer than
    ``min_success_ratio`` of brightness chunks succeeded) and the
    operator did not opt into the degraded fallback path.

    The unified scorer once silently substituted
    ``DEFAULT_WEATHER_CLASS = "overcast"`` for any failed measurement,
    baking the wrong class into ``drive_scores.npz``. On a slow
    cross-mount source every chunk can time out, which would silently grade
    the whole run as overcast.

    Robust-by-construction fix: raise by default. Operator either
    fixes the underlying cause (copy source to ext4) or passes
    ``--allow-degraded-weather`` to explicitly accept the fallback.
    """
