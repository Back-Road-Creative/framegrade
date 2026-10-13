"""Phase D validation gates.

Extracted from ``phase_d.py`` — the three conceptually
unified phase-D entry/exit gates that fire around the upload sequence:
sidecar integrity (pre-upload), final per-LF validation + audio-loudness
hard fail (post-encode), and the 1080x1920 shorts quality gate.
Pure cut-and-move; no semantic changes.

Logger is pinned to ``gomoveshift.pipeline.drive_phases.phase_d`` so
existing ``caplog``-anchored tests keep matching the records emitted
from these helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...config import Config
from ..media_api import deliverables_for_publish

logger = logging.getLogger("gomoveshift.pipeline.drive_phases.phase_d")


def assert_video_quality_hard_fail(video_quality_failures: list[str]) -> None:
    """Hard-fail Phase D when any LF's FINAL.mp4 is off-spec on codec,
    resolution, bitrate floor, profile, or pixel format.

    ``video_quality`` and ``bitrate_preservation`` were the last WARN-only
    gates on the drive path: a regressed 8-bit or under-floor
    long-form logged a warning and shipped, which is exactly the banded-skies
    failure class. Both compare the final against FROZEN
    ``encoding_constants`` values — deterministic, no judgment — so a failure
    means the encode is genuinely off-contract and the run aborts before any
    Drive/YouTube upload (its caller ``_run_final_validation_gate`` runs ahead
    of the upload block in ``PhaseD.run``), mirroring
    ``assert_audio_loudness_hard_fail``.
    """
    if not video_quality_failures:
        return
    raise RuntimeError(
        "Video quality HARD FAIL — at least one LF's FINAL.mp4 is off-spec "
        "(codec, resolution, bitrate floor, profile, or pixel format) against "
        "the frozen encoding constants. Re-encode the affected LF; do NOT "
        "publish.\nAffected LFs:\n" + "\n".join(video_quality_failures)
    )


def _default_perceptual_check(lf_path: Path, duration: float | None):
    """Real ffmpeg-backed spot-check; built here so the gate stays injectable
    without the ``PipelineValidator`` facade growing an unused passthrough."""
    import subprocess

    from ..validation.ffprobe_client import FFprobeClient
    from ..validation.longform_validator import LongformValidator

    return LongformValidator(FFprobeClient(subprocess)).check_perceptual_spot(
        lf_path, duration=duration
    )


def assert_perceptual_spot_hard_fail(perceptual_failures: list[str]) -> None:
    """Hard-fail Phase D when a final has no picture at all.

    Nothing looked at a pixel before this — every pre-upload gate read
    container/stream metadata, so an LF that encoded to a black picture
    satisfied all of them and uploaded. ``check_perceptual_spot`` only fails on
    an all-black final (a render fault, never footage). Raises ahead of any
    upload, mirroring ``assert_audio_loudness_hard_fail``.
    """
    if not perceptual_failures:
        return
    raise RuntimeError(
        "Perceptual spot-check HARD FAIL — at least one LF's FINAL.mp4 has no "
        "picture in any sampled window (black render). Re-encode the affected "
        "LF; do NOT publish.\nAffected LFs:\n" + "\n".join(perceptual_failures)
    )


def _run_sidecar_integrity_gate(
    *, state, drive_plan, output_base: Path, force_sidecar_reset: bool
) -> None:
    """Assert ``youtube_uploads.json`` / ``superlong_uploads.json`` sidecars
    agree with ``drive_plan`` / ``state.superlong_parts`` respectively.

    Strictly LOCAL — no YouTube API
    reconciliation. Wipes BOTH sidecars + the LF state mirror first when
    ``--force-sidecar-reset`` was passed. No-op only when there are
    neither regular LFs nor superlong parts to check.
    """
    from ..youtube_sidecar import (
        assert_sidecar_integrity,
        assert_superlong_sidecar_integrity,
        iter_superlong_parts,
        save_sidecar,
        save_superlong_sidecar,
    )

    has_regular_lfs = bool(drive_plan and drive_plan.regular_lfs)
    has_superlong = bool(iter_superlong_parts(state))
    if not has_regular_lfs and not has_superlong:
        return

    if force_sidecar_reset:
        logger.warning(
            "Phase D: --force-sidecar-reset passed; wiping "
            "youtube_uploads.json and superlong_uploads.json at %s "
            "before integrity check",
            output_base,
        )
        save_sidecar(output_base, {})
        state.youtube_uploads = {}
        save_superlong_sidecar(output_base, {})
    if has_regular_lfs:
        assert_sidecar_integrity(output_base, drive_plan)
    if has_superlong:
        assert_superlong_sidecar_integrity(output_base, state)


def _default_picture_check(lf_path: Path, duration: float | None, source_sharpness, changed):
    """Real picture gate (post-rendition softness / ringing); injectable like the spot-check."""
    from ..validation.longform_validator import picture_quality_gate

    return picture_quality_gate(
        lf_path, duration or 0.0, source_sharpness=source_sharpness, changed=changed
    )


def _lf_picture_inputs(state) -> dict:
    """``{lf_index: (source_sharpness, picture_changed)}`` for the picture gate."""
    from ..quality import picture_inputs_from_state

    sharpness, changed = picture_inputs_from_state(state)
    return {idx: (sharpness, flag) for idx, flag in changed.items()}


def _lf_day_records(state) -> dict:
    """``{lf_index: day master_record}`` so the 4K gate sees each LF's upscale facts."""
    records: dict = {}
    for art in getattr(state, "day_artifacts", None) or []:
        record = getattr(art, "master_record", None)
        if isinstance(record, dict):
            for idx in getattr(art, "regular_lf_indices", None) or []:
                records[idx] = record
    return records


def _run_final_validation_gate(
    *,
    validator_cls,
    config: Config,
    drive_plan,
    regular_lf_paths,
    metadata_per_lf: dict,
    perceptual_check_fn=None,
    picture_inputs: dict | None = None,
    picture_check_fn=None,
    day_records: dict | None = None,
    claim_native_resolution_only: bool = True,
) -> None:
    """Run per-LF post-encode validation + audio-loudness HARD FAIL.

    Sub-gates raise ``RuntimeError`` rather than warn: ``audio_loudness``
    (LOUDER than the near-silent integrated ceiling, or breaching the true-peak
    cap), ``audio_momentary`` (the
    speech-intelligibility belt: loudest 400 ms window above the momentary
    ceiling),
    ``location_invariant`` (far-road desync), ``file_integrity`` (corrupt /
    truncated / unreadable FINAL.mp4), ``content_safety`` (blocklist
    violation), and ``metadata_quality`` (title/location mismatch,
    business-name-in-title, fragmented location tags — previously
    WARN-only, which let a title whose location diverged from its own
    description ship). ``content_safety`` is a JUDGMENT-call escalation
    (blocklist could false-match a place name); the rest are deterministic.
    ``perceptual_spot`` is the only gate that looks at pixels rather
    than metadata; it hard-fails only when a final has NO picture in any
    sampled window — a render fault, never footage.
    ``video_quality`` / ``bitrate_preservation`` escalate too — both
    compare the final against frozen ``encoding_constants``, so a warning there
    meant a regressed 8-bit or under-floor long-form shipped anyway. The
    bitrate gate needs no switching on: its floor is absolute, so ``validate``
    runs it on every call.
    A missing or under-1-MiB FINAL.mp4 is itself a ``file_integrity`` HARD FAIL
    — recorded and raised below rather than silently skipped, so a
    truncated LF can't sail past the pre-upload gates. Escape hatch:
    ``config.drive.skip_audio``.
    """
    from . import phase_d as _pd

    skip_audio = config.drive.skip_audio
    gate_validator = validator_cls(
        max_lufs=config.output.audio_max_lufs,
        max_true_peak=config.output.audio_target_tp,
    )
    audio_failures: list[str] = []
    location_failures: list[str] = []
    integrity_failures: list[str] = []
    content_failures: list[str] = []
    metadata_failures: list[str] = []
    video_quality_failures: list[str] = []
    perceptual_failures: list[str] = []
    perceptual_check = perceptual_check_fn or _default_perceptual_check
    picture_check = picture_check_fn or _default_picture_check
    for lf, lf_path in zip(drive_plan.regular_lfs, regular_lf_paths):
        lf_path = Path(lf_path)
        # A missing or truncated (<1 MiB) FINAL.mp4 is exactly the
        # corrupt class ``file_integrity`` exists to block. This loop used to
        # ``continue`` past it silently, so a truncated LF sailed through every
        # pre-upload gate and the other LFs' Drive/YouTube uploads proceeded —
        # the corruption surfaced only at the post-upload quality gate, AFTER
        # irreversible uploads. Record it as an integrity failure (which
        # hard-fails via ``assert_file_integrity_hard_fail`` below, upstream of
        # any upload) instead of skipping. ``validate`` can't run on an
        # absent/undersized file, so skip it but keep the failure.
        if not lf_path.exists():
            integrity_failures.append(f"  LF {lf.index} ({lf_path.name}): FINAL.mp4 missing")
            continue
        if lf_path.stat().st_size < 1024 * 1024:
            integrity_failures.append(
                f"  LF {lf.index} ({lf_path.name}): FINAL.mp4 truncated "
                f"({lf_path.stat().st_size} bytes, under the 1 MiB floor)"
            )
            continue
        claim_kwargs = (
            {
                "day_record": day_records[lf.index],
                "claim_native_resolution_only": claim_native_resolution_only,
            }
            if day_records and lf.index in day_records
            else {}
        )
        gate_result = gate_validator.validate(
            video_path=lf_path,
            expected_duration=lf.total_duration,
            metadata_json=metadata_per_lf.get(lf.index),
            skip_audio=skip_audio,
            **claim_kwargs,
        )
        if not gate_result.passed:
            logger.warning(
                "Validation gate WARNING for LF %d:\n%s",
                lf.index,
                gate_result.summary(),
            )
        # The location invariant is the second HARD escalation
        # (the only non-audio gate that aborts). A failing
        # ``location_invariant`` gate is the far-road class — a title county
        # in ZERO sampled GPS regions. Everything else stays WARN-only above.
        location_gate = next(
            (g for g in gate_result.gates if g.gate == "location_invariant"),
            None,
        )
        if location_gate is not None and not location_gate.passed:
            location_failures.append(f"  LF {lf.index} ({lf_path.name}): {location_gate.message}")
        # file_integrity (deterministic: corrupt/truncated/unreadable FINAL.mp4)
        # and content_safety (blocklist violation) are block-or-bust gates; both
        # escalate to HARD FAIL below. content_safety is the JUDGMENT call.
        integrity_gate = next(
            (g for g in gate_result.gates if g.gate == "file_integrity"),
            None,
        )
        if integrity_gate is not None and not integrity_gate.passed:
            integrity_failures.append(f"  LF {lf.index} ({lf_path.name}): {integrity_gate.message}")
        content_gate = next(
            (g for g in gate_result.gates if g.gate == "content_safety"),
            None,
        )
        if content_gate is not None and not content_gate.passed:
            content_failures.append(f"  LF {lf.index} ({lf_path.name}): {content_gate.message}")
        # ``metadata_quality`` escalates too — its keyword check is the one
        # place the title and its own description/location are compared.
        metadata_gate = next(
            (g for g in gate_result.gates if g.gate == "metadata_quality"),
            None,
        )
        if metadata_gate is not None and not metadata_gate.passed:
            metadata_failures.append(f"  LF {lf.index} ({lf_path.name}): {metadata_gate.message}")
        # The encode-contract pair. Both measure the FINAL against frozen
        # encoding_constants, so neither is a judgment call — they abort.
        for gate_name in ("video_quality", "bitrate_preservation", "picture_quality"):
            quality_gate = next(
                (g for g in gate_result.gates if g.gate == gate_name),
                None,
            )
            if quality_gate is not None and not quality_gate.passed:
                video_quality_failures.append(
                    f"  LF {lf.index} ({lf_path.name}): {quality_gate.message}"
                )
        # Post-rendition picture gate; ``picture_inputs`` maps LF index to
        # ``(source_sharpness, picture_changed)``. Absent -> recorded as skipped.
        source_sharpness, changed = (picture_inputs or {}).get(lf.index, (None, False))
        picture_gate = picture_check(lf_path, lf.total_duration, source_sharpness, changed)
        if picture_gate is not None and not picture_gate.passed:
            video_quality_failures.append(
                f"  LF {lf.index} ({lf_path.name}): {picture_gate.message}"
            )
        # The only gate that looks at pixels rather than metadata.
        perceptual_gate = perceptual_check(lf_path, lf.total_duration)
        if perceptual_gate is not None and not perceptual_gate.passed:
            perceptual_failures.append(
                f"  LF {lf.index} ({lf_path.name}): {perceptual_gate.message}"
            )
        if not skip_audio:
            # Both audio gates hard-fail upstream of upload: the integrated
            # ceiling AND the momentary (speech-intelligibility) belt.
            for gate_name in ("audio_loudness", "audio_momentary"):
                audio_gate = next(
                    (g for g in gate_result.gates if g.gate == gate_name),
                    None,
                )
                if audio_gate is not None and not audio_gate.passed:
                    audio_failures.append(f"  LF {lf.index} ({lf_path.name}): {audio_gate.message}")
    _pd.assert_audio_loudness_hard_fail(audio_failures)
    _pd.assert_location_invariant_hard_fail(location_failures)
    _pd.assert_file_integrity_hard_fail(integrity_failures)
    _pd.assert_content_safety_hard_fail(content_failures)
    _pd.assert_metadata_quality_hard_fail(metadata_failures)
    assert_video_quality_hard_fail(video_quality_failures)
    assert_perceptual_spot_hard_fail(perceptual_failures)


def _assert_shorts_quality_gate(state, *, validator_cls, probe_duration_fn=None) -> None:
    """Run the 1080x1920 shorts quality gate, raise on failure.

    Reads ``state["shorts_per_lf"]`` as the audit source; honours the
    ``--allow-bad-shorts`` escape hatch (state field
    ``skip_shorts_quality_gate``). Closes the day-first 16:9 regression
    by failing before any
    Drive upload happens.

    After the shorts pass validation, ffprobe each
    short and record its real duration into
    ``state["short_durations_per_lf"]`` as ``{lf_index: [duration_sec, …]}``.
    That state field is declared + registered (``_INT_KEY_DICT_FIELDS``)
    and read by ``preflight_plan._collect_short_durations`` to estimate
    Phase D wall-clock on resume, but was never WRITTEN — so the estimate
    always fell back to the 30s-per-short default. Populating it here (the
    one place every short is already enumerated and on disk, right after
    the dimension validation) makes the resume estimate honest. The probe
    reuses the canonical ``phase_bc_day_first._probe_duration`` helper (no
    new ffprobe wrapper); ``probe_duration_fn`` is injectable for tests. An
    unprobeable short (``None``) is dropped from its LF list rather than
    poisoning the estimate or raising — the dimension gate above already
    owns hard-fail. No durations are recorded when the gate is skipped or
    there are no shorts.
    """
    if bool(getattr(state, "skip_shorts_quality_gate", False)):
        return
    shorts_paths_by_lf_for_audit = state.shorts_per_lf or {}
    if not shorts_paths_by_lf_for_audit:
        # An empty/absent ``shorts_per_lf`` while regular LFs exist is the
        # glob-ship hole — the downstream upload used to fall back to
        # globbing ``output_base/shorts/lf_NN/`` and ship whatever sat on disk
        # with ZERO LUFS/geometry/OST gating. Refuse rather than validate
        # nothing and let ungated shorts ship. (A genuine zero-shorts day
        # carries per-LF keys with empty lists — ``{i: []}`` — which is truthy
        # and validates as "nothing to check". Escape: ``--allow-bad-shorts``.)
        if any(d.regular_lf_paths for d in deliverables_for_publish(state)):
            raise RuntimeError(
                "shorts_per_lf is empty/absent at Phase D while regular LFs "
                "exist — refusing to ship shorts with no LUFS/geometry/OST "
                "validation (the legacy glob-ship hole). Rebuild shorts_per_lf "
                "from the day-first cache, or pass --allow-bad-shorts to override."
            )
        return
    all_short_files: list[Path] = []
    for _paths in shorts_paths_by_lf_for_audit.values():
        for _p in _paths or []:
            all_short_files.append(Path(_p))
    if not all_short_files:
        return
    logger.info(
        "  D-shorts-quality-gate: validating %d shorts (1080x1920 enforcement)",
        len(all_short_files),
    )
    from ..quality import picture_inputs_from_state

    _sharp, _by_lf = picture_inputs_from_state(state)
    _qr = validator_cls().validate_short_files(
        all_short_files, source_sharpness=_sharp, picture_changed=any(_by_lf.values())
    )
    if not _qr.passed:
        # D-shorts-quality-gate identity preserved for log greps, but the
        # human-readable cause comes from the validator's own per-gate summary
        # rather than asserting resolution. validate_short_files runs resolution
        # AND duration / bitrate / audio-LUFS / integrity checks, so a
        # non-resolution failure (e.g. off-LUFS audio) must not be mislabeled as
        # a 1080x1920 9:16 regression.
        raise RuntimeError(
            "Shorts quality gate FAILED (D-shorts-quality-gate) — at "
            "least one short failed validation. Do NOT upload these to "
            "Drive / Metricool. See the per-gate detail below for the real "
            "cause (resolution, duration, bitrate, or audio LUFS); re-render "
            "the affected short(s) against the preserved day masters. "
            f"Details:\n{_qr.summary()}"
        )
    logger.info("  D-shorts-quality-gate: ALL %d shorts pass", len(all_short_files))

    # Record per-LF measured short durations for the preflight estimate.
    if probe_duration_fn is None:
        from .phase_bc_day_first import _probe_duration as probe_duration_fn
    durations_per_lf: dict[int, list[float]] = {}
    for lf_index, _paths in shorts_paths_by_lf_for_audit.items():
        durs: list[float] = []
        for _p in _paths or []:
            probed = probe_duration_fn(Path(_p))
            if probed is None:
                continue
            try:
                val = float(probed)
            except (TypeError, ValueError):
                continue
            if val > 0:
                durs.append(val)
        if durs:
            durations_per_lf[int(lf_index)] = durs
    state.short_durations_per_lf = durations_per_lf
