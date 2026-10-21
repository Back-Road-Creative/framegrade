"""ReviewGate -- human-review checkpoints at pipeline phase boundaries.

Each gate runs automated quality checks first (fail-fast, no prompt) then
prompts the user for confirmation before the next phase begins.  When
``interactive=False`` the human prompt is skipped but automated checks still
run, allowing CI / headless runs to catch regressions.
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .validation.content_gates import _label_quality_diagnosis


class ReviewFailedError(RuntimeError):
    """Raised when a review gate fails automated checks or the human rejects."""


GATE_ANSWERS_SUBDIR = "_gates"

#: Default cap on how long a non-interactive gate waits for a console
#: operator to answer a question it just wrote. Zero (the default) means
#: "don't wait" -- the old skip-and-leave-question-pending behaviour --
#: since most non-interactive callers (cron, a headless CI/CLI run) have no
#: console attached and nobody will ever write the answer file. Opt in via
#: ``GMS_GATE_ANSWER_TIMEOUT_SEC`` (seconds); the console container sets it
#: (``docker/console-compose.yaml``) because it IS the one place a console
#: operator can answer a job it launched.
DEFAULT_GATE_ANSWER_TIMEOUT_SEC = 0
_GATE_ANSWER_POLL_INTERVAL_SEC = 5


def _poll_for_answer(answer_path: Path, *, timeout_sec: "float | None" = None) -> bool:
    """Block until ``answer_path`` exists or ``timeout_sec`` elapses.

    Used by ``_consume_or_ask`` so a non-interactive run that just wrote a
    question file waits for a console operator to answer it, instead of
    returning ``None`` (and quietly proceeding) on the very same call.
    ``timeout_sec`` defaults to ``GMS_GATE_ANSWER_TIMEOUT_SEC`` (env) or
    :data:`DEFAULT_GATE_ANSWER_TIMEOUT_SEC`, which is ``0`` -- opt-in only,
    since most non-interactive callers have nobody to answer them. A
    non-positive timeout returns immediately with no sleep at all (the old
    behaviour). Isolated to one small function so tests can monkeypatch
    ``time.sleep`` to fake the passage of time.
    """
    if timeout_sec is None:
        timeout_sec = float(
            os.environ.get("GMS_GATE_ANSWER_TIMEOUT_SEC", DEFAULT_GATE_ANSWER_TIMEOUT_SEC)
        )
    if timeout_sec <= 0:
        return answer_path.exists()
    deadline = time.monotonic() + timeout_sec
    logged = False
    while not answer_path.exists():
        if time.monotonic() >= deadline:
            return False
        if not logged:
            print(f"\n  [waiting for console answer at {answer_path}]")
            logged = True
        time.sleep(_GATE_ANSWER_POLL_INTERVAL_SEC)
    return True


def _read_and_consume_answer(answer_path: Path, choices: tuple) -> "str | None":
    """Read, validate, and retire an answer file.

    A malformed answer file (unparsable JSON, unreadable) is left in place
    and a ``<name>.rejected`` sidecar records the parse error -- the
    operator's decision is never silently destroyed, and a corrected answer
    can still be written and picked up later. A well-formed file is
    consumed -- renamed to ``<name>.used`` -- regardless of whether its
    decision is one of ``choices``.
    """
    try:
        payload = json.loads(answer_path.read_text(encoding="utf-8"))
        decision = payload.get("decision")
    except (OSError, json.JSONDecodeError) as exc:
        rejected_path = answer_path.with_name(answer_path.name + ".rejected")
        rejected_path.write_text(str(exc), encoding="utf-8")
        return None
    answer_path.rename(answer_path.with_name(answer_path.name + ".used"))
    if decision in choices:
        return decision
    return None


def _gate_answer_path(output_dir: Path, gate_id: str) -> Path:
    return Path(output_dir) / GATE_ANSWERS_SUBDIR / f"{gate_id}.answer"


def _gate_question_path(output_dir: Path, gate_id: str) -> Path:
    return Path(output_dir) / GATE_ANSWERS_SUBDIR / f"{gate_id}.question.json"


def _label_quality_warning(metadata_json: dict) -> "str | None":
    """Return ``validation.content_gates.check_label_quality``'s WARN
    message text (or ``None``) instead of only logging it.

    That gate is WARN-only by design and its ``GateResult`` never carries
    the warning text, so a flagged LF label scrolls into the log ~400 lines
    above the Gate C/D prompt and the operator has to catch it by eye. This
    calls the SAME ``_label_quality_diagnosis`` helper ``check_label_quality``
    uses, so Gate C/D can print it inline, right under the title it concerns,
    without the two ever disagreeing on the verdict.

    No-op (returns ``None``) when there is nothing to flag -- same as
    ``check_label_quality`` (GPS-less / legacy LFs, or a clean label).
    """
    diagnosis = _label_quality_diagnosis(metadata_json)
    if diagnosis is None:
        return None
    kind, settlements = diagnosis
    if kind == "title":
        return (
            f"label is a title, not a place; use {settlements[0]!r} as the "
            "location -- keep the title wording for the OST only"
        )
    return (
        f"county-only label; settlement(s) available: {settlements[:5]} -- "
        "re-run with --per-lf-location 'INDEX=...' to anchor the title"
    )


def compute_gate_cd_hash(*, state, output_dir: Path) -> str:
    """Stable sha256 of Gate C/D content payload.

    Covers LF count, per-LF title + youtube.json sha, consolidated
    Metricool CSV row count + header sha, and (when present) each
    ``*_publish_plan.json`` sidecar's canonicalized content sha -- the
    first-party publish-plan artefact migrating shorts off the Metricool
    CSV. Phase D's ``_emit_publish_plan_sidecar`` DOES write it now, but
    only when a ``DriveConfig.publishing_route_*`` is ``first_party``; the
    ``plans`` key is added to the hashed payload only when at least one
    sidecar exists, so a run with every route on ``metricool`` hashes
    byte-identically to before the sidecar existed.

    This hash is what ``cli/handlers/enqueue_plan.py`` compares against
    ``state.gate_cd_confirmation_hash`` before spooling any delivery, so a
    plan edited after review cannot reach the queue. Excludes wall-clock fields (``generated_at``,
    mtimes) so a stale state file + freshly-regenerated artefact pair
    don't churn the hash.
    """
    output_dir = Path(output_dir)
    lf_count = len(list(state.regular_lf_paths or []))
    lf_entries: list[dict] = []
    for path in sorted(output_dir.glob("*_youtube.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        meta_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        lf_entries.append(
            {"file": path.name, "title": str(payload.get("title", "") or ""), "meta_sha": meta_sha}
        )
    csv_entries: list[dict] = []
    for csv_path in sorted(output_dir.glob("*_metricool.csv")):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except OSError:
            rows = []
        header = rows[0] if rows else []
        header_sha = hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8")).hexdigest()
        csv_entries.append(
            {"file": csv_path.name, "row_count": len(rows), "header_sha": header_sha}
        )
    plan_entries: list[dict] = []
    for plan_path in sorted(output_dir.glob("*_publish_plan.json")):
        try:
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plan_payload = {}
        plan_sha = hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        plan_entries.append({"file": plan_path.name, "plan_sha": plan_sha})

    payload = {"lf_count": lf_count, "lfs": lf_entries, "csvs": csv_entries}
    if plan_entries:
        # Only added when a sidecar exists -- see docstring: this keeps the
        # hash byte-identical to the pre-plan-coverage payload when no
        # sidecar is written, which is every run today.
        payload["plans"] = plan_entries
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# Pipeline evidence embedded in the metadata JSON, judged by its own gate
# (final_text_record) -- never shipped as title/description/tag text.
_EVIDENCE_RECORD_KEYS = frozenset({"final_text_review"})

# Waypost's description entries: each ``text`` ships, while ``kind`` /
# ``surface`` / ``template_id`` only label where the text came from.
_SHIPPED_ENTRY_KEYS = frozenset({"final_text_description"})


def _string_values(obj) -> list[str]:
    """Every string value in a parsed JSON document, keys excluded."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        return [s for item in obj for s in _string_values(item)]
    return []


class ReviewGate:
    MIN_BITRATE_KBPS = 40_000  # 40 Mbps floor — above YouTube 4K rec (35-45 Mbps)
    MAX_AV_DRIFT_PCT = 2.0  # A/V duration drift threshold (%)
    EXPECTED_STREAM_COUNT = 1  # minimum: video stream required (audio optional)
    EXPECTED_CSV_COLS = 31  # Metricool format (warn-only)
    PLACEHOLDER_PATTERNS = [  # Strings that must NOT appear in metadata
        "guess the location",
        "LOCATION_PLACEHOLDER",
        "TODO",
        "<location>",
        "INSERT_",
        "TEMPLATE",
    ]

    def __init__(self, interactive: bool = True) -> None:
        self.interactive = interactive
        # Set by a gate method just before calling ``_prompt``/
        # ``_prompt_required`` so the shared non-TTY path knows
        # where to write/poll an answer file -- kept off the public
        # ``_prompt``/``_prompt_required`` signatures so every existing
        # test fake/monkeypatch of those two methods keeps working
        # unchanged.
        self._gate_answer_output_dir: "Path | None" = None
        self._gate_answer_id: "str | None" = None

    # ------------------------------------------------------------------
    # Public gates
    # ------------------------------------------------------------------

    def gate_a(
        self,
        *,
        state: dict,
        output_dir: Path,
        location: str,
        skeleton_lf_indices: list[int] | None = None,
    ) -> None:
        """Gate after Phase A: verify episode breakdown and OST text.

        Automated check: episode_files is non-empty.
        Human prompt: confirm episode breakdown and OST text.

        OST display priority (highest first — matches the actual burn
        cascade in phase_bc / ost_applier):
          1. Per-LF overrides (``state["per_lf_overrides"]``) — operator
             passed ``--per-lf-location`` on the CLI, this is the text
             that will actually burn for the LF.
          2. Per-episode OST (``state["episode_locations"]``) — populated
             by GPS-centroid reverse-geocoding for GPS-bearing drives.
          3. Drive-wide ``--location`` — Super LF / non-LF fallback only.

        Previously this method showed (3) at the bottom even when (1)
        was set, training operators to confirm a value that the burn path
        ignores. (1) is now displayed prominently so the prompt reviews
        the effective burn text.
        """
        self._print_header("GATE A -- Episode Breakdown Review")

        episode_files = state.episode_files or []
        episode_dates = state.episode_dates or []
        file_durations = state.file_durations or []
        gps_per_episode = state.gps_per_episode or []

        # Print episode table
        print(f"  Episodes found: {len(episode_files)}")
        for i, (ep_path, date, dur) in enumerate(zip(episode_files, episode_dates, file_durations)):
            gps_entry = gps_per_episode[i] if i < len(gps_per_episode) else None
            has_gps = bool(gps_entry)
            dur_min = dur / 60.0 if dur else 0.0
            gps_label = "GPS OK" if has_gps else "no GPS"
            print(f"    [{i:02d}] {date}  {dur_min:.1f} min  {gps_label}  {ep_path}")

        # OST display — priority cascade
        per_lf_overrides = state.per_lf_overrides or {}
        episode_locations = state.episode_locations or []

        if per_lf_overrides:
            print("\n  Per-LF OST overrides (--per-lf-location — BURNED into each LF):")
            for idx in sorted(per_lf_overrides):
                print(f'    [{idx:02d}] "{per_lf_overrides[idx]}"')
            # hide drive-wide label when per-LF overrides cover every
            # skeleton LF — the label is dead data the resolver would never
            # use. Robust by construction: the display layer can't surface a
            # value the burn cascade ignores.
            all_lfs_covered = (
                skeleton_lf_indices is not None
                and len(skeleton_lf_indices) > 0
                and set(per_lf_overrides.keys()) >= set(skeleton_lf_indices)
            )
            if all_lfs_covered:
                print("\n  Drive-wide location: not used — every LF has a per-LF override")
            else:
                print(f'\n  Drive-wide --location (Super LF / non-LF fallback only): "{location}"')
        elif episode_locations and any(episode_locations):
            print("\n  Per-episode OST (burned into each episode during encode):")
            for i, ep_ost in enumerate(episode_locations):
                print(f"    [{i:02d}] {ep_ost or '(no GPS — will use drive-level)'}")
            print(f'\n  Drive-level OST (super LF / fallback): "{location}"')
        else:
            print(f'\n  OST text: "{location}"')

        # Automated check
        if not episode_files:
            self._fail("No episode files found -- cannot proceed")

        self._gate_answer_output_dir = output_dir
        self._gate_answer_id = "gate_a"
        self._prompt("Episode breakdown correct and OST text exact?")

    def gate_preview(self, *, output_dir: Path) -> None:
        """Gate after preview encode: verify preview clips exist and look right.

        Automated check: at least one preview clip exists.
        Human prompt: confirm previews look correct.

        Audio is intentionally NOT reviewed here: every shipped surface is
        deterministically attenuated to near-silence (the operator near-silent
        rule) and hard-gated, so an "audio OK?" question is moot — the preview
        clips carry that same attenuation for fidelity, not for judgement.
        """
        self._print_header("GATE PREVIEW -- Preview Clip Review")

        preview_dir = output_dir / "_stages" / "preview"
        clips = sorted(preview_dir.glob("ep_*_preview.mp4")) if preview_dir.exists() else []

        # Automated check
        if not clips:
            self._fail(f"No preview clips found in {preview_dir}")

        for clip in clips:
            bitrate_kbps = self._probe_bitrate_kbps(clip)
            bitrate_mbps = bitrate_kbps / 1000.0
            warn = ""
            if bitrate_kbps < self.MIN_BITRATE_KBPS:
                warn = "  [WARN: bitrate low -- previews use quality 20, not CQ 1]"
            print(f"  {clip.name}  {bitrate_mbps:.1f} Mbps{warn}")
            print(f"    Playback: ffplay -autoexit {clip}")

        self._gate_answer_output_dir = output_dir
        self._gate_answer_id = "gate_preview"
        self._prompt("All previews watched? OST visible + correct, color natural?")

    def gate_b(self, *, state: dict, output_dir: Path) -> None:
        """Gate after Phase B/BC: verify encoded file quality (streams, drift, bitrate).

        Automated checks: stream count, A/V drift, bitrate floor.
        Human prompt: spot-check an output file visually.

        Supports both old Phase B layout (ep_*_FINAL.mp4) and day-first
        layout (day_*/day_*_master.mp4 + day_*/regular_lfs/lf_*.mp4).
        """
        self._print_header("GATE B -- Encode Quality Check")

        # Collect output file paths — try day-first artifacts first, then old layout
        final_paths: list = []
        day_artifacts = state.day_artifacts or []
        if day_artifacts:
            for art in day_artifacts:
                # art may be a DayArtifacts dataclass OR a dict (legacy
                # state rehydration). When it's a dataclass with
                # master_path=None (the intentional state after the per-day
                # master purge and finals-only reconstruction), `getattr(...) or art.get(...)` falls
                # into `.get()` which doesn't exist on dataclasses →
                # AttributeError. Check dict-ness explicitly.
                if isinstance(art, dict):
                    master = art.get("master_path")
                    lfs = art.get("regular_lf_paths") or []
                else:
                    master = getattr(art, "master_path", None)
                    lfs = getattr(art, "regular_lf_paths", None) or []
                if master and Path(master).exists():
                    final_paths.append(Path(master))
                for lf in lfs:
                    if lf and Path(lf).exists():
                        final_paths.append(Path(lf))
        if not final_paths:
            # Fall back to day masters in _stages
            stages_dir = output_dir / "_stages"
            final_paths = sorted(stages_dir.glob("day_*/day_*_master.mp4"))
        if not final_paths:
            # Fall back to old Phase B layout
            processed_files: dict = state.processed_files or {}
            if processed_files:
                final_paths = [Path(v) for v in processed_files.values()]
            else:
                final_paths = sorted(
                    (output_dir / "_stages" / "processed").glob("ep_*_FINAL.mp4"),
                )

        if not final_paths:
            self._fail("No encoded files found for quality check")

        print(f"  {'File':<30} {'Streams':>7} {'Bitrate (Mbps)':>14} {'Drift %':>8} {'Status':>6}")
        print("  " + "-" * 70)

        failures: list[str] = []

        for path in final_paths:
            path = Path(path)
            stream_count = self._stream_count(path)
            drift = self._av_drift_pct(path)
            bitrate_kbps = self._probe_bitrate_kbps(path)
            bitrate_mbps = bitrate_kbps / 1000.0

            row_fail = False
            reasons: list[str] = []

            if stream_count < self.EXPECTED_STREAM_COUNT:
                reasons.append(
                    f"stream count {stream_count} < {self.EXPECTED_STREAM_COUNT} (no video)"
                )
                row_fail = True
            if drift >= self.MAX_AV_DRIFT_PCT:
                reasons.append(f"A/V drift {drift:.2f}% >= {self.MAX_AV_DRIFT_PCT}%")
                row_fail = True
            if bitrate_kbps <= self.MIN_BITRATE_KBPS:
                reasons.append(
                    f"bitrate {bitrate_mbps:.1f} Mbps <= {self.MIN_BITRATE_KBPS / 1000:.0f} Mbps floor"
                )
                row_fail = True

            status = "FAIL" if row_fail else "PASS"
            print(
                f"  {path.name:<30} {stream_count:>7} {bitrate_mbps:>14.1f} {drift:>8.2f} {status:>6}"
            )
            if reasons:
                for r in reasons:
                    failures.append(f"{path.name}: {r}")

        if failures:
            self._fail("FINAL file checks failed:\n  " + "\n  ".join(failures))

        # Spot-check playback hint
        if final_paths:
            print(f"\n  Spot-check: ffplay {final_paths[0]}")

        self._gate_answer_output_dir = output_dir
        self._gate_answer_id = "gate_b"
        self._prompt("Visually spot-checked FINAL files? OST visible, color correct?")

    def gate_pre_render(
        self,
        *,
        lf_index: int,
        ost_text: str,
        manifest_path: Path | None = None,
        social_preview: str = "",
        output_dir: Path,
    ) -> None:
        """Gate before LF OST burn-in: automated OST checks + human confirmation.

        The automated checks below — empty/whitespace OST, placeholder
        strings, an unreadable published-titles ledger, and an exact
        channel-title duplicate — are HARD gates that run and can raise
        regardless of ``interactive``; they are the real protection against a
        bad OST burn. The trailing human "OST exact?" confirmation is a prompt
        like every other gate and IS skipped under ``--no-review``
        (``interactive=False``). An earlier revision claimed this gate "cannot
        be bypassed by --no-review" — that was false (only the automated checks
        are unconditional) and is corrected here.

        If manifest_path is provided, displays GPS evidence from the manifest
        so the user can verify the title against actual route data. The user
        may edit the manifest file before approving — the caller reads back
        the (possibly edited) proposed_title after approval.
        """
        self._print_header(f"GATE PRE-RENDER -- LF {lf_index:02d} OST Approval")

        manifest: dict = {}
        if manifest_path and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            print(f'\n  Proposed title: "{manifest.get("proposed_title", ost_text)}"')
            print(f"  Duration: {manifest.get('total_duration_s', 0):.0f}s")

            # GPS evidence
            gps_s = manifest.get("gps_start")
            gps_e = manifest.get("gps_end")
            if gps_s:
                print(f"  GPS start: {gps_s['lat']}, {gps_s['lon']}")
            if gps_e:
                print(f"  GPS end:   {gps_e['lat']}, {gps_e['lon']}")

            # Source episodes with their OSTs
            episodes = manifest.get("source_episodes", [])
            if episodes:
                print(f"\n  Source episodes ({len(episodes)}):")
                for ep in episodes:
                    print(f"    [{ep['index']:02d}] {ep.get('ost') or '(no OST)'}")

            # Derivation details
            deriv = manifest.get("derivation", {})
            print(
                f"\n  Derivation: start={deriv.get('start_town')!r}"
                f"  end={deriv.get('end_town')!r}"
                f"  road={deriv.get('road')!r}"
                f"  state={deriv.get('state')!r}"
            )

            print(f"\n  Edit title in: {manifest_path}")
        else:
            print(f'\n  OST location text (will be burned into video):\n    "{ost_text}"')
            if social_preview:
                print(f"\n  Social post preview:\n{social_preview}")

        print(
            "\n  [OST checks below are unconditional; the human confirmation "
            "is skipped under --no-review]"
        )

        if not ost_text or not ost_text.strip():
            self._fail("OST text is empty — cannot burn in blank text")
        hits = [p for p in self.PLACEHOLDER_PATTERNS if p.lower() in ost_text.lower()]
        if hits:
            self._fail(f"OST text contains placeholder pattern(s): {hits}")

        # Hard gate: the FINAL title must not exactly duplicate an already-
        # published channel video. This is the last checkpoint before the
        # irreversible OST burn — a true duplicate is a blocking failure, not
        # a warning. Upstream, ``ensure_unique_titles`` already deduped FINAL
        # titles against the channel, so this only fires if that dedup was
        # bypassed. Nothing writes ``proposed_title`` in production,
        # so falling back to the raw OST label compared the wrong namespace —
        # the label ships as its assembled final title, so assemble it here
        # (same pure reshape the metadata generator applies).
        from .longform_metadata import (  # noqa: PLC0415
            LEDGER_UNREADABLE,
            assemble_final_title,
            read_published_titles,
        )

        # distinguish a MISSING ledger (fresh channel, no history → the
        # dedup check legitimately finds nothing) from an UNREADABLE one
        # (OSError/decode error → dedup CANNOT be verified). Shipping on an
        # unverified ledger could silently duplicate a live title, so refuse
        # the irreversible OST burn rather than degrade to a silent empty set.
        if read_published_titles()[1] == LEDGER_UNREADABLE:
            self._fail(
                "Published-titles ledger is unreadable — cannot verify the "
                "FINAL title is not a duplicate of a live channel video; "
                "refusing OST burn on unverified dedup"
            )

        final_title = manifest.get("proposed_title") or assemble_final_title(ost_text)
        exact_dupes = self._check_past_title_collisions(final_title, exact=True)
        if exact_dupes:
            self._fail(
                "FINAL title exactly duplicates an already-published channel "
                f"title (dedup vs channel failed upstream): {exact_dupes[:3]}"
            )

        # Advisory (non-blocking): OST location text echoes a past title.
        collisions = self._check_past_title_collisions(ost_text)
        if collisions:
            print(f"\n  [WARN] OST text may collide with {len(collisions)} past title(s):")
            for c in collisions[:5]:
                print(f"    - {c}")
            if len(collisions) > 5:
                print(f"    ... and {len(collisions) - 5} more")

        self._gate_answer_output_dir = output_dir
        self._gate_answer_id = f"gate_pre_render_{lf_index:02d}"
        self._prompt_required(f"OST text exact? Approve LF {lf_index:02d} render?")

    def gate_cd(self, *, state: dict, output_dir: Path, date_key: str) -> None:
        """Gate after Phase C+D for one day: verify LFs, shorts, metadata, CSVs.

        Automated checks: LF stream count, drift, non-zero video duration,
        placeholder strings in metadata JSON.
        CSV column count is warn-only (format varies between shorts/LF CSVs).
        Human prompt: confirm all LF content reviewed.
        """
        self._print_header(f"GATE C/D -- Day {date_key} Content Review")

        failures: list[str] = []

        # --- LF checks ---
        lf_paths = state.regular_lf_paths or []
        if lf_paths:
            print(f"\n  Long-form files ({len(lf_paths)}):")
        for lf_path in lf_paths:
            lf_path = Path(lf_path)
            stream_count = self._stream_count(lf_path)
            drift = self._av_drift_pct(lf_path)
            v_dur, _ = self._probe_av_durations(lf_path)
            v_min = v_dur / 60.0 if v_dur else 0.0

            row_fail = False
            if stream_count < self.EXPECTED_STREAM_COUNT:
                failures.append(
                    f"{lf_path.name}: stream count {stream_count} != {self.EXPECTED_STREAM_COUNT}"
                )
                row_fail = True
            if drift >= self.MAX_AV_DRIFT_PCT:
                failures.append(
                    f"{lf_path.name}: A/V drift {drift:.2f}% >= {self.MAX_AV_DRIFT_PCT}%"
                )
                row_fail = True
            if v_dur == 0:
                failures.append(f"{lf_path.name}: video duration is zero")
                row_fail = True

            status = "FAIL" if row_fail else "PASS"
            print(
                f"    {lf_path.name:<40} {v_min:.1f} min  streams={stream_count}"
                f"  drift={drift:.2f}%  {status}"
            )
            print(f"    Playback: ffplay {lf_path}")

        # --- Shorts counts ---
        shorts_per_lf: dict = state.shorts_per_lf or {}
        for lf_idx, shorts in shorts_per_lf.items():
            count = len(shorts) if hasattr(shorts, "__len__") else 0
            if count == 0:
                print(f"  [WARN] LF index {lf_idx}: 0 shorts extracted (can be flaky)")

        # --- Metadata checks ---
        json_files = sorted(output_dir.glob("*_youtube.json"))
        if json_files:
            print(f"\n  Metadata files ({len(json_files)}):")
        label_quality_warnings = 0
        for json_path in json_files:
            hits = self._check_placeholders(json_path)
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                title = str(data.get("title", ""))[:80]
            except Exception:
                data = {}
                title = "(unreadable)"
            print(f"    {json_path.name}: {title!r}")
            warning = _label_quality_warning(data)
            if warning:
                label_quality_warnings += 1
                print(f"      !! LABEL-QUALITY: {warning}")
            if hits:
                failures.append(f"{json_path.name}: Placeholder pattern(s) found: {hits}")

        # --- CSV checks (warn only -- format varies between shorts/LF CSVs) ---
        csv_files = sorted(output_dir.glob("*_metricool.csv"))
        if csv_files:
            print(f"\n  CSV files ({len(csv_files)}):")
        for csv_path in csv_files:
            col_count = self._count_csv_columns(csv_path)
            status = "OK" if col_count == self.EXPECTED_CSV_COLS else "WARN"
            print(f"    {csv_path.name}: {col_count} columns  [{status}]")
            if col_count != self.EXPECTED_CSV_COLS:
                print(
                    f"    [WARN] {csv_path.name}: column count {col_count}"
                    f" != expected {self.EXPECTED_CSV_COLS}"
                )

        if failures:
            self._fail(f"Day {date_key} checks failed:\n  " + "\n  ".join(failures))

        # An explicit prior confirmation — ``--gate-cd-confirmed`` carried on
        # state together with a still-matching content hash (e.g. a resume of
        # an already-human-reviewed day) — waives the re-prompt. A material
        # content change flips the hash and reopens the gate.
        current_hash = compute_gate_cd_hash(state=state, output_dir=output_dir)
        if state.gate_cd_confirmed is True and state.gate_cd_confirmation_hash == current_hash:
            print("\n  [Gate C/D auto-passed — --gate-cd-confirmed + hash match]")
            return
        # Known hole: a run whose stdin is not a TTY used to
        # auto-pass AND PERSIST gate_cd_confirmed + hash — so ONE headless run
        # silently waived this content review for EVERY later run of that state
        # (including interactive human ones) until the content changed. That
        # non-tty auto-pass is removed. Nothing but a genuine human 'y' (or the
        # explicit prior --gate-cd-confirmed waiver above) may pass, and neither
        # a --no-review run nor a non-tty run may PERSIST a waiver.
        # An answer file may pre-resolve this gate (e.g. a browser
        # console answering on the operator's behalf) -- check it BEFORE the
        # interactive-TTY requirement below, so a headless run backed by an
        # answer file never needs a TTY at all. This runs regardless of
        # ``interactive`` -- a non-interactive (--no-review) run used to
        # return above before ever reaching this, so a console/headless run
        # never wrote question.json and could never be answered out-of-band.
        message = f"Played each LF and reviewed all metadata for {date_key}?"
        artifacts = [str(p) for p in lf_paths] + [str(p) for p in json_files]
        decision = self._consume_or_ask(
            output_dir=output_dir,
            gate_id=f"gate_cd_{date_key}",
            message=message,
            artifacts=artifacts,
        )
        if decision == "approve":
            state.gate_cd_confirmed = True
            state.gate_cd_confirmation_hash = current_hash
            return
        if decision == "reject":
            self._fail(f"Human review rejected via answer file: {message}")

        if not self.interactive:
            # --no-review, no pre-staged answer: the human prompt is skipped
            # like every other gate, but the waiver is NOT persisted
            # (persisting it is the poison) -- question.json above is left
            # in place for a later console to answer.
            return

        if not sys.stdin.isatty():
            self._fail(
                "Gate C/D content review requires an interactive TTY "
                "confirmation, but stdin is not a terminal. Re-run with a TTY, "
                "or pass --gate-cd-confirmed only after a prior human review of "
                "this exact content (its hash must still match)."
            )
        if label_quality_warnings:
            print(
                f"\n  [WARN] {label_quality_warnings} LF label(s) flagged by LABEL-QUALITY "
                "above -- review before confirming"
            )
        self._prompt(message)
        state.gate_cd_confirmed = True
        state.gate_cd_confirmation_hash = current_hash

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_streams(self, path: Path) -> list[dict]:
        """Run ffprobe and return list of stream dicts. Returns [] on error."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            return data.get("streams", [])
        except Exception:
            return []

    def _stream_count(self, path: Path) -> int:
        """Return number of streams in the file."""
        return len(self._probe_streams(path))

    def _probe_av_durations(self, path: Path) -> tuple[float, float]:
        """Return (video_duration_s, audio_duration_s). Returns (0, 0) on error."""
        streams = self._probe_streams(path)
        v_dur = 0.0
        a_dur = 0.0
        for s in streams:
            try:
                dur = float(s.get("duration", 0))
            except (ValueError, TypeError):
                dur = 0.0
            if s.get("codec_type") == "video":
                v_dur = dur
            elif s.get("codec_type") == "audio":
                a_dur = dur
        return v_dur, a_dur

    def _av_drift_pct(self, path: Path) -> float:
        """Return abs(v-a)/max(v,1)*100. Returns 0.0 for video-only files."""
        v_dur, a_dur = self._probe_av_durations(path)
        if a_dur == 0.0:
            return 0.0
        return abs(v_dur - a_dur) / max(v_dur, 1.0) * 100.0

    def _probe_bitrate_kbps(self, path: Path) -> float:
        """Return bitrate in kbps from format/bit_rate. Returns 0 on error."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return 0.0
            data = json.loads(result.stdout)
            bit_rate = data.get("format", {}).get("bit_rate", "0")
            return float(bit_rate) / 1000.0
        except Exception:
            return 0.0

    def _check_placeholders(self, json_path: Path) -> list[str]:
        """Return list of matched placeholder patterns found in the JSON file.

        Only string values are checked: key names such as ``insert_count`` or
        ``template_id`` are schema, not shipped text. Unparseable JSON falls
        back to the raw text. Embedded evidence records (``final_text_review``)
        carry policy ids, not shipped text, and have their own gate. Of
        Waypost's description entries only the ``text`` is checked."""
        try:
            text = json_path.read_text(encoding="utf-8")
        except Exception:
            return []
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            data = {
                k: [e.get("text") for e in v if isinstance(e, dict)]
                if k in _SHIPPED_ENTRY_KEYS and isinstance(v, list)
                else v
                for k, v in data.items()
                if k not in _EVIDENCE_RECORD_KEYS
            }
        if data is not None:
            text = "\n".join(_string_values(data))
        matched = []
        for pattern in self.PLACEHOLDER_PATTERNS:
            if pattern.lower() in text.lower():
                matched.append(pattern)
        return matched

    def _count_csv_columns(self, csv_path: Path) -> int:
        """Return column count from first header row."""
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return 0
                return len(header)
        except Exception:
            return 0

    def _check_past_title_collisions(self, text: str, *, exact: bool = False) -> list[str]:
        """Return past published YouTube titles that collide with ``text``.

        Titles come from :func:`longform_metadata.load_published_titles`,
        which reads the analytics event ledger the YouTube sync populates.
        The previous implementation read a bare ``SEOConfig()`` default path
        (``./data/seo``) that never matched the configured store, so the
        check silently found nothing in production.

        ``exact=False`` (default): advisory substring match — any past title
        that *contains* ``text``; used to warn when an OST location string
        echoes a live title. ``exact=True``: the past title *equals* ``text``
        case-insensitively (trimmed) — a genuine duplicate of a live video,
        used as a hard gate on the FINAL title before the irreversible OST
        burn.
        """
        from gomoveshift.pipeline.longform_metadata import (  # noqa: PLC0415
            load_published_titles,
        )

        needle = text.lower().strip()
        if not needle:
            return []
        published = load_published_titles()
        if exact:
            hits = {t for t in published if t.lower().strip() == needle}
        else:
            hits = {t for t in published if needle in t.lower()}
        return sorted(hits)

    def _consume_or_ask(
        self,
        *,
        output_dir: Path,
        gate_id: str,
        message: str,
        artifacts: "list[str] | None" = None,
        choices: tuple = ("approve", "reject"),
    ) -> "str | None":
        """Answer-file path shared by every gate that can run without a TTY
        (an answer file first, then the console).

        Before a gate falls back to its interactive prompt it checks for
        ``<output_dir>/_gates/<gate_id>.answer`` (JSON: ``decision``,
        ``note``, ``answered_at``). If present it is consumed -- renamed to
        ``<gate_id>.answer.used`` -- and its ``decision`` is returned so the
        caller never prompts. Absent an answer file, ``<gate_id>.question.json``
        is written describing what the gate is asking (a later console can
        write the answer file).

        Whether this then polls for that answer no longer depends on
        ``self.interactive`` -- it depends only on
        ``GMS_GATE_ANSWER_TIMEOUT_SEC``. Unset/``0`` (the default) returns
        ``None`` immediately -- a headless cron/CLI run with no console
        attached must never block waiting for an answer nobody will give.
        A positive value polls (see :func:`_poll_for_answer`) up to that
        timeout; if no answer ever arrives, this RAISES rather than
        returning ``None`` -- a question that was written and never
        answered must fail the run, never let it silently proceed past a
        review it asked for. ``question.json`` is left in place either way
        so a later run/console can still answer it.
        """
        answer_path = _gate_answer_path(output_dir, gate_id)
        if answer_path.exists():
            return _read_and_consume_answer(answer_path, choices)
        question_path = _gate_question_path(output_dir, gate_id)
        question_path.parent.mkdir(parents=True, exist_ok=True)
        question_path.write_text(
            json.dumps(
                {
                    "gate_id": gate_id,
                    "question": message,
                    "choices": list(choices),
                    "artifacts": artifacts or [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        timeout_sec = float(
            os.environ.get("GMS_GATE_ANSWER_TIMEOUT_SEC", DEFAULT_GATE_ANSWER_TIMEOUT_SEC)
        )
        if timeout_sec <= 0:
            return None
        if _poll_for_answer(answer_path, timeout_sec=timeout_sec):
            return _read_and_consume_answer(answer_path, choices)
        raise ReviewFailedError(
            f"No console answer for gate {gate_id!r} within {timeout_sec:.0f}s "
            f"(question left at {question_path}): {message}"
        )

    def _prompt(self, message: str) -> None:
        """Print message, wait for y/N input. Raises ReviewFailedError on N.

        In non-interactive mode, auto-passes (no input read). See
        :meth:`_prompt_impl` for the non-TTY behaviour.
        """
        self._prompt_impl(message)

    def _prompt_required(self, message: str) -> None:
        """Prompt for irreversible steps. Auto-passes in non-interactive mode.

        See :meth:`_prompt_impl` for the non-TTY behaviour.
        """
        self._prompt_impl(message)

    def _prompt_impl(self, message: str) -> None:
        """Shared body of ``_prompt``/``_prompt_required``.

        ``self.interactive`` picks review on/off exactly as before. When
        review is on, ``sys.stdin.isatty()`` then picks HOW: a real
        terminal reads ``input()`` unchanged; anything else (a console job
        with review left on has no console attached to its stdin -- calling
        ``input()`` there raises ``EOFError``) routes through the same
        answer-file mechanism Gate C/D already used, via
        :meth:`_consume_or_ask`. An approve/reject answer resolves the
        prompt; no answer within the poll window is a hard failure -- never
        a silent pass on a review the operator asked for by leaving
        ``--no-review`` off.

        ``output_dir``/``gate_id`` come from ``self._gate_answer_output_dir``/
        ``self._gate_answer_id`` -- set by the calling gate method just
        before it calls ``_prompt``/``_prompt_required`` -- rather than a
        parameter, so the public two-method surface every test fake already
        monkeypatches stays ``(self, message)``.
        """
        if not self.interactive:
            return
        if sys.stdin.isatty():
            answer = input(f"\n  {message} [y/N] ").strip().lower()
            if answer != "y":
                raise ReviewFailedError(f"Human review rejected: {message}")
            return
        output_dir = self._gate_answer_output_dir
        gate_id = self._gate_answer_id
        if output_dir is None or gate_id is None:
            raise ReviewFailedError(
                f"Review requires a TTY confirmation, but stdin is not a "
                f"terminal and no answer-file gate is available: {message}"
            )
        decision = self._consume_or_ask(output_dir=output_dir, gate_id=gate_id, message=message)
        if decision == "approve":
            return
        if decision == "reject":
            raise ReviewFailedError(f"Human review rejected via answer file: {message}")
        # _consume_or_ask only returns None when GMS_GATE_ANSWER_TIMEOUT_SEC
        # is unset/0 -- otherwise it raises on timeout itself -- so a real
        # console answer was never possible for this run either.
        raise ReviewFailedError(
            f"Review requires a TTY confirmation, but stdin is not a "
            f"terminal and no console answer arrived: {message}"
        )

    def _print_header(self, title: str) -> None:
        """Print a formatted section header."""
        bar = "=" * 60
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)

    def _fail(self, message: str) -> None:
        """Print failure message and raise ReviewFailedError."""
        print(f"\n  [FAIL] {message}")
        raise ReviewFailedError(message)
