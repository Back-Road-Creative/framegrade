## 158 - Production video policy: what each output is and who decides it

Status: descriptive. This records what the code does. It adds no behaviour.
Paths are relative to the repository root unless stated. Where the code does not settle a point, the text says **unverified**.

Four things are kept apart on purpose: what the code does, the operator's editorial preference, measured channel evidence, and SEO claims nobody has measured.

## 1. The outputs

| Output | How its length is decided | Produced today? |
|---|---|---|
| Long-form (LF), `best_moments` mode | Score-picked highlights packed into LFs of `min_lf_duration` 480 s to `max_lf_duration` 720 s (`gomoveshift/config.py:1351`, `:1355`; `video_types.py:17-19` holds the 480/1800 constants). Packing lives in `pipeline/longform_assembler.py:384` (`group_into_regular_lfs`); GPS-dead days pack up to 1800 s (`:424-430`). The high-moment filter runs only when `best_moments_mode` is on (`:418`), and its target window is 900-1500 s (`config.py:1446-1462`). `--lf-target-duration N` pins one LF of N minutes (`cli/parser.py:467-480`). | It is the config default: `lf_selection_mode` Literal default `"best_moments"` (`config.py:1907`). Whether unattended runs use it is **unverified** (see 1.1). |
| LF, `full_drive` mode (parts) | The whole unmasked (non-stop) day, cut into parts. `plan_full_drive_parts` (`pipeline/regular_lf_planner.py:312-361`) returns one part unless BOTH `full_drive_part_min_sec` and `full_drive_part_max_sec` are set (both default None, `config.py:1914-1915`; validator `:1917-1930` needs them set together with 0 < lo <= hi). Ceiling per part: `full_drive_max_duration_sec` 7200 (`config.py:1908`). | Opt-in by `--lf-mode full_drive` (`cli/parser.py:451-461`) or config. **Unverified** which the live backlog uses. |
| Superlong (`sl_NN.mp4`) | The day's whole moving footage, split only when kept time exceeds `superlong_part_max_duration_sec` 14400 * (1 + `superlong_split_grace` 0.08) (`config.py:1936-1938`; `pipeline/superlong_planner.py`). Needs a stop mask or it raises (`day_pipeline.py:2627-2660`). The docstring says "1-3 parts" but the code has no upper bound on part count. | Off by default: `superlong_enabled = False` (`config.py:1939`). Opt-in via `--superlong`, which is rejected with `full_drive` and with `--lf-target-duration` (`cli/handlers/drive.py:626-645`). |
| Shorts | Per clip 15-55 s, target 38 s (`config.py:1366-1368`; validated so max never exceeds 60, `:2159-2174`). Count per LF is a budget capped by `shorts_target_count` 9 (a legacy cap, `config.py:1358`) and by Waypost's floor/ceiling (section 4). `state["shorts_target"]` overrides it (`day_pipeline.py:2439`); `plan_shorts` is called at `day_pipeline.py:2523`. | Always, as part of a day. |
| "Highlights" | There is no separate highlights mode in code. `best_moments` is the highlight-like mode; `--lf-target-duration` is the only "highlight LF" knob (`cli/parser.py:462`, `:478`). | n/a |

### 1.1 What production generates today

- No tracked cron or launcher in this repo starts `gomoveshift drive` for new drives. The scheduled jobs that ship here are health, journal, metrics, social-dispatch, catalog-snapshot and library-hygiene. The health job only resumes failed drives with an audit-only resume (`gomoveshift/health/heal.py`).
- A hand-started live drive is launched from the CLI with an explicit `--publish` or `--lf-day`.
- The unattended path is a nightly audit runner plus an hourly publisher, described in `docs/unattended-cadence.md`. The runner itself is outside this repo. Each backlog row carries its own `lf_mode`; the drive is audited with that mode and `register --lf-mode` stores it in `ready_drives` (field `lf_mode` in Waypost's scheduling store).
- The backlog file is tenant data and was not read for this doc. **Unverified: which mode and which part bounds the live backlog uses.**
- The operator's private config sets the part bounds and leaves `lf_selection_mode` commented out. That is operator config; its content is not reproduced here. The public fields it feeds are `full_drive_part_min_sec` / `full_drive_part_max_sec` (`config.py:1914-1915`) and `lf_selection_mode` (`config.py:1907`).

## 2. The operator's editorial preference (not a code rule, not measured)

- One LF covers one calendar day.
- Parts of about 60-90 minutes for full-drive days.
- Never pad, repeat, slow down or join footage.

These are preferences. The code enforces only parts of them: one group per calendar day (`longform_assembler.py:862`; day keys from `day_grouper._episode_date_key`, `docs/pipeline/73-day-attribution.md:3`) and no padding or repetition in the planners (section 6). The 60-90 minute figure exists only as operator config, not as a default in code.

## 3. Measured channel evidence

None recorded in tracked files. `docs/social-metrics-long-form.md:3-4` says LF duration is collected "so a later report can compare LF length with performance" and that no report reads it yet.

## 4. What Waypost decides in code

Waypost is the `seo_manager` package. GMS installs a pinned public copy (see `pyproject.toml`), so the code a checkout reads may differ from what runs; any such mismatch is **unverified**.

What Waypost's code decides:

- Shorts per LF: floor 7, ceiling 10 (`analytics/volume_policy.py`, `SHORTS_FLOOR`, `SHORTS_CEILING`; GMS reads the floor through `config.py:1336-1341`). Nightly decision rows come from `analytics/volume_decide.py`.
- LF weekdays: floor 1, ceiling 3 per region and 4 per week (`volume_policy.py`); weekday choice by median day-7 views (`scheduling/lf_days.py:43`, `:87`).
- Shorts posting hours: 8-18 (`volume_policy.py`; `scheduling/shorts_times.py`).
- Slots and cadence: `scheduling/slots.py`, `scheduling/claim.py`, `scheduling/backlog.py` (`next_row`, `shortfall`).
- Titles: `generation/youtube_generator.py:72-135` (70-character limit).

What it does not decide: LF length or part length. `analytics/evidence.py:54` records `LongformRecord.duration_sec` as evidence, but `volume_decide` and `lf_days` do not use it. `youtube/video_seo_generator.py` takes duration only as an input. So the claim "Waypost picks LF length" is false in code.

## 5. Unverified SEO claims

Nothing in the code encodes an ideal LF length. Any statement that 60-90 minute parts, or any other length, rank or retain better is **unverified**.

## 6. Duration rules

Contract: never pad, never repeat footage, never join unrelated days.

- Stops: a per-second stop mask (True = cut). `stop_mask_enabled` defaults True (`config.py:1421`). Built per day at `day_pipeline.py:1795-1804`; drive-wide for `full_drive` by `_build_drive_stop_mask` (`phase_bc_day_first.py:1513`, called near `:1475`). `full_drive` and superlong raise without a mask (`regular_lf_planner.py` `plan_regular_lfs`; `superlong_planner.py`).
- Part boundaries: `split_ranges_balanced` (`regular_lf_planner.py:210-309`) uses K = ceil(kept / max); if K > 1 and kept / K < min, K drops by one, so a part may exceed max but never falls under min. Cuts land on stop-range edges where possible. Parts tile the input with no gap or overlap and keep real timestamps; a final check raises ValueError.
- Part ceiling: a part above `full_drive_max_duration_sec` raises, "no silent truncation" (`plan_full_drive_parts`, `:312-361`).
- Insufficient footage today: in `full_drive`, kept time under 480 s raises ValueError (`plan_regular_lfs`, `:364`). A day above 480 s but under `full_drive_part_min_sec` gets K <= 1, `split_ranges_balanced` returns the ranges unchanged, and ONE short part ships with no warning.
- In `best_moments`, an undersized tail goes through `_rebalance_undersized_tail`; if still short it ships with a `logger.warning` and `plan.undersized = True` (`longform_assembler.py:531`, `:543`). Nothing reads `undersized` afterwards (`models.py:306`, `:320`, `:336`). When the adaptive threshold search exhausts its floors it returns the best available with a WARNING (`longform_assembler.py:682-`).
- A day with no scores is skipped with a WARNING (`longform_assembler.py:879-880`, `:1249-1250`).
- Legacy non-day-first path: `_build_full_drive_plan` (`longform_assembler.py:1338`, called from `:991-992`) collapses a whole drive into one LF and raises if more than one part results. This would join days. **Unverified** whether it is reachable.

### 6.1 Pending (not shipped)

A planned change ("LF policy resolves by construction") adds an `InsufficientFootageError`: below the part minimum the run refuses by default. There is no escape flag; the operator has not approved one. Until it ships, the silent short part above describes production. A related compatibility migration, inferring the mode from the part bounds, is also pending; this doc describes neither as shipped.

## 7. Trace: decision to payload

1. Decision: operator or backlog row picks `lf_mode`; part bounds come from config.
2. Config: `DriveConfig` in `config.py` (`:1351`, `:1355`, `:1907`, `:1914-1915`). `min_lf_duration` exists on `DriveConfig`, not on top-level `Config` (`:2251`).
3. Launcher: `--lf-mode` (`cli/parser.py:451-461`, default None = use config); `cli/handlers/drive.py:587-589` copies it into config; `console/drives.py:38-39`, `:93-94` append `--lf-mode` only if the form set it.
4. State: `pipeline_state.json` (`pipeline/orchestrator.py:204`); `drive_plan` is a field on the state (`models.py:553`), not a separate file. The CLI-to-state registry is `pipeline/flag_registry.py` (`apply_cli_to_state` `:122`, `merge_on_resume` `:147`). It has no row for `lf_mode`, `lf_selection_mode` or `lf_target_duration` (rows `:67-110`), so a resume without `--lf-mode` falls back to config, not to the saved mode.
5. Planner: `regular_lf_planner.py` (`plan_regular_lfs`, `plan_full_drive_parts`, `split_ranges_balanced`); `superlong_planner.py`.
6. Render: `longform_assembler.py` (`build_drive_plan_per_day` `:779`; `_build_full_drive_plan_per_day` `:1213-1335`) and `day_pipeline.py:1894-2028` (full_drive plan synthesis, per-part checks `:2001-2018`, more than one plan without parts raises `:2020-2025`).
7. Payload: `longform_metadata.py:1670-1690` (`_apply_title_decision`, part suffix), `phase_d_metadata.py:266`, `:432`, `:464` (part_count > 1 cases), `youtube_uploader.py:912-980` (title_decision check), `publishing/plan_builder.py`.

### 7.1 Override points

| Point | File:line | What it can change | Logged? |
|---|---|---|---|
| `--lf-mode` | `cli/handlers/drive.py:587-589` | LF selection mode for this run | No log line found. Not saved to state or restored on resume (no flag-registry row). |
| `--lf-target-duration` | `cli/handlers/drive.py:600-619` | Turns on `best_moments_mode`, pins target min/max, may widen `max_lf_duration` to 1.2x or lower `min_lf_duration` | No. Mutates config silently. Errors for bad combos are printed (`:593-599`). |
| `--superlong` | `cli/handlers/drive.py:626-645` | Enables superlong | Rejections printed; enable itself is a SENTINEL_CLI_WINS registry row (`flag_registry.py:94-95`). |
| Console form | `console/drives.py:42-100` | Appends `--lf-mode`; default dry-run | Argv only. |
| Backlog row `lf_mode` | `scheduling/backlog.py`, `store.py` | Mode used by audit and publish | Recorded in `ready_drives`; row content **unverified**. |
| Operator config part bounds | `config.py:1914-1915` | Splits full_drive days into parts | Not logged. Private values not shown here. |
| `stop_mask_enabled`, `skip_stop_verify` | `config.py:1421` | Whether stops are cut | Not logged as an override. |
| Part min read by the render step | `day_pipeline.py:1919` | Reads `min_lf_duration` from top-level `Config`, which has no such field, so it always uses the default 480 and ignores `config.drive.min_lf_duration` | No. Values match by default; latent mismatch. |
| Undersized best_moments LF | `longform_assembler.py:531`, `:543` | Ships a short LF | Warning only; no gate. |
| Day under part minimum (full_drive) | `regular_lf_planner.py:210-309` | Ships one short part | No. |
| Adaptive floors exhausted | `longform_assembler.py:682-` | Returns best available | WARNING. |
| Day with no scores | `longform_assembler.py:879-880`, `:1249-1250` | Skips the day | WARNING. |
| Resume with STATE_WINS conflict | `flag_registry.py:147` | CLI value ignored in favour of state | Logged: "Resume CLI override conflict". |
| Planner hard stops | `regular_lf_planner.py` | Refuse bad plans (kept under 480, part over ceiling, no mask) | Raise ValueError (not an override). |

## 8. Human gates in production

The OST audit gate is a human gate. Before the irreversible burn, a drive writes `_stages/ost_audit_<date>.json` and stops until one of these is present:

- a matching `ost_audit_<date>.json.confirmed` file, or
- `--ost-audit-confirmed` passed to the run (`cli/parser.py:834`).

Today no tracked launcher, cron or console form passes `--ost-audit-confirmed` (`bin/`, `console/`, `health/`, `scheduling/` searched). Nothing confirms the gate automatically, and nothing may.

**Who confirms in production (operator decision).** Routine on-screen text is to be checked by a Waypost-owned automated review. A person reviews only text whose location evidence is missing, unresolved or conflicting. This approves building that review path. It does not approve adding `--ost-audit-confirmed` to any launcher.

The automated approval must:

- cover the exact long-form and Shorts burn text;
- match verified location evidence and the intended output;
- be voided by any material change to the text, the evidence or the output;
- be tested for missing evidence, a wrong location, changed text and resume.

Until that path is built and qualified, the existing confirmation gate stays. Any review it needs goes to the operator as one consolidated, readable packet, not as raw JSON files. `tests/test_policy_ost_review_and_plate_scope.py` fails if any launcher, cron or scheduler passes `--ost-audit-confirmed`.

Observed defect (on an older checkout): a run that stopped at this gate, with zero LFs and zero Shorts, still printed "Drive pipeline complete!" and exited 0. The fix is in the LF policy and launch-visibility change: a run held at a human gate prints `DRIVE WAITING` with the gate, the file to review and the next action, and exits non-zero. Until that change ships, an exit code of 0 does not mean the drive produced outputs.

## 9. Plate scope (operator decision)

Every licence plate is in scope, whatever its apparent readability. That includes identified plates about 15 px wide. There is no unreadable-plate exemption, and none may be added.

- The plate path has no size floor today. Plate boxes are filtered only by detector confidence, the aspect band (`blur_plate_min_aspect` / `blur_plate_max_aspect`) and an oversized-box cap (`blur_plate_max_area_frac`). The vehicle height floor `blur_vehicle_min_height_frac` gates only the occupant (cabin) blur, not plates.
- Plate size is always stated at a named resolution, for example "15 px wide at 1920x1080".
- Acceptance inspects the final output, including Short crops, thumbnails and other derived images, not only the day master or the attestation.
- Tests cover small plates and moving plates.
- Reports keep two questions apart: detector coverage (was the plate found?) and blur application (was a found plate blurred in every frame it is visible?). Detection misses are reported as misses, never folded into a pass.

`tests/test_policy_ost_review_and_plate_scope.py` checks that a 15x6 px plate survives the plate filters at 1080p, at 4K (tiled) and in a 1080x1920 Short frame. It also checks that `DriveConfig` has no plate size-floor field.
