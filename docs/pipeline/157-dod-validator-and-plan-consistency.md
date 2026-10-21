## 157 - DoD validator gate and publish-plan consistency (GMS-17c)

**Terminal DoD validator.** Every drive run that finishes its phases calls
`run_validation(mode=ValidationMode.AUTO)` (`drive_orchestrator._run_dod_validation_gate`).
AUTO resolves to `POST_UPLOAD` unless the run is `skip_upload`
(`tools/validate_pipeline_output.derive_mode_from_state`). Any failed check, or a validator
crash, raises `PipelineValidationError` outside the phase loop's `try`, so the CLI exits
non-zero. Tests: `tests/test_orchestrator_validation_gate.py`.

**`--skip-validation` fails closed.** The flag (`config.drive.skip_validation`) is honoured
only on a `--skip-upload` run (an audit or preview pass), with a WARNING. On any other run
it raises `PipelineValidationError` naming `--skip-validation`, before the validator would
run: a skipped check never reads as a PASS on a run that uploads. A test that needs to
isolate the phases from the gate stubs a passing `run_validation`, never the flag.

**Publish-plan consistency on load.** `plan_io.load_publish_plan`, the one reader behind
`enqueue-plan` (dry run and `--apply`) and `dispatch-queue`'s `SidecarRecordLoader`, checks
each short after rehydrating it and raises `PublishPlanFormatError` naming the file:

- every hashtag in the YouTube title appears in that short's description (the builder
  takes title hashtags from the description, so a mismatch means they were edited apart);
- no surface (title, description, tags, captions, TikTok title) names a county outside the
  clip's `sampled_regions` (`plan_gates.assert_shorts_metadata_in_region`);
- no surface carries a preposition splice (`plan_gates.assert_plan_text_grammar`).

The two `plan_gates` checks are the same ones `build_publish_plan` runs at build time, read
through `plan_io._gate_view`. Running them again on load catches a plan that was edited or
written apart from the builder. Tests: `tests/test_publishing_plan_io.py`.
