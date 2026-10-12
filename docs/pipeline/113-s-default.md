### `enqueue-plan`'s default `--output-dir` (`cli/handlers/enqueue_plan.py`)

Without an explicit `--output-dir`, the default resolves from the run state's own `output_base`
field (the project directory Phase D wrote `*_publish_plan.json` sidecars into) rather than
`pipeline_state.json`'s parent directory — the two can diverge whenever `stages_base` is
relocated off the output mount (`GMS_INTERMEDIATE_DIR`, or automatically for a drvfs-rooted run).
A legacy state with no `output_base` falls back to the state file's parent, or its grandparent
when the parent is literally named `_stages`. A no-sidecar run (every route off) still
exits 0, but the message now names the directory searched and how to point elsewhere.
