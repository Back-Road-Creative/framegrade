### Tracking store default root (`tracking/store.py`)

`TrackingStore()`'s default `root` resolves lazily to `gomoveshift.paths.gms_output_dir()` at
construction time, replacing a hardcoded host path nothing wrote to. Phase D's post-upload hook
(`drive_phases/phase_d.py::_collect_metrics_best_effort`) receives `output_base` = the *per-project*
directory (`<GMS_OUTPUT_DIR>/<project>`), and the store appends `<project>/tracking` itself, so the
hook roots its store at `output_base.parent` — the same root `metrics collect`/`velocity`/`report`
resolve by default. It used to use `output_base` directly and looked under
`<GMS_OUTPUT_DIR>/<project>/<project>/tracking`, where nothing is written.
