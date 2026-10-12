## Proxy media and camera metadata <a id="proxy-media-and-camera-metadata"></a>

**Gap:** the drives review gate
(`console/drives.py`) served only the full-weight source clip, and per-clip
ingest metadata carried nothing from the camera itself — only what ffprobe's
codec/resolution probe (`pipeline/video_detect.py`) reports.

### Camera model / firmware

`pipeline/ingest/gopro_parser.extract_camera_metadata(video_path)` reads
`ffprobe -show_format`'s `format.tags` and returns `(camera_model, firmware)`,
trying a short list of known container-tag spellings
(`_CAMERA_MODEL_TAG_KEYS` / `_CAMERA_FIRMWARE_TAG_KEYS`). Either or both come
back `None` when the container carries no recognized tag, or on any ffprobe
failure — never invented, never raised.

`scan_video_directory`'s GoPro-mode path calls it per chapter and stores the
result on `GoProChapter.camera_model` / `GoProChapter.firmware`, alongside
the chapter's existing `duration` / `creation_time` fields — the per-clip
record this ingest pass already builds.

This is a container-tag read, not a GPMF binary parse: it does not touch the
GPMF `DVNM`/`FIRM` telemetry stream the way `gopro_gps.py`'s GPS extraction
does. If a camera's model/firmware is only present in GPMF telemetry and
never mirrored into the container's udta tags, this path returns `None` —
deferred.

### Review proxies

`pipeline/proxy_media.build_review_proxy(source, stages_dir)` renders a
640px-wide, muted (`-an`), fast-start (`+faststart`) H.264 proxy at
`stages_dir/proxies/<stem>_proxy.mp4` via `ffmpeg`. `stages_dir` is the
caller's staging/stages root (e.g. `gms_intermediate_dir(...)`) — never
`/tmp`, never `--output-dir`.

Proxies are strictly optional: any `ffmpeg` failure, non-zero exit, or empty
output logs a `WARNING` and returns `None`; nothing calling it hard-fails.

`console/drives.resolve_gate_video_media` prefers a proxy over the original
when one sits alongside the resolved video (`<stem>_proxy<suffix>`) — same
containment rule as the original lookup, so a request still only ever
indexes into the gate's own artifact list; falls back to the original file
when no proxy exists.

**Deferred (not in this change):** wiring `build_review_proxy` into the ingest
pipeline itself so a proxy is actually produced (and lands next to the
original artifact) during a normal drive run — this change ships the builder
and the gate's preference for it, not the call site that invokes it during
ingest. GPMF-only camera metadata (cameras that don't mirror model/firmware
into container tags) is also out of scope; see above.
