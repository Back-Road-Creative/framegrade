### Route Story audio and final encode (`assembly.py`, `encode.py`, `beats.py`)

<a id="route-story-film-audio"></a>
A Route Story film follows the service's near-silent, attenuate-only audio rule like every other
published video. A source clip keeps its own audio in place; map and photo legs carry silence
(padding only -- nothing is generated and no music is bundled), and every clip shares one audio
layout so the concat stays in sync. Every part (frame sequences, the placed clip, the recap) is
encoded with one shared spec, `assembly._PART_VIDEO_ARGS` (x264 crf 8 veryfast, high profile,
BT.709, fixed timescale), so `_concat` stream-copies the video and re-encodes only the audio: the
parts are visually lossless scratch files and `encode_final` is the film's single lossy generation
(a frame once went crf 23 part → crf 23 concat → crf 18 delivery, and the concat spent
a whole-film encode pass). The recap map is rendered once per film; each recap frame is a copy with
its own fade of the numbers drawn on. `encode.encode_final` runs once on the assembled cut:
delivery-quality single pass, BT.709 tags written into the stream itself (the container-level flags
alone leave a probe reading "unknown"), and one audio gain from `encode._attenuate_filter` -- the
long-form measurement pass feeding `loudnorm_measure.build_attenuate_filter`, so the gain is never
positive and a failed measurement falls back to the fixed attenuation. The film used to run a bare
`loudnorm`, documented here as a "deliberate exception" no operator had asked for: it normalises
TOWARD a target, and lifted the example-route phone clip from -51.4 LUFS going in to -46.0 LUFS in
the film. Audio already under the ceiling is now left exactly as recorded. The output rate is
pinned to the clips' own 48 kHz stereo (`assembly._AUDIO_CODEC_ARGS`); the `loudnorm`-era film
shipped 96 kHz without it. The chapter list has ONE source, `beats.chapter_rows`; `beats.write_chapters_vtt` and
`encode.write_publish_json` both mirror it, so `chapters.vtt` and `publish.json` cannot disagree.

`encode_final` is a single fixed delivery pass; `encode.encode_rendition(master, platform, output)`
is a separate, table-driven encoder for a platform-specific derivative cut of a master -- the drive
shorts ladder is one such derivative (`encode_rendition(master, "youtube_short", out)`). The
route-story **social** cut is NOT a derivative: `film.main` renders it natively at its preset's
9:16 size and delivers it with the SAME `encode_final` call as the `full` preset -- there is no
second encode. `--preset social`'s `rendition="route_story_short"` names which row of
`encoding_constants.RENDITIONS` is this cut's rendition of record purely for after-the-fact
verification (`receipt["preset"]["rendition_check"]`, computed from the same `check_delivery`
measurement the receipt already makes): audio present, sized right, under the ceiling. Each
platform's shape lives in `encoding_constants.RENDITIONS` as a frozen `Rendition` (width, height,
max_seconds, max_bytes, fps, codec, faststart, audio); a field is `None` when this file has no
constant for it yet -- the table never invents a platform ceiling. `youtube_short` mirrors the
legacy `SHORTS_WIDTH`/`SHORTS_HEIGHT` (1080x1920), `hevc` codec (`hevc_nvenc`), faststart,
and no audio stream (that is the DRIVE shorts rule, not a route-story one); `route_story_short`
shares the same 1080x1920 frame but keeps audio (this cut's audio is
attenuated dialogue, not cabin audio, so the drive-shorts `-an` rule does not apply to it) and sets
`max_seconds=90.0` to mirror `route_story.presets.SOCIAL.max_seconds` exactly. `youtube_longform`
keeps audio and leaves width/height `None` because long-form resolution follows the source (both a
4K and a 1080p bitrate floor exist -- there is no single fixed frame size to put in the table).
`encode_rendition` probes the master's duration and raises `RenditionCeilingError` before invoking
ffmpeg at all when a rendition's `max_seconds` is set and the master exceeds it -- `route_story_short`
is the first row that exercises this path; the drive-shorts and long-form rows still leave it `None`.
A social delivery missing its audio stream is a `check_delivery` issue like any other off-spec
film (exactly-one-audio-stream is enforced for every preset, never conditioned on `rendition`), so
that failure already takes the run's existing non-zero exit path.

`encoding_constants.PAYLOAD_LIMITS` is `RENDITIONS`'s sibling table for the OTHER half of a
delivery -- per-platform payload TEXT limits (title/description/tags/caption length), checked by
`publishing.models.validate_payloads` at `enqueue-plan` time rather than at encode time. See
`docs/social-publishing-queue.md`'s enqueue-gates list.

### Licensed music bed and per-brand audio profile

`pipeline.audio_profile.load_audio_profile(slug, brands_dir=...)` reads an optional `audio:` block
from a brand's `<brands_dir>/<slug>/brand.yaml`, beside `publishing.brand_policy`'s `policy:` block.
The engine never imports publishing code (`tests/test_media_boundary.py`), so the orchestration-side
caller passes the shipped-brands directory (`brand_policy._SHIPPED_BRANDS_DIR`) in. A missing
file/block/key falls back to a disabled `AudioProfile` (no bed). A present block sets `enabled`, `bed_path` (resolved relative to the brand's directory),
`gain_db` (default -18.0) and `fade_out_s` (default 2.0), and an optional `ledger_path` (defaulting
to `<brand-dir>/music/ledger.json`).

**GoMoveShift's slug is refused unconditionally** -- `load_audio_profile("gomoveshift", ...)` raises
`GmsAudioRefused` before any file is even read. Its audio contract, described at the top of this file, is near-silent and attenuate-only; a music bed is a different thing and no `brand.yaml` can turn it
on.

A bed is only ever burned when its exact file has a matching entry in the brand's
`LicenceLedger` (`music/ledger.json`: a JSON `{"entries": [...]}` list, each entry `{file, title,
source, licence, url, added}`). `resolve_bed(profile, ledger)` looks the bed up by filename and
raises `UnlicensedBedError` when it's missing -- checked, and enforced, BEFORE any ffmpeg call at
all (including the loudness measurement pass), so an unlicensed bed can never even trigger a
read-only probe of the source. The ledger's `source` field is free text -- the seed default is
`"pixabay"` (Pixabay Music, Pixabay Content License) -- so adding a different music
library later needs a new ledger entry, never a code change.

When `encode_final(source, output, audio_profile=..., ledger=...)` resolves a bed, the ffmpeg
command adds the bed as a second, looped input (`-stream_loop -1`) and mixes it under the
already-attenuated source via `-filter_complex`: the bed is gained by `gain_db` and faded out over
its last `fade_out_s` before `amix`'s `duration=first` keeps the mixed track no longer than the
source. Omitting `audio_profile` (or passing one with `enabled=False`) leaves the ffmpeg command
byte-identical to the no-bed path. `encode_final` returns the music-bed receipt
(`{file, ledger_entry, source, licence}`) for the caller to hand to `write_publish_json`, which
always writes a `music_bed` key into `publish.json` -- the resolved receipt, or `null` when no bed
was mixed in -- so a published film's own sidecar can prove whether it carries licensed music and,
if so, under what licence.
