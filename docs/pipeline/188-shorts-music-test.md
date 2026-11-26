## Licensed music on Facebook and Instagram Shorts: a 50/50 test

GoMoveShift Shorts are rendered muted (`shorts_extractor`, `-an`) so that no cab
conversation can ever be heard, and the Shorts validator still refuses a master with
an audio stream. That rule is unchanged. The owner asked (2026-11-26) to test music
on Facebook and Instagram Shorts, so `publishing/shorts_music.py` adds a licensed
track at upload time, to those two platforms only, for half the Shorts.

**Declared in `brand.yaml`.** `policy.shorts_music` takes `start`, `stop_after_days`
and `platforms` (a subset of `facebook` and `instagram`; nothing else is accepted).
With no block, the Meta readers in `dispatch-queue` are the plain muted readers.

**Which Shorts.** `music_arm(project, short_index)` is a stable hash, so a Short is
in the same half on both platforms and on every retry. Only Shorts (format `short`
or unstated) are considered, and only while the test runs.

**Which track.** Tracks live outside the public tree: `GMS_MUSIC_DIR`, or by default
`music/<slug>/` beside the host brands dir (`GMS_BRANDS_DIR`, so
`_private/svc-gomoveshift-video/music/gomoveshift/` on the host). A track is used only
when its exact file has a complete entry in that folder's `ledger.json`, the same shape
as Route Story's licence ledger (doc 26): `{"entries": [{file, title, source, licence,
url, added}]}`. A Short's track is a stable hash over the licensed tracks. With no
licensed track, every Short posts muted and none is counted in the test.

**The upload.** `ensure_music_copy` probes the file first and refuses one with ANY
audio stream (or one it cannot probe): a bed is never laid over real sound. It then
copies the video stream untouched and adds the track as the only audio (loudness
-16 LUFS, 0.5 s fade in, 1.5 s fade out, AAC 128 kb/s 48 kHz stereo, inside Meta's
Reels spec), as a sibling `<stem>.music-<tag>.mp4` that is reused. Instagram's copy
is made from its 25 Mb/s rendition. The master on disk stays muted; YouTube and
TikTok, long-form drives and Route Story films (`GmsAudioRefused`) never get a bed.
Any failure posts the muted file and logs why; music never blocks a post.

**Measuring it.** Each Short that actually went out with a track appends a line to
`music_receipts.jsonl` in the SEO storage dir (project, short, platform and the
track's ledger licence). `metrics growth-review` (doc 186) adds a `music:on` /
`music:off` group to each Facebook and Instagram sample posted during the test, so
its `GROWTH_FORMAT` lines compare day-7 views with and without music. A music-half
Short with no receipt is left out of both groups, never counted as silent.

**Risks the test cannot remove.** A royalty-free track can still be matched by
Meta's rights tools if its author registered it; the receipt names the track so a
claim can be traced. Popular commercial songs are never added here.
