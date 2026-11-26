## Owner lines: one sentence in the owner's own voice

Every caption and description GMS writes is composed from facts (place, road,
date, chapters). That keeps the text honest, but it reads the same post after
post and carries nothing only the owner could say. An owner line is one short
sentence the owner writes for a post, which opens that post's text.

**The file.** `<project_dir>/social/owner_lines.yaml`, beside the existing
`titles.yaml` hand-title file ([135](135-shorts-titles-and-interactive-family.md)):

```yaml
shorts:
  3: Stopped here for chips, best in the county.
longform:
  1: My favourite stretch of the whole trip.
```

`shorts` is keyed by the Short's global index (the same key as `titles.yaml`),
`longform` by the LF index. Both sections and the file itself are optional.
`pipeline/owner_lines.py` loads it: a line is folded to one line of at most 100
characters, with no link and no hashtag (those stay the pipeline's own). A
malformed file is a hard `ValueError` naming the path and the key, the same as
`load_hand_titles`.

**Where a line goes.**

- A Short's line opens its Facebook and Instagram captions as their first
  sentence (`SocialGenerator._build_platform_captions`), ahead of the composed or
  hand text, so the place sentence and its checks are unchanged. The YouTube
  caption keeps its composed text, and TikTok's 150-character caption has no
  room. An `interactive` Short's line must not name the place, like its hand
  caption.
- A long-form's line opens its YouTube description as the first paragraph
  (`phase_d_metadata._build_metadata_per_lf`), before Waypost's review of the
  final text, so the review and the upload re-check cover the line too. A line
  that names a place the drive's route does not back is refused there, like any
  other unbacked place claim.

**When to write it.** The line is read when the drive's text is generated, so it
has to be in the file before Phase D builds the text that gets approved. A line
added after approval changes nothing already reviewed; regenerate the drive's
text to pick it up.
