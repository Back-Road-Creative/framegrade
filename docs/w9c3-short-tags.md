# Short tags limited to what Waypost can review

A Short's YouTube tags are now limited to three things, all of which Waypost
reviews: its own tag templates, the clip's own county names, and the clip's own
GPS road names. Long-form (LF) tags are unchanged.

## Data flow (clip roads)

1. `content_gates._make_short_regions_fn` votes `dominant_location` over the
   clip's own trimmed GPS; `_VotedRegions.roads` holds that clip's road votes,
   most-voted first. Nothing is read from the parent LF.
2. `day_pipeline` writes `roads` into the per-short OST audit sidecar entry.
   Roads are not burned text, so they are in `_HASH_EXCLUDED_ENTRY_KEYS` (the
   OST confirmation hash is unchanged).
3. `phase_d_scheduling._per_short_sidecar_roads` reads them back by rank and
   passes `per_short_roads` to `SocialGenerator.generate_all`, which sets
   `SocialPost.sampled_roads` and builds the Short's tags.
4. `plan_builder` stores them as `ShortPublishRecord.sampled_roads`; `plan_io`
   loads plans written before the tag rule as `()`.
5. `youtube_adapter` turns each into `kind=road` evidence
   (`source=clip_gps_road`); a road tag cites it. The plan fingerprint includes
   the sorted roads, so a changed road invalidates a saved review
   (`stale:` refusal).

A clip with no GPS road emits no road tag, and a road tag for it is refused
(`tag_unreviewable:<tag>`, insert count 0). Roads are never invented.

## Dropped for Shorts only

`social_generator._build_tags` keeps, for a Short with its own clip regions,
only `reviewable_short_tags(...)`: Waypost tag templates, the clip's county
names and its road names. Measured over short indexes 1-12 on a synthetic clip,
the tags dropped (previously published on Shorts) are: Pellmere (the label
tag), ambient, asmr driving, calming drive, car ride, dashboard camera, dashcam
footage, drive footage, drive life, driving, explore, first person, highway,
highway drive, nature, no talking, on the road, open road, pov car, pov road
trip, road adventure, road therapy, road trip vibes, slow travel, travel by
car, travel vlog, weekend drive, window view; plus the weather tags (sunny
drive, rain driving, rainy day and the like) when a weather result exists.
Kept: the Waypost templates (POV driving, drive, drive with me, driving route,
driving tour, long drive, relaxing drive, road trip, virtual travel, dashcam),
the clip's county and the clip's roads.

This narrows Short tags. It is a discoverability tradeoff, to be revisited by
adding Waypost tag templates, not by bypassing the review in GMS.

## Still refused

- POI: there is no per-clip POI in the record and no Waypost POI evidence
  kind, so a POI tag is refused. A Waypost `poi` kind is a separate change.
- Any other tag: `tag_unreviewable:<tag>`.

Tests: `tests/test_w9c3_short_tags.py` (including LF tags unchanged).
