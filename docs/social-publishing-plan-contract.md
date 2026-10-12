# Social-publishing plan contract (`gomoveshift/publishing/`)

Models-only package: immutable publish-plan records plus a deterministic
content hash. No dispatcher, no network call, no mutation path; `plan_builder.py` and
`phase_d.py` produce these records and the queue consumes them.

## Models (`gomoveshift/publishing/models.py`)

One frozen `ShortPublishRecord` per short:

- identity: `project_name`, `lf_index`, global `short_index`, local
  canonical `media` (`MediaFingerprint`: path, size, duration, dimensions,
  codec, sha256);
- `schedule` (`ScheduleWindow`): scheduled instant in UTC + display timezone;
- `recording_date` + `sampled_regions`;
- `parent_lf_video_id` + `parent_lf_publish_at`;
- `payloads` (`PlatformPayloads`): separate typed structures per platform —
  YouTube (title/description/tags/category), Instagram (caption), Facebook
  (description), TikTok (title + privacy level + disclosure flags). No
  shared catch-all caption at dispatch time;
- `consent_hash` + `reviewed_plan_hash`;
- `routes` (`RouteSelection`): subset of `{youtube, instagram, facebook,
  tiktok}` selected for delivery — each platform route is `first_party` or
  `off` (`gomoveshift/config.py`); Metricool is retired, there is no
  rollback publisher. Empty `selected` raises at construction;
- `generated_at`: wall-clock bookkeeping only, excluded from the hash below.

`PublishPlan` wraps a tuple of `ShortPublishRecord`. Every model is
`@dataclass(frozen=True, slots=True)` — mutation raises
`dataclasses.FrozenInstanceError`.

## Hashing (`gomoveshift/publishing/plan.py`)

`compute_plan_hash(plan)` mirrors the exclusion discipline of
`gomoveshift.pipeline.review_gate.compute_gate_cd_hash`: sha256 over
`json.dumps(payload, sort_keys=True)`, excluding `generated_at` so
re-running the plan builder on identical inputs yields an identical hash.
Everything else — media identity, per-platform payloads, schedule,
parent-LF linkage, consent hash, route selection — participates.
