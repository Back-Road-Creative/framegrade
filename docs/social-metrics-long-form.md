# Waypost long-form tracking

`tracking/` used to follow shorts only (`TrackingRecord.shorts`). Long-forms are now tracked in
`TrackingRecord.long_forms` (`schema.LongFormRecord`) so a later report can compare LF length with
performance. This page covers collection only; no report reads it yet.

## Registration

No upload hook. `TrackingStore.sync_long_forms_from_ledger(entries)` enumerates the content
ledger: every `longform` entry with a `video_id` and `project_name` (a ledger entry exists only
once an upload is confirmed) becomes a `LongFormRecord`. Identity is the
`video_id`: sync and `register_long_form` look a record up by it and never overwrite another
record's `video_id`, duration or metrics. The ledger stores no LF index, so a new id is offered
its 1-based date position within the project and gets `max(existing)+1` when that slot is taken
(a backfilled earlier entry therefore cannot displace a known one). Idempotent on `video_id`.
`register_long_form` is the single-record primitive.

## Collection

`MetricsCollector.collect_long_forms(record)` calls `YouTubeCollector.collect_long_form_metrics`
for each LF: the Data API `videos.list` call already made for shorts (its ISO duration was parsed
then discarded; it is now persisted as `duration_sec`, `duration_source="youtube_data_api"`), the
lifetime Analytics query, and the per-day query (28 days from the booking date).

- Per-day points keep `avg_view_duration_seconds` and `views`; the lifetime query adds
  `averageViewPercentage` (`PlatformMetrics.avg_view_percentage`). The extra column is requested
  on the long-form path ONLY (`include_avg_view_percentage`); the shorts query text is unchanged.
  A short's per-day rows also keep `avg_view_duration_seconds`.
- A long-form whose only reported number is `averageViewPercentage` counts as measured.
- A metric the API does not report is `None`, never 0. An unknown duration is `None`, never 0.
- A collection that returns no numbers never overwrites a prior measurement, and a missing
  duration never overwrites a known one.
- No new endpoints: the same YouTube Data / Analytics calls the collector already makes.

Shorts tracking is unchanged. Records written without long-form data load with `long_forms == []`.

## Length evidence (no report, no decision yet)

`gomoveshift lf-length-report` and `tracking/length_evidence.py` were removed: the
pipeline has no long-form length knob, so a report-only length comparison had nothing to feed
and nothing read it. Each long-form row in `gomoveshift metrics export-evidence` carries
`duration_sec` and `avg_view_percentage`, so a later Waypost length decision has its data; that
decision is the named follow-on and needs the knob first. The `drive.lf_length_*` config fields
were removed too.
