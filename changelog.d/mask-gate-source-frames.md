### Fixed

- A day whose raw and master were purged can now prove its blur mask to the publish gate: the pairing check persists the proven source frame count in `day_<date>_source_frames.json`, and `mask_frame_gate` accepts it only while the sidecar and mask still match it. Anything missing, stale or unreadable stays `mask_frame_count_unknown`; nothing existing is relaxed.
