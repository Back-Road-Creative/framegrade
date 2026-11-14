### Fixed

- TikTok inbox mode no longer refuses every upload after a switch from pack mode. The 3-draft pending cap now counts only `awaiting_manual_publish` rows with a `platform_upload_id` (real inbox drafts). Pack-parked rows, which never reached TikTok, no longer fill it.
