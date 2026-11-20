### Fixed

- A Short's `videos.insert` now carries `recordingDetails.recordingDate`, the measured filmed-on date on its publish record (a blank date sends nothing), as long-form and superlong uploads already did.
- Every upload sends `snippet.defaultAudioLanguage`, from the same config value as `defaultLanguage` (`drive.youtube_default_language`).
