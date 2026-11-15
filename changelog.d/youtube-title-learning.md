### Added

- YouTube titles now read YouTube performance. The nightly `metrics collect` joins each
  shipped title's chosen template to its video id (`social_posts.json` for a Short,
  `youtube_uploads.json` for a long-form), asks Waypost's `title_evidence` for each template's
  thumbnail click-through from the collected reach rows, and writes `title_evidence.json`
  beside the queue. Every shipped-title decision (Shorts, long-forms, the pre-render
  long-form title, superlong) now passes that file to Waypost's `select_title` instead of an
  empty snapshot. No reach store yet, or no file, keeps today's behaviour: the baseline ships
  as `insufficient_evidence`. A title moves only once the baseline and a non-baseline template
  each have 400 impressions of their own; today only the baseline ships, so no alternative can
  earn evidence until exploration exists (a separate decision). Pins Waypost to #114's branch
  head.
