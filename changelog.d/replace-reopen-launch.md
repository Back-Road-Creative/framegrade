### Fixed

- `health backlog --replace-scheduled` can now replace the scheduled uploads of a drive whose
  schedule-store row is `complete`: Waypost's `reopen_for_replace` is asked first (dry run, before
  any rename) and then called for real in place of `register`; a `claimed` row, or a `complete`
  row with no replace plan, is still refused.
- `health publish` refuses a replace whose re-render split into a different number of long-forms
  than the replace plan covers, leaving the store's long-form count unchanged.
- The launch anchor gate accepts the past `--lf-publish-date` of a replace run, but only when its
  `replace_plan.json` keeps an LF1 published on that date.
