### Fixed

- A label changed on resume (bracket relabel or a corrected `--placeless-label`) now re-burns the day's long-forms and Shorts. The relabel cleared the day's completion record, but the recovery hook re-added it from the files on disk seconds later, so the day was skipped and every final kept the old burned text (rec0063: 3 long-forms and 21 Shorts still read "United States"). Such days are now recorded in `state.reburn_days` until they complete again, and the Shorts audit confirmation of the old text is cleared too.
