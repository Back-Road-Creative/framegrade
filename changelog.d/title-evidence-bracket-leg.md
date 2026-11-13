### Fixed

- Phase D no longer refuses a GPS-less long-form titled with its bracket leg ("Virginia to Pennsylvania in 4K"): the title-evidence gate read the two-state label as town endpoints and found no location-evidence record. A valid `states`-level bracket record now backs exactly its own leg label; any other "A to B" title with no record is still refused (nctopa-2026-03-05-rec0063).
