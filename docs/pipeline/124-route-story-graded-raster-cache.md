## Route Story graded-raster cache (`gomoveshift/pipeline/route_story/layers.py`)

`_graded_raster_image` applies the still-photo grade to every opening raster. For a typical
bundle that is 14 rasters / 298 MP at ~2.7 s per MP, so **~13.5 minutes before the first frame
of every render** (a stack sample showed `grade.py` as essentially the whole pre-render stage), and
again whenever the eight-entry memory cache evicted a raster.

- The graded pixels are a pure function of the raster's **bytes** and of `grade.py`, so they are
  cached on disk as `.npy` keyed on `sha256(raster bytes + sha256(grade.py))`. Content, not
  path: the bundle's opening rasters are re-written into a fresh temp dir every render.
- Location follows the frame cache: `$GMS_INTERMEDIATE_DIR/route-story-grade-cache`, else
  `route-story/graded-rasters` under the cache home (`$GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`,
  else `~/.cache/gomoveshift`; see `paths.py`), resolved per call, never at import.
- Capped at `GMS_ROUTE_STORY_GRADE_CACHE_MAX_GB` (default 8; one bundle's set is ~0.9 GB),
  oldest first. `GMS_ROUTE_STORY_GRADE_CACHE=0` turns it off. An unreadable entry is regraded;
  a cache that cannot be written makes a render slower, never failed.
- The test suite runs with the cache off (`tests/conftest.py`); identical synthetic rasters in
  two tests would otherwise share an entry.
- **Photo stills share the same cache** (`graded_cache.py`). `hud._load_still` graded the full
  20.7 MP camera still on every call (53–63 s each), and a render loaded each kept
  photo twice (selection, then its beat): ~27 min per render. The key adds the still source
  (`developed`/`camera`) and the identity of `grade.py`, `hud.py` and `photos/develop.py`, so any
  edit that could change a still's pixels regrades it. `still_selection.load_stills` grades the
  candidates in a spawn process pool (`GMS_ROUTE_STORY_STILL_WORKERS`, default 4) and returns them
  in input order; one worker or one photo stays in-process.
