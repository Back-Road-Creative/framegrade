### Fixed

- A multi-LF day no longer ships an LF with no shorts. The day-wide shorts planner could put every top-scoring window in one LF's footage; the per-LF cap then trimmed that LF and nothing refilled the others. `fill_shorts_to_per_lf_capacity` now plans each under-capacity LF's missing shorts from that LF's own source window (stop mask and GPS-gap veto kept, other LFs' footage and already-picked shorts masked). The SHORTS-CONTRACT gate now judges the final post-cap, post-refill count, so an LF whose window truly cannot hold its share is refused unless `--allow-short-count` is passed.
