## Route Story telemetry (`gomoveshift/pipeline/route_story/metrics.py`)

Route Story is a presentation of a recorded journey, not a navigation engine. `summarize_track`
accepts ordered source segments and derives every fact from their original timestamps and geometry.
It does not join the last point of one segment to the first of the next: distance and elapsed time
cover only accepted edges, while the timestamp separation becomes `unobserved_gap_seconds`. A map
may animate that recording at any speed without changing `current_speed_mps`, moving/elapsed
averages, or trip duration.

The summary stores SI values. Display code selects units and labels `moving average` separately
from `elapsed average`; a classified stopped edge remains in elapsed time and total distance. Grade
is signed percent rise over horizontal route distance: uphill is positive and downhill negative.
The contract exposes net, uphill, downhill and absolute average grade because an out-and-back can
have a net grade of zero while still accumulating substantial ascent and descent. Elevation-dependent
values are `None` when no accepted edge has two elevations; zero is only a measured zero. Current
grade comes from the final observed elevation-valid edge. A later versioned elevation-treatment
stage may smooth or reconcile source altitudes before this calculation, and its identity must be
recorded with the Route Story receipt.

<a id="route-story-display-smoothing"></a>
