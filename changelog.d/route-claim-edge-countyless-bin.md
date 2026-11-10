### Fixed

- A Short whose last bin holds no fix of its own (the next fix lands just past
  the cut, inside `GAP_S`) no longer refuses its county label: a county-less
  bin with no county-bearing bin on one side takes the nearest source fix
  outside the cut on that side, the same stand-in a head or tail gap already
  uses, and only for a stateless county in one country. Short windows now
  carry those outside fixes (`short_route_window`).
