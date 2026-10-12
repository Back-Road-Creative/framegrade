## Route Story satellite opening (`gomoveshift/pipeline/route_story/opening.py`)

`SatelliteOpening` is a deterministic north-up camera path from an appropriately broad U.S. pose
to the visible route start. It uses a smooth endpoint-eased trajectory and returns its persisted
endpoint exactly, so the overview and the first route-follow frame cannot drift. `overview` scenes
freeze source recording time; camera movement never becomes travel, speed, distance, or elevation.
The camera module does not fetch imagery: later bundle preparation supplies only local, versioned
assets to both preview and final render paths.
