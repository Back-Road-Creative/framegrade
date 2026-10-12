## Route Story timeline (`gomoveshift/pipeline/route_story/timeline.py`)

The storyboard maps presentation frame boundaries to original recording time. Scenes use half-open
integer frame ranges, so a transition cannot render the same frame twice or create a missing-frame
hole. Route-follow and source-video scenes may advance recording time; photo holds, grouped stops,
gap/day transitions, and recap scenes freeze it. Preview and final render code must read this shared
mapping rather than browser wall time, keeping measured speed, elevation, and grade independent of
animation speed.
