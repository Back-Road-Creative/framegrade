## Route Story beats (`gomoveshift/pipeline/route_story/beats.py`)

`extract_beats` turns accepted route evidence into a deterministic, time-ordered list of `Beat`s
(start, stop, high point, steepest sustained climb, media cluster, signal gap, finish), each with a
shot from the vocabulary and label/value `facts` computed only from data. Stops merge adjacent
slow edges within one segment; the steepest climb is the best elevation gain within a sliding
distance window; media clusters group anchored items by capture-time proximity; a segment break is
always a signal-gap beat. A beat with no supporting evidence (no elevation, no anchored media) is
omitted rather than invented. `write_chapters_vtt` renders the beat list to a WebVTT chapter track
given a caller-supplied `route_time -> film_time` mapping, never assuming a particular timeline.
`beats_receipt` gives JSON-safe rows for the project receipt.
