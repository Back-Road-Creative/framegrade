### Fixed

- A Short whose title named a town its own route record measured for every retained second ("Feb 24, 6th stretch: Quebec") was refused on every dispatch with `description_no_clip_place`, because the review took clip places only from the county votes. Such a town is now a clip place under the route record's evidence id, and the Short's text must name only that town. A town held for part of the clip is still refused, and county-titled Shorts are reviewed as before (docs/w9c2-final-text-bind.md).
