### Fixed

- A drive description's hook and `Location` line now name the first place in the LF label that holds every retained second, not the first town the label happens to say: "Beaufort County, South Carolina" over a route that also crosses Colleton County now prints "South Carolina", not "Beaufort". The upload's zero-tolerance place gate also runs where the text is made, so an audit (`--skip-upload`) pass refuses a description the upload would refuse (`place_outside_text`) instead of agreeing and failing at `videos.insert`.
