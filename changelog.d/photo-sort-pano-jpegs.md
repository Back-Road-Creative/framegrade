### Fixed

- `photos sort` panoramas no longer fail on RAW cards with "could not be read as an image":
  the stitcher is now handed the developed JPEGs the run wrote instead of the camera RAWs,
  which `cv2.imread` cannot open.
