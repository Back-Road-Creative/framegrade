### Fixed

- Facebook and Instagram: a byte upload that breaks off (a write timeout) now starts over on a
  fresh upload. Facebook kept the unfinished video's id and re-read its status every hour for good;
  a retained id whose live status shows the upload never finished and publishing never began now
  starts a fresh upload too. Instagram raised the timeout raw, skipped its fresh-container retry and
  spent its attempts polling a container that never got its bytes. The Day 10 reel hit both.
