### Fixed

- A route-claim reverse lookup that fails (a timeout, a dropped connection) is retried after
  5, 15 and 45 s before its sample counts as unresolved. One timed-out lookup during a network
  blip had refused a Short label the GPS supports and failed the whole day after its encode.
  A lookup that keeps failing is still unresolved, and a lookup that answers with nothing is
  not retried.
