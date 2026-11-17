### Fixed

- Overpass requests now wait up to 90 s for an answer instead of 25 s. The 25 s figure is the server's own query budget; a busy mirror queues a request before that budget starts, so the client hung up on mirrors that were slow but answering, and a drive's place lookup failed on every endpoint.
