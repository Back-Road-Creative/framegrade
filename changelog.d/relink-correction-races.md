### Fixed

- A second stale duplicate is queued for delete, adopting a comment never deletes it, and a relink
  during an in-flight edit stays pending for the next pass instead of being lost.
- A 404 edit still runs the row's pending deletes; `forbidden` is a re-auth when posting again (a
  refusal only for corrections); a `channels.list` scope gap is a re-auth.
