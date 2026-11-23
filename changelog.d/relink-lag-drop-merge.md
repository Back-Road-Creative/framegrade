### Fixed

- A second lag drop on a row whose `first_comment_lag_delete_dropped` incident is still open joins
  it (one incident naming every dropped id, pushed again) instead of overwriting the first.
- The incident's hand-delete advice is never cut: only the ids and reasons are length-capped, and
  a posted row's failing ids are grouped by reason like a capped row's, so eight failing Meta ids
  or a long Graph error keep every id and the whole advice.
