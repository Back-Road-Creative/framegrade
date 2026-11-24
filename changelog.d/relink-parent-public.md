### Fixed

- `relink_replaced_lf` takes `{old: (new, new_publish_at)}` and never links a private video: an
  unposted row drops the link, a pending edit becomes a delete, and the drain re-checks the gate
  before editing or posting.
- A Meta replace never ends at zero comments: after a confirmed delete a refused post returns the
  row to unposted with the new text; a delete answered "does not exist" counts as success; a
  failing correction pass no longer stops Meta posting; a capped incident never says the drain
  will edit.
