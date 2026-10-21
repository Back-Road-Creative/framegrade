## Waypost titles for a verified county or state

After scoring, the LF planner knows the retained segments. The pre-encode
route gate checks their GPS and either keeps the proposed location or demotes
it to a supported county or state. A county label is eligible for a
region-only Waypost decision only when **both** its county claim and its state
claim pass the retained-route gate. The label must match that exact county and
state (full name or postal abbreviation). A state-only label must match the
supported state. A town that the route does not support still refuses before
the title selector runs.

GMS passes the approved region label and the region evidence ID to Waypost's
`region_only_lf` long-form shape. Waypost keeps the factual baseline until
fresh measured title CTR favors its factual region-drive candidate. GMS checks
the chosen title against the same retained route before video encoding and
persists the decision for Phase D and upload. It never invents town endpoints
to make a county fallback fit a route template.

GMS converts the Waypost decision to strict JSON before saving pipeline state.
Candidate and evidence lists therefore have the same shape before and after a
resume; a value that cannot be serialized refuses before encoding.

This path requires the installed Waypost pin containing the region-only title
policy. An upload-disabled acceptance render must verify the combined release
before publication; unsupported titles continue to refuse.
