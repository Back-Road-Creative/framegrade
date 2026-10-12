## Route Story receipt verification (`gomoveshift/pipeline/route_story/verify.py`)

`verify_receipt(path)` re-checks a receipt written by a prior offline step without any network
call or trust in the receipt's own claims: it detects the receipt's kind from its top-level
`schema` field, then independently recomputes every fact it can from the files the receipt
references. Two kinds are recognised — a bundle `manifest.json` (`bundle.MANIFEST_SCHEMA`) and a
`route-story-proof-receipt/2` (`proof.render_proof_clip`'s output) — any other or missing schema
is reported as `kind="unknown"` with a single failing check, never silently treated as passing.

- **Bundle manifest**: delegates straight to `bundle.verify_bundle(root)`, one `Check` per problem
  string it returns (missing file, checksum mismatch, extra file — each names the specific
  `relpath`), plus one passing check when the list is empty.
- **Proof receipt**: checks the required keys are present (`output`, `gpx`, `bundle`, `width`,
  `height`, `duration_seconds`); that the recorded `gpx` source file still exists; that the
  recorded `bundle` directory's own manifest still verifies (same `bundle.verify_bundle` reuse,
  so a tampered bundle file surfaces here too); that the recorded `output` video file still
  exists; and, via an injectable `video_prober` (defaults to `FFprobeClient.probe_or_none`), that
  the output's actual width/height match the receipt and its actual duration is within
  `max(2/fps, 0.25)` seconds of `duration_seconds`.

`VerifyReport` carries an ordered list of `Check(name, ok, detail)` and an `ok` property that is
`True` only when there is at least one check and every one passed (an empty report is never
treated as a pass). The `route-story-verify <receipt>` CLI prints one line per check and exits 0
only when `report.ok` is true.
