# Facebook Page photo byte client

`publishing.facebook_photo_client.FacebookPagePhotoClient` is a standalone,
unwired transport for one JPEG on a Facebook Page. It avoids the Instagram
container client and does not require a publicly accessible media URL. It is
not constructed by the queue, CLI, media bridge or handler factories.

## Contract

The constructor takes the existing SOPS-loaded credential mapping: `page_id`,
`page_access_token`, and optional `graph_version`. It follows the shared version
default, currently `v21.0`, while requiring numeric Page IDs and a `vN.N` version.
The token is sent in the Authorization header, never in the URL or result.

`upload_photo(jpeg_bytes, caption=..., expected_sha256=..., published=...)`
requires an immutable `bytes` snapshot, its exact lowercase SHA-256, a final
caption string, and an explicit boolean publication intent. Omitting `published`
or passing a string/integer is refused before network access. It sends one
multipart `source` to `/{page_id}/photos`, with `caption` unchanged and `published`
encoded as `true` or `false`. There is no update, schedule, gallery, deletion,
unpublished-to-published conversion or automatic retry operation.

The JPEG marker check only rejects clearly wrong transport input; it does not
decode or certify the image. The caller must validate the actual media and
capture the final approved bytes before calling. Hash equality detects a
snapshot mismatch; supplying a hash is not itself evidence of user approval.
The client never opens paths, fetches Drive URLs or changes title/description
formatting. The caller renders those fields into the exact approved caption.

Successful creation returns a frozen `PhotoUploadReceipt` containing Photo ID,
Page ID, byte hash, requested publication intent, requested Graph version and a
validated response version header when present. **A creation receipt is not a
publication or ownership receipt.** Persist it before starting readback.

`read_status(receipt)` first checks that the receipt belongs to this client's
Page. It then reads exactly:

1. Photo `id,page_story_id`, checking the exact Photo ID and Page-prefixed story ID.
2. Story `id,is_published`, requiring the same story ID and an actual boolean.
3. Story `id,from`, requiring the same story ID and exact Page owner.

Only all three valid responses produce `PhotoStatus`. Its `is_published` is the
observed value, independently of what the upload requested. False means observed
unpublished; it never authorizes another upload. The separate reads are not an
atomic snapshot. `object_id` is neither requested nor required.

## Unknown outcomes and caller responsibilities

Every post-dispatch creation error, non-200 response (including redirects),
malformed JSON, Graph error body or invalid/missing ID raises
`PhotoUploadUnknownError`. This deliberately does not classify a POST as safe to
repeat. A request may have reached Facebook even when no ID reaches the caller.
Process termination can also occur between receipt and durable persistence.

Readback errors raise `PhotoReadbackUnknownError`, preserving the known Photo ID
and any validated story ID obtained before failure. The original creation receipt
remains unchanged. A caller may retry status reads, never infer permission to
repeat a write. These exceptions have no dispatcher retry-state annotation and
must not simply be passed into the queue's generic retry path.

Messages and explicit exception attributes omit raw platform errors, captions,
tokens and requests, and the underlying exception is not chained. Python
tracebacks still reference local variables: do not log traceback locals, session
objects or credential mappings. No request/response logging is performed here.

The default Requests session uses zero transport retries, disabled environment
credentials/proxies, a 10-second connection/60-second read timeout, and no HTTP
redirects. A supplied session is a trusted test seam: it must honor those request
options and cannot add retries through more-specific adapters or overrides.
Timeouts are not a total execution deadline. No background activity is started.

Production activation requires a separate queue/approval unit that durably
records the exact approved plan, media kind, hash, caption, Page and write-ahead
attempt **before** dispatch. That unit must retain uncertain attempts even when
no Photo ID returns, reconcile Photos separately from Reels, and prevent replay
across crashes, leases and retry limits. Fresh privacy/account checks and trusted
approval remain mandatory. This client provides neither approval validation,
idempotency nor exactly-once delivery.

## Qualification evidence

The previously authorized single synthetic JPEG diagnostic verified multipart
`source`, `published=false`, Photo-to-story linkage, story publication status
`false`, and matching Page ownership. `object_id` failed with Graph error code 12.
The request named `v21.0` but response headers reported **v26.0**; this does not
establish empirical support for v21.0 behavior. The receipt records both versions
instead of silently calling them equivalent.

Meta's official Python Business SDK 26.0.4
([`Page.create_photo`, pinned commit b08e5c1](https://github.com/facebook/facebook-python-business-sdk/blob/b08e5c1d36a2adad3c4ede6578654c8eaca5f225/facebook_business/adobjects/page.py#L3903-L3990))
declares the `published` boolean.
The `true` request path here is contract-based and tested offline only. Public
publication, scheduling and gallery behavior have not been qualified by that
diagnostic. This change makes no live calls and does not activate publishing.

The focused offline tests cover both explicit intents, snapshot binding, exact
IDs and ownership, strict status values, unknown writes, preserved readback IDs,
sanitized exceptions, version provenance, and disabled default transport retries.
