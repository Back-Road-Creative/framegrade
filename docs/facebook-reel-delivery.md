# Authorized Reel delivery

`FacebookReelDelivery` is an explicitly injected consumer for an approved general
content video. It reuses `FacebookReelsAdapter` without inventing driving or
long-form metadata. No production CLI, registry or queue factory constructs it
yet. This change does not enable a live publishing route.

## Authority and timing

The caller maps a positive local delivery ID to an immutable approved reservation
and provides `authorize(delivery_id) -> AuthorizedReel`. The port must verify the
real owner decision, reservation identity, current draft revision, Page access,
media format and privacy review, and snapshot the exact approved bytes. The
dataclass alone is not evidence of authorization; it must never be populated
directly from chat, Tasks, a Drive note or an unverified request.

Every proof binds the plan and approval-receipt hashes, nonce, ordered item index,
Page, complete caption, video hash, public audience, publication-media purpose,
and execution window. The consumer checks the byte hash and requires a fresh
evidence deadline at most 15 minutes ahead. Media decoding and platform format
qualification remain the trusted port's responsibility.

The execution window is `[not_before, not_after)`. A future schedule waits in the
parent scheduler; this consumer sends an immediate `PUBLISHED` finish only once
the approved window is open. It does not translate a missed slot into a late
post or use Facebook's native scheduled-post mode.

Authority is obtained before start and again after upload. Only `valid_until`
may refresh; all approved fields and the media bytes must remain identical.
The clock and client account are checked after database waits, before start and
byte upload, and immediately before finish. A request already sent can complete
after the deadline; the consumer cannot retract an in-flight provider request.

## Durable states and recovery

Call `init_reel_ledger(connection)` explicitly. Its additive schema uses a
savepoint and preserves caller transactions. Delivery and reconciliation require
their own durable transactions. An authority callback that leaves unrelated
database work uncommitted is refused without committing or rolling back that work.

| Retained state | Meaning | Next invocation |
| --- | --- | --- |
| `start_unknown` | Start intent committed; provider ID not retained | Hold for manual reconciliation; never repeat start |
| `upload_started` | Provider ID committed before byte transfer | Read status only; never replay bytes or finish |
| `finish_unknown` | Finish intent committed before publish request | Read status only; never repeat finish |
| `published` | Exact finish or status confirmation retained | Return the verified local receipt |

The unique delivery ID and `(plan_sha256, item_index)` prevent a second local
attempt for the same signed item, including competing connections. Crash or
provider failure preserves the last committed state. Known-ID recovery does not
load mutable draft/media/approval data; it only reads the retained provider ID
through the same Page client. Negative or malformed status never authorizes a
new write. Changed Page identity, inconsistent local bindings or receipts fail
closed. This is conservative recovery, not an exactly-once provider guarantee.

No upload URL or token is persisted. Receipts contain the Page/video IDs, original
plan/receipt hashes, nonce, item index, media hash, confirmation type and derived
permalink. Provider exception text and chains are replaced with fixed pending
errors; callers must also avoid logging traceback locals or client credentials.

## Composition still required

The bridge must supply the real authority port and a qualified runtime containing
this consumer. Its provider client must use SOPS-managed credentials, bounded
timeouts, no automatic POST retries and no credential-forwarding redirects.
The existing low-level adapter does not establish that transport qualification
by itself. Upload URLs are additionally checked here against the existing exact
Facebook HTTPS hosts, with userinfo, custom ports and fragments refused.

Mixed photo/video batches remain ordered, separate provider items. The parent
must preserve partial outcomes; it cannot promise atomic publication or rollback.
No first-comment or extra publication action is issued by this consumer.

## Validation

Tests use actual SQLite checkpoints and independent observer/contender connections,
plus mocked provider I/O. One test exercises the real low-level adapter against
a fake HTTP session. Coverage includes absent/stale authority, account rotation,
changed bindings, interrupted start/upload/finish, malformed responses, clock
expiry during upload/checkpoint/DB wait, duplicate attempts and corrupted receipts.
No live public-publishing qualification is claimed by these offline checks.
