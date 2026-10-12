"""Deterministic ``PublishPlan`` hashing — mirrors ``review_gate.compute_gate_cd_hash``:
sha256 of canonicalized JSON, excluding wall-clock/generation fields."""

from __future__ import annotations

import dataclasses
import hashlib
import json

from gomoveshift.publishing.models import PublishPlan, ShortPublishRecord

# Fields excluded from the hash: wall-clock/generation-only, never
# consent-relevant. Everything else in ShortPublishRecord participates.
_EXCLUDED_RECORD_FIELDS = frozenset({"generated_at"})


def _canonicalize_record(record: ShortPublishRecord) -> dict:
    payload = dataclasses.asdict(record)
    for excluded in _EXCLUDED_RECORD_FIELDS:
        payload.pop(excluded, None)
    return payload


def compute_plan_hash(plan: PublishPlan) -> str:
    """Stable sha256 of a ``PublishPlan``'s consent-relevant content."""
    payload = {"shorts": [_canonicalize_record(record) for record in plan.shorts]}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
