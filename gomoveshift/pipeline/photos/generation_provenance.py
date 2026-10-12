"""Provenance record for one LLM-generated caption.

A shipped caption and a fallback string are otherwise indistinguishable once
saved to YAML — both carry the batch-level ``model``/``generated_at`` fields
and nothing else. ``GenerationProvenance`` records, per caption, what actually
produced it: the model invoked, a hash of the prompt template that was in
effect, the metadata keys it was grounded in, whether validation forced a
fallback, how many attempts were spent, and how long it took.

Deliberately absent: a dollar cost. No dollar cost is recorded because the
caption call goes through the local CLI, which reports no per-call price.
Recording an invented estimate would be worse than recording nothing. ``duration_s`` and
``attempts`` are the honest, measurable substitutes.

Also deliberately absent: a fabricated confidence/uncertainty score. The
honest signal available here is which validation checks passed and which
failed — ``validation_passed``/``validation_failed`` record that instead of a
made-up probability.

"Not recorded" (an old batch saved before this field existed) is a distinct
state from "recorded, no fallback used" — collapsing them would launder
silence into a clean bill of health. Use ``GenerationProvenance.not_recorded()``
/ ``from_dict(None)`` for the former; check ``.recorded`` before trusting any
other field.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GenerationProvenance:
    """What produced one generated caption (or the fact that we don't know)."""

    recorded: bool
    model: Optional[str]
    recipe_version: Optional[str]
    evidence: tuple = field(default_factory=tuple)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    attempts: int = 0
    duration_s: float = 0.0
    validation_passed: tuple = field(default_factory=tuple)
    validation_failed: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if not self.recorded:
            # The sentinel — every other field MUST stay at its "unknown"
            # default, enforced rather than merely documented. Without this a
            # caller could build recorded=False alongside model="sonnet" and
            # fallback_used=True, and a reader that trusted any field but
            # .recorded would read a claim nobody ever made. That is exactly
            # the laundering this class exists to stop, so it raises.
            populated = {
                name: value
                for name, value in (
                    ("model", self.model),
                    ("recipe_version", self.recipe_version),
                    ("evidence", self.evidence),
                    ("fallback_used", self.fallback_used),
                    ("fallback_reason", self.fallback_reason),
                    ("attempts", self.attempts),
                    ("duration_s", self.duration_s),
                    ("validation_passed", self.validation_passed),
                    ("validation_failed", self.validation_failed),
                )
                if value not in (None, False, 0, 0.0, ())
            }
            if populated:
                raise ValueError(
                    "recorded=False is 'we do not know what produced this'; it cannot "
                    f"carry findings: {sorted(populated)}. Use not_recorded(), or set "
                    "recorded=True if these fields are real."
                )
            return
        if self.fallback_used and self.model is None:
            raise ValueError(
                "fallback_used=True requires a known model (the model that was "
                "actually attempted before falling back) — use not_recorded() "
                "instead of model=None for an unknown provenance"
            )
        if self.fallback_used and not self.fallback_reason:
            raise ValueError("fallback_used=True requires a non-empty fallback_reason")
        if self.attempts < 1:
            raise ValueError("a recorded provenance needs attempts >= 1")

    @classmethod
    def not_recorded(cls) -> "GenerationProvenance":
        """Sentinel for a caption whose batch predates provenance tracking.

        NOT the same as "recorded, fallback_used=False" — that would claim
        knowledge ("the model produced this cleanly") this instance doesn't
        have.
        """
        return cls(
            recorded=False,
            model=None,
            recipe_version=None,
        )

    def to_dict(self) -> dict:
        return {
            "recorded": self.recorded,
            "model": self.model,
            "recipe_version": self.recipe_version,
            "evidence": list(self.evidence),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "attempts": self.attempts,
            "duration_s": self.duration_s,
            "validation_passed": list(self.validation_passed),
            "validation_failed": list(self.validation_failed),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "GenerationProvenance":
        """Absent/empty/malformed input is "not recorded" — never fabricated."""
        if not data:
            return cls.not_recorded()
        if not data.get("recorded", False):
            return cls.not_recorded()
        return cls(
            recorded=True,
            model=data.get("model"),
            recipe_version=data.get("recipe_version"),
            evidence=tuple(data.get("evidence") or ()),
            fallback_used=bool(data.get("fallback_used", False)),
            fallback_reason=data.get("fallback_reason"),
            attempts=int(data.get("attempts", 1)),
            duration_s=float(data.get("duration_s", 0.0)),
            validation_passed=tuple(data.get("validation_passed") or ()),
            validation_failed=tuple(data.get("validation_failed") or ()),
        )
