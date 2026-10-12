"""
Content Validator — enforces safe, accurate text in all generated metadata.

Blocks subjective claims, misleading promises, and phrases that contradict
the pipeline's audio/video behavior. Runs on ALL generated titles, descriptions,
captions, and hooks before they are written to output files.

If any content fails validation, the pipeline MUST stop.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation."""

    valid: bool
    violations: list[str]

    def __str__(self) -> str:
        if self.valid:
            return "PASS"
        return f"FAIL: {'; '.join(self.violations)}"


# Phrases that must NEVER appear in any generated content.
# Case-insensitive matching.
BLOCKED_PHRASES = [
    "turn up the volume",
    "turn it up",
    "crank up the volume",
    "crank it up",
    "volume up",
    "blast the speakers",
    "wait for the view",
    "wait for it",
    "wait till you see",
    "you won't believe",
    "you wont believe",
    "unbelievable",
    "never disappoints",
    "never fails to impress",
    "mind-blowing",
    "mind blowing",
    "jaw-dropping",
    "jaw dropping",
    "insane",
    "this is unreal",
    "is unreal",
    "absolutely insane",
    "blows my mind",
    "blow your mind",
    "life-changing",
    "life changing",
    "once in a lifetime",
    "you have to see",
    "you need to see",
    "trust me",
    "i promise",
    "guaranteed to",
    "will change your life",
    "best road ever",
    "best drive ever",
    "best route ever",
    "most beautiful road",
    "most scenic road",
    "complete guide",
    "full guide",
    "full tour",
    "travel guide",
    # Unverifiable road character claims (GPS can't confirm)
    "only locals know",
    "locals know this",
    "best kind of lost",
    "hidden gem",
    "sunday drive",
    "quiet roads",
    "quiet road",
    "quiet ride",
    # Audio contradiction (pipeline outputs quiet audio)
    "turn the sound on",
    "turn on the sound",
]

# Regex patterns for broader unsafe content detection.
# Each is a tuple of (pattern, description).
BLOCKED_PATTERNS = [
    (
        r"\bturn\s+(up|it\s+up)\b",
        "Audio volume reference (pipeline outputs quiet audio)",
    ),
    (
        r"\bwait\s+(for|till|until)\s+(the|this)\b",
        "Promises specific upcoming content without knowing what it is",
    ),
    (
        r"\byou\s+won'?t\s+believe\b",
        "Clickbait phrasing",
    ),
    (
        r"\b(most|best)\s+(beautiful|scenic|amazing|incredible|stunning)\b",
        "Superlative claim about unknown content",
    ),
    (
        r"\b(insane|unreal|crazy|wild)\s+(road|drive|route|view|scenery)\b",
        "Hyperbolic subjective claim",
    ),
    # False claims about being lost on GPS-guided routes
    (
        r"\blost\s+in\s+.+\u2014",
        "False claim about being lost — GPS-guided route",
    ),
    # Subjective road character claims
    (
        r"\bthis\s+road\s+is\s+(incredible|amazing|unbelievable)\b",
        "Subjective road claim — unverifiable from GPS data",
    ),
    # False discovery claims on planned routes
    (
        r"\b(found|discovered)\s+this\s+(road|route)\b",
        "False discovery claim — planned GPS-recorded route",
    ),
]

# Self-contradictory (location visible/named alongside the text): asks
# viewers to guess/name/place a location the OST burns on-screen and the
# caption often prints two lines above. Also catches
# "Guess below", "Where is this?", "Where in the world…", "Name this road",
# "Do you recognise this place?" (the deleted discovery caption class).
#
# These four are the ONLY patterns an ``interactive`` short's
# ``validate(..., interactive=True)`` call skips — an interactive short's
# whole point is to ask "where is this?" with the place withheld until the
# description tail / pinned comment, so the OST-visible assumption behind
# these patterns doesn't hold for it. Every other rule (blocked phrases,
# every other pattern) still applies unchanged.
SELF_CONTRADICTION_PATTERNS = [
    (
        r"\bguess\s+(where|the\s+location|below)\b",
        "Self-contradictory — location is visible alongside this text",
    ),
    (
        r"\bwhere\s+(is\s+this|in\s+the\s+world)\b",
        "Self-contradictory — 'where is this' while the location is shown/named",
    ),
    (
        r"\bname\s+this\s+(road|place|route|spot|location|town)\b",
        "Self-contradictory — asks viewers to name a location already shown",
    ),
    (
        r"\b(recognise|recognize)\s+this\s+(place|road|route|spot|location|town)\b",
        "Self-contradictory — asks recognition of a location already shown",
    ),
]

BLOCKED_PATTERNS = BLOCKED_PATTERNS + SELF_CONTRADICTION_PATTERNS


class ContentValidator:
    """
    Validates all generated text content against safety rules.

    Usage:
        validator = ContentValidator()
        result = validator.validate(title)
        if not result.valid:
            raise ValueError(f"Unsafe content: {result}")

        # Or validate everything at once:
        validator.validate_all(titles=[...], descriptions=[...], captions=[...])
    """

    def __init__(
        self,
        extra_blocked_phrases: Optional[list[str]] = None,
        extra_blocked_patterns: Optional[list[tuple[str, str]]] = None,
    ):
        self.blocked_phrases = list(BLOCKED_PHRASES)
        if extra_blocked_phrases:
            self.blocked_phrases.extend(extra_blocked_phrases)

        self.blocked_patterns = list(BLOCKED_PATTERNS)
        if extra_blocked_patterns:
            self.blocked_patterns.extend(extra_blocked_patterns)

        # Pre-compile patterns for performance
        self._compiled_patterns = [
            (re.compile(pat, re.IGNORECASE), desc) for pat, desc in self.blocked_patterns
        ]
        # The subset of ``self.blocked_patterns`` an interactive short's
        # ``validate(interactive=True)`` skips (the 4 canonical
        # SELF_CONTRADICTION_PATTERNS baked into BLOCKED_PATTERNS above) —
        # a caller-supplied ``extra_blocked_patterns`` is never in this set,
        # so a custom pattern always still applies.
        self._self_contradiction_patterns = {pat for pat, _ in SELF_CONTRADICTION_PATTERNS}
        self._compiled_patterns_no_self_contradiction = [
            (compiled, cdesc)
            for (compiled, cdesc), (pat, _desc) in zip(
                self._compiled_patterns, self.blocked_patterns, strict=True
            )
            if pat not in self._self_contradiction_patterns
        ]

    def validate(
        self, text: str, field_name: str = "content", *, interactive: bool = False
    ) -> ValidationResult:
        """
        Validate a single piece of text content.

        Args:
            text: The text to validate
            field_name: Name of the field (for error messages)
            interactive: when True, skips the 4 self-contradiction
                patterns (``SELF_CONTRADICTION_PATTERNS``) that assume the
                place is visible/named alongside this text. An interactive
                short withholds the place on purpose, so "where is this?"
                is truthful, not self-contradictory. Every other rule still
                applies. Default ``False`` preserves today's behaviour.

        Returns:
            ValidationResult with pass/fail and violation details
        """
        if not text:
            return ValidationResult(valid=True, violations=[])

        violations = []
        text_lower = text.lower()

        # Check blocked phrases
        for phrase in self.blocked_phrases:
            if phrase.lower() in text_lower:
                violations.append(
                    f"{field_name}: blocked phrase '{phrase}' found in: '{text[:80]}...'"
                    if len(text) > 80
                    else f"{field_name}: blocked phrase '{phrase}' found in: '{text}'"
                )

        # Check blocked patterns
        patterns = (
            self._compiled_patterns_no_self_contradiction
            if interactive
            else self._compiled_patterns
        )
        for pattern, description in patterns:
            match = pattern.search(text)
            if match:
                violations.append(f"{field_name}: {description} — matched '{match.group()}'")

        # Deduplicate violations (phrase + pattern may overlap)
        unique_violations = list(dict.fromkeys(violations))

        return ValidationResult(
            valid=len(unique_violations) == 0,
            violations=unique_violations,
        )

    def validate_all(
        self,
        titles: Optional[list[str]] = None,
        descriptions: Optional[list[str]] = None,
        captions: Optional[list[str]] = None,
        hooks: Optional[list[str]] = None,
        *,
        interactive: bool = False,
    ) -> ValidationResult:
        """
        Validate multiple content fields at once.

        Raises ValueError if ANY content fails validation.

        Args:
            interactive: forwarded to :meth:`validate` for every field
                (see its docstring). Default ``False`` preserves today's
                behaviour.

        Returns:
            Combined ValidationResult
        """
        all_violations = []

        for label, items in [
            ("title", titles or []),
            ("description", descriptions or []),
            ("caption", captions or []),
            ("hook", hooks or []),
        ]:
            for i, text in enumerate(items):
                result = self.validate(text, f"{label}[{i}]", interactive=interactive)
                all_violations.extend(result.violations)

        return ValidationResult(
            valid=len(all_violations) == 0,
            violations=all_violations,
        )

    def enforce(
        self,
        titles: Optional[list[str]] = None,
        descriptions: Optional[list[str]] = None,
        captions: Optional[list[str]] = None,
        hooks: Optional[list[str]] = None,
        *,
        interactive: bool = False,
    ) -> None:
        """
        Validate and RAISE if any content is unsafe.

        This is the enforcement method — call this to guarantee
        no unsafe content passes through the pipeline.
        """
        result = self.validate_all(
            titles=titles,
            descriptions=descriptions,
            captions=captions,
            hooks=hooks,
            interactive=interactive,
        )
        if not result.valid:
            violation_summary = "\n  ".join(result.violations)
            raise ContentValidationError(
                f"Content validation failed with {len(result.violations)} violation(s):\n"
                f"  {violation_summary}"
            )


class ContentValidationError(Exception):
    """Raised when generated content fails safety validation."""

    pass
