"""LLM-powered caption generation for photo posts.

Uses the local Claude CLI for batch caption generation with voice style prompts.
Caches results to YAML for human review before publishing.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from .._atomic_write import atomic_write_yaml
from ..content_validator import ContentValidator
from .generation_provenance import GenerationProvenance
from .selector import PhotoSelection
from .voice_styles import VOICE_EXAMPLES, VOICE_PROMPTS, VoiceStyle

logger = logging.getLogger(__name__)

# Claude CLI model alias
DEFAULT_MODEL = "sonnet"


def default_caption_dir() -> Path:
    """Directory for cached captions: ``<photo output dir>/photo_captions``.

    Resolved at call time from ``gms_photo_output_dir()`` (``GMS_PHOTO_OUTPUT_DIR``
    overrides the per-user data-home default), the same root ``PhotoOrchestrator``
    writes its captions under.
    """
    from ...paths import gms_photo_output_dir

    return gms_photo_output_dir() / "photo_captions"


# Shared blocklist validator (hype/misleading phrases) applied to every LLM
# caption field before it can ship. Instantiated once — patterns are compiled.
_CONTENT_VALIDATOR = ContentValidator()

# Hashtags are capped in count and per-tag length so an over-long LLM list
# can't bloat the CSV.
_MAX_HASHTAGS = 30
_MAX_HASHTAG_LEN = 100


@dataclass
class PhotoCaption:
    """Generated caption for a single photo."""

    photo_index: int
    main_caption: str  # 250 chars max
    tiktok_caption: str  # 90 chars max
    hashtags: List[str]
    journey_slug: str = ""
    filename: str = ""
    location: str = ""
    first_comment_hook: str = ""  # Engagement question for Instagram first comment
    # What produced this caption (model, recipe version, evidence, fallback use,
    # attempts, validation outcome). GenerationProvenance.not_recorded().to_dict()
    # for a caption predating this field — see generation_provenance.py.
    provenance: dict = field(default_factory=lambda: GenerationProvenance.not_recorded().to_dict())


@dataclass
class CaptionBatch:
    """A batch of generated captions with metadata."""

    week_start: str
    voice_style: str
    captions: List[PhotoCaption]
    reviewed: bool = False
    generated_at: str = ""
    model: str = ""
    # Set by a successful publish, together with the slots that publish booked.
    # A republish reuses them instead of scheduling (and paying for) a second week.
    published: bool = False
    published_slots: List[dict] = field(default_factory=list)

    def save(self, path: Path):
        """Save to YAML for human review (atomic — never a truncated file)."""
        data = {
            "week_start": self.week_start,
            "voice_style": self.voice_style,
            "reviewed": self.reviewed,
            "generated_at": self.generated_at,
            "model": self.model,
            "published": self.published,
            "published_slots": self.published_slots,
            "captions": [asdict(c) for c in self.captions],
        }
        atomic_write_yaml(path, data, default_flow_style=False, sort_keys=False)
        logger.info(f"Captions saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CaptionBatch":
        """Load from YAML."""
        data = yaml.safe_load(path.read_text())
        captions = [PhotoCaption(**c) for c in data.get("captions", [])]
        return cls(
            week_start=data.get("week_start", ""),
            voice_style=data.get("voice_style", ""),
            captions=captions,
            reviewed=data.get("reviewed", False),
            generated_at=data.get("generated_at", ""),
            model=data.get("model", ""),
            published=bool(data.get("published", False)),
            published_slots=data.get("published_slots") or [],
        )


# Control characters (ASCII 0x00-0x08, 0x0b-0x1f, 0x7f) are stripped from prompt
# fields. \t (0x09) and \n (0x0a) are preserved because they appear legitimately
# in day summaries.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Markers that must NEVER appear in interpolated prompt fields — they could
# pivot the downstream caption-LLM call. The list covers triple-backtick fenced
# code blocks plus the common chat-template tokens used by Anthropic / OpenAI
# style chat formats.
_PROMPT_INJECTION_MARKERS = (
    "```",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)


def _sanitize_prompt_field(text: str | None, max_len: int) -> str:
    """Sanitize an operator-/LLM-sourced string before f-stringing into a prompt.

    - Coerces None and non-str inputs to "" defensively (Pydantic / dataclass
      defaults sometimes leak through).
    - Strips ASCII control chars except \\t and \\n.
    - Rejects (raises ValueError naming the offending marker) any input
      containing a triple-backtick fenced-code delimiter or a chat-template
      token (<|im_start|>, <|im_end|>, <|system|>, <|user|>, <|assistant|>).
    - Truncates to max_len with no ellipsis (caller controls semantics).
    - Strips outer whitespace AFTER truncation.
    """
    if text is None or not isinstance(text, str):
        return ""

    cleaned = _CONTROL_CHARS_RE.sub("", text)

    for marker in _PROMPT_INJECTION_MARKERS:
        if marker in cleaned:
            raise ValueError(f"prompt-injection marker {marker!r} found in caption-prompt field")

    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]

    return cleaned.strip()


def _safe_stat(value) -> str:
    """Sanitize a statistics scalar before f-stringing it into a prompt.

    Stats come from journey YAML and are often string-typed, so they carry the
    same payloads every neighbouring field is guarded against. str() first, so
    real int/float stats aren't blanked by the non-str branch of the guard.
    """
    return _sanitize_prompt_field(str(value), max_len=50)


# Static instruction blocks used by _build_prompt, independent of any
# particular photo/voice-style. Hashed together into RECIPE_VERSION so the
# prompt "recipe" a caption was generated under is derived from the prompt
# text itself — it cannot go stale from someone editing the prompt below and
# forgetting to bump a hand-typed version constant.
_CRITICAL_RULES = """CRITICAL RULES:
- ONLY reference locations and details explicitly listed in the photo metadata above.
- If location is "Unknown Location", use the journey name and day summary ONLY. Do NOT invent a place.
- Do NOT describe weather, lighting, sounds, or sensory details — you cannot see the photo.
- Do NOT use first-person sensory claims ("I saw", "I felt", "I heard").
- Do NOT mention elevation gain or elevation statistics — these are cumulative totals that sound misleading.
- Statistics (distance, duration, photo count) are supporting details, not the main story. Use at most one stat per caption, and only when it reveals something genuinely surprising. Most captions should focus on the place.
- Keep captions factual: journey name, location, day. Let the photo speak for itself.
- Include the month and year from the photo date naturally in the caption when it adds context (e.g., "January 2025" or "June 2018"). Do NOT include the exact day number unless it's part of the story ("Day 3")."""

_HOOK_INSTRUCTION = (
    "\nFIRST COMMENT HOOK RULES:\n"
    "- Generate a first_comment_hook: a single sentence with a specific, surprising detail.\n"
    "- It should make someone curious enough to comment without being asked.\n"
    "- NEVER use: 'Rate 1-10', 'Tag someone', 'Guess the location', 'Drop your answer',\n"
    "  'Would you?', 'Have you ever?', or any direct CTA.\n"
    "- Good hooks state something unexpected. Bad hooks ask generic questions.\n"
    "- Keep under 100 characters. No hashtags.\n"
)


def _compute_recipe_version() -> str:
    """Short stable hash of the static prompt instruction blocks.

    Recompute (never hand-edit) after changing _CRITICAL_RULES or
    _HOOK_INSTRUCTION so a saved caption's recipe_version always reflects the
    prompt that actually produced it.
    """
    template_text = _CRITICAL_RULES + _HOOK_INSTRUCTION
    return hashlib.sha256(template_text.encode("utf-8")).hexdigest()[:12]


RECIPE_VERSION = _compute_recipe_version()


def _is_empty_metadata(sel: PhotoSelection) -> bool:
    """Check if a selection has no useful metadata for captioning."""
    has_title = bool(sel.journey_title and sel.journey_title.strip())
    has_location = bool(
        sel.location and sel.location.strip() and sel.location != "Unknown Location"
    )
    has_locations = bool(sel.locations_visited)
    has_summary = bool(sel.day_summary and sel.day_summary.strip())
    return not (has_title or has_location or has_locations or has_summary)


def _evidence_keys(sel: PhotoSelection) -> tuple:
    """Metadata keys actually fed into the prompt for one selection.

    Mirrors the field inclusion logic in _build_prompt so `evidence` reflects
    what the model really had to work with, not a hand-maintained guess.
    """
    keys = []
    if sel.journey_title:
        keys.append("journey_title")
    if sel.location:
        keys.append("location")
    keys.append("day_number")
    keys.append("day_date")
    if sel.locations_visited:
        keys.append("locations_visited")
    if sel.day_summary:
        keys.append("day_summary")
    if sel.statistics:
        for stat_key in ("distance_km", "duration_hours", "photo_count"):
            if sel.statistics.get(stat_key):
                keys.append(f"statistics.{stat_key}")
    return tuple(keys)


def _build_prompt(
    selections: List[PhotoSelection],
    voice_style: VoiceStyle,
) -> str:
    """Build the LLM prompt for batch caption generation."""
    style_instruction = VOICE_PROMPTS[voice_style]
    examples = VOICE_EXAMPLES[voice_style]

    photo_context = []
    for i, sel in enumerate(selections):
        safe_journey_title = _sanitize_prompt_field(sel.journey_title, max_len=200)
        safe_location = _sanitize_prompt_field(sel.location, max_len=200)
        safe_locations_visited = [
            _sanitize_prompt_field(loc, max_len=200) for loc in sel.locations_visited
        ]
        ctx = (
            f"Photo {i + 1}:\n"
            f"  Journey: {safe_journey_title}\n"
            f"  Location: {safe_location}\n"
            f"  Day {sel.day_number} ({sel.day_date})\n"
            f"  Places visited this day: {', '.join(safe_locations_visited)}\n"
        )
        if sel.day_summary:
            safe_day_summary = _sanitize_prompt_field(sel.day_summary, max_len=1000)
            ctx += f"  Day summary: {safe_day_summary}\n"
        if sel.statistics:
            stats = sel.statistics
            if stats.get("distance_km"):
                ctx += f"  Distance: {_safe_stat(stats['distance_km'])} km\n"
            if stats.get("duration_hours"):
                ctx += f"  Duration: {_safe_stat(stats['duration_hours'])} hours\n"
            if stats.get("photo_count"):
                ctx += f"  Photos taken this day: {_safe_stat(stats['photo_count'])}\n"
        photo_context.append(ctx)

    examples_text = "\n".join(f'  - "{ex}"' for ex in examples)

    # Soft CTA in main caption only for engagement styles
    is_engagement_style = voice_style in (
        VoiceStyle.ENGAGING_CONVERSATIONAL,
        VoiceStyle.INTERACTIVE,
    )
    engagement_instruction = ""
    if is_engagement_style:
        engagement_instruction = (
            "\nENGAGEMENT RULES:\n"
            "- End the main_caption with a soft call-to-action (question or prompt).\n"
        )

    # First comment hook instruction for ALL styles
    hook_instruction = _HOOK_INSTRUCTION

    # Build output field list — first_comment_hook always included
    output_fields = (
        "1. main_caption: The primary caption (max 250 characters). No hashtags in the caption.\n"
        "2. tiktok_caption: A very short version (max 90 characters)."
    )
    if is_engagement_style:
        output_fields += " Prefer a question format."
    output_fields += (
        "\n3. hashtags: 5-8 relevant hashtags as a JSON array (without # prefix).\n"
        "4. first_comment_hook: A single surprising-detail sentence for "
        "Instagram first comment (max 100 chars, no hashtags)."
    )

    # Build JSON schema example — always include first_comment_hook
    json_fields = (
        '    "main_caption": "...",\n'
        '    "tiktok_caption": "...",\n'
        '    "hashtags": ["tag1", "tag2", ...],\n'
        '    "first_comment_hook": "..."'
    )

    return f"""Generate social media captions for {len(selections)} travel photos from GoMoveShift road trip journeys.

VOICE STYLE: {voice_style.value}
{style_instruction}

EXAMPLES of this voice:
{examples_text}

{_CRITICAL_RULES}
{engagement_instruction}{hook_instruction}
PHOTO DETAILS:
{"".join(photo_context)}

For each photo, generate:
{output_fields}

Return ONLY valid JSON — an array of objects, one per photo, in the same order:
[
  {{
{json_fields}
  }},
  ...
]"""


# Common English words to exclude from proper-noun extraction
_COMMON_WORDS = {
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "has",
    "his",
    "how",
    "its",
    "may",
    "new",
    "now",
    "old",
    "see",
    "way",
    "who",
    "did",
    "get",
    "let",
    "say",
    "she",
    "too",
    "use",
    "day",
    "got",
    "him",
    "hit",
    "just",
    "like",
    "make",
    "many",
    "some",
    "time",
    "very",
    "when",
    "come",
    "could",
    "than",
    "them",
    "then",
    "into",
    "over",
    "such",
    "take",
    "year",
    "been",
    "have",
    "from",
    "much",
    "that",
    "they",
    "this",
    "will",
    "with",
    "each",
    "which",
    "their",
    "said",
    "what",
    "about",
    "would",
    "there",
    "after",
    "back",
    "been",
    "being",
    "came",
    "down",
    "even",
    "first",
    "good",
    "great",
    "here",
    "high",
    "home",
    "last",
    "long",
    "look",
    "made",
    "more",
    "most",
    "must",
    "next",
    "only",
    "open",
    "part",
    "real",
    "right",
    "same",
    "still",
    "sure",
    "thing",
    "think",
    "those",
    "through",
    "today",
    "took",
    "turn",
    "under",
    "upon",
    "well",
    "went",
    "where",
    "while",
    "along",
    "also",
    "away",
    "before",
    "both",
    "between",
    "does",
    "done",
    "during",
    "every",
    "found",
    "going",
    "keep",
    "know",
    "left",
    "life",
    "little",
    "might",
    "never",
    "other",
    "place",
    "point",
    "road",
    "trip",
    "morning",
    "evening",
    "afternoon",
    "night",
    "breath",
    "something",
    "nothing",
    "everything",
    "another",
    "without",
    "around",
    "always",
    "enough",
    "world",
    "start",
    "stop",
    "end",
    "drive",
    "driving",
    "walk",
    "walking",
    "travel",
    "rate",
    "tag",
    "guess",
    "drop",
    "name",
    "check",
    "share",
    "save",
    "somewhere",
    "nobody",
    "everyone",
    "someone",
    "anywhere",
    "nowhere",
    "kept",
    "pulled",
    "stayed",
    "passed",
    "crossed",
    "reached",
    "spent",
    # Month names
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    # Geographic terms
    "peninsula",
    "pass",
    "coast",
    "county",
    "lake",
    "river",
    "mountain",
    "valley",
    "island",
    "bay",
    "cape",
    "canyon",
    "ridge",
    "falls",
    "creek",
    "cliff",
    "beach",
    "forest",
    "castle",
    "bridge",
    "park",
    "trail",
    # Directions
    "north",
    "south",
    "east",
    "west",
    "northwest",
    "northeast",
    "southwest",
    "southeast",
    "northern",
    "southern",
    "eastern",
    "western",
}

# Stop words for metadata tokenization
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "in",
    "on",
    "at",
    "to",
    "for",
    "is",
    "it",
    "by",
    "or",
    "as",
    "be",
    "if",
    "no",
    "so",
    "up",
    "we",
    "do",
    "my",
    "with",
    "from",
    "that",
    "this",
    "was",
    "are",
    "has",
    "had",
    "not",
}


_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
_CAP_WORD_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


def _extract_proper_nouns(caption_text: str) -> List[str]:
    """Extract place-like proper nouns (capitalized, 3+ letters, not common).

    A capitalized word that is the FIRST token of a sentence is skipped UNLESS
    it is part of a multi-word capitalized phrase (the next word is also
    capitalized, e.g. "Molas Pass", "Rock of Cashel"). Sentence-casing routinely
    capitalizes ordinary leading words ("Standing at…", "From here…") that are
    not places; requiring those to be grounded would bounce valid captions.
    """
    nouns: List[str] = []
    for m in _CAP_WORD_RE.finditer(caption_text):
        word = m.group(1)
        if word.lower() in _COMMON_WORDS:
            continue
        prefix = caption_text[: m.start()]
        sentence_initial = prefix.strip() == "" or bool(_SENTENCE_END_RE.search(prefix))
        if sentence_initial:
            rest = caption_text[m.end() :].lstrip()
            if not re.match(r"[A-Z][a-z]{2,}\b", rest):
                continue
        nouns.append(word)
    return nouns


def _content_issues(text: str, field_name: str = "caption") -> List[str]:
    """Run the shared ContentValidator (hype/blocklist pass) — return issues."""
    return list(_CONTENT_VALIDATOR.validate(text, field_name).violations)


def _validate_caption(caption_text: str, selection: PhotoSelection) -> List[str]:
    """Check caption against metadata — return list of issues found.

    Word-level overlap: extracts place-like proper nouns from the caption and
    requires EVERY one to be grounded in the metadata. A single ungrounded
    (hallucinated) proper noun fails the caption — these ship as public place
    claims, so partial grounding is not safe. Also runs the ContentValidator
    blocklist so hype/misleading phrases are caught here too.
    """
    issues = []

    # Check for elevation references
    if re.search(r"\belevation\b", caption_text, re.IGNORECASE):
        issues.append("Caption references elevation data")

    # Blocklist / hype pass (shared with all other generated text).
    issues.extend(_content_issues(caption_text))

    # Build set of meaningful words from all metadata fields
    metadata_text_parts = []
    if selection.location and selection.location != "Unknown Location":
        metadata_text_parts.append(selection.location)
    for loc in selection.locations_visited:
        metadata_text_parts.append(loc)
    if selection.journey_title:
        metadata_text_parts.append(selection.journey_title)
    if selection.day_summary:
        metadata_text_parts.append(selection.day_summary)

    if selection.day_date:
        try:
            dt = datetime.strptime(selection.day_date, "%Y-%m-%d")
            metadata_text_parts.append(dt.strftime("%B"))  # "January"
            metadata_text_parts.append(str(dt.year))  # "2025"
        except (ValueError, TypeError):
            pass

    metadata_words = set()
    for part in metadata_text_parts:
        for word in part.split():
            cleaned = re.sub(r"[^\w]", "", word).lower()
            if cleaned and cleaned not in _STOP_WORDS and len(cleaned) >= 2:
                metadata_words.add(cleaned)

    # Extract place-like proper nouns (skips sentence-cased common words).
    proper_nouns = _extract_proper_nouns(caption_text)

    if not proper_nouns:
        # Still run stats density even with no proper nouns.
        stats_issue = _validate_stats_density(caption_text)
        if stats_issue:
            issues.append(stats_issue)
        return issues

    # Require EVERY proper noun to be grounded in the metadata (100% grounding).
    for noun in proper_nouns:
        if noun.lower() not in metadata_words:
            issues.append(f"Caption mentions '{noun}' not found in metadata")

    # Check stats density
    stats_issue = _validate_stats_density(caption_text)
    if stats_issue:
        issues.append(stats_issue)

    return issues


# Written-out numbers for stats detection
_WRITTEN_NUMBERS = (
    "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    "fifty|sixty|seventy|eighty|ninety|hundred"
)

# Units that indicate a statistic
_STAT_UNITS = (
    r"km|mi|miles?|hours?|hrs?|minutes?|mins?|frames?|photos?|shots?|stops?|"
    r"counties|peninsulas?"
)

# Compiled patterns
_NUMERIC_STAT_RE = re.compile(rf"\b(\d[\d,]*\.?\d*)\s*({_STAT_UNITS})\b", re.IGNORECASE)
_WRITTEN_STAT_RE = re.compile(rf"\b({_WRITTEN_NUMBERS})\s+({_STAT_UNITS})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DAY_N_RE = re.compile(r"\bDay\s+\d+\b", re.IGNORECASE)


def _count_stats(text: str) -> tuple:
    """Count statistics in text. Returns (stat_count, leads_with_stat).

    Detects number+unit patterns (e.g. '198 km', 'fourteen hours').
    Excludes years (1900-2099) and 'Day N' patterns.
    """
    stats = []

    for m in _NUMERIC_STAT_RE.finditer(text):
        num_str = m.group(1)
        # Exclude years
        if _YEAR_RE.fullmatch(num_str):
            continue
        # Exclude "Day N" patterns — check if preceded by "Day "
        prefix = text[: m.start()]
        if re.search(r"\bDay\s*$", prefix, re.IGNORECASE):
            continue
        stats.append(m.start())

    for m in _WRITTEN_STAT_RE.finditer(text):
        stats.append(m.start())

    stat_count = len(stats)

    # leads_with_stat: first clause (before first '.', '—', or ',') contains a stat
    leads_with_stat = False
    if stats:
        first_break = len(text)
        for sep in [". ", " — ", "—", ", "]:
            idx = text.find(sep)
            if idx != -1 and idx < first_break:
                first_break = idx
        leads_with_stat = any(pos < first_break for pos in stats)

    return (stat_count, leads_with_stat)


def _validate_stats_density(text: str) -> Optional[str]:
    """Check if text has too many statistics or leads with one.

    Returns issue string or None if valid.
    """
    stat_count, leads_with_stat = _count_stats(text)

    if stat_count >= 2:
        return f"Caption contains {stat_count} statistics; max 1 allowed"
    if stat_count == 1 and leads_with_stat:
        return "Caption leads with a statistic"

    return None


def _format_date_label(day_date: str) -> str:
    """Format YYYY-MM-DD as 'Month YYYY' for caption use."""
    try:
        dt = datetime.strptime(day_date, "%Y-%m-%d")
        return dt.strftime("%B %Y")
    except (ValueError, TypeError):
        return ""


def _fallback_caption(selection: PhotoSelection) -> str:
    """Generate a safe fallback caption from confirmed metadata."""
    date_label = _format_date_label(selection.day_date)
    date_suffix = f" ({date_label})" if date_label else ""

    # Build stats fragment when available
    stats_parts = []
    if selection.statistics:
        km = selection.statistics.get("distance_km")
        if km:
            stats_parts.append(f"{km} km")
    stats_fragment = f" {'. '.join(stats_parts)}." if stats_parts else ""

    if selection.location and selection.location != "Unknown Location":
        return (
            f"{selection.location}.{stats_fragment} "
            f"{selection.journey_title}, Day {selection.day_number}{date_suffix}."
        )
    if selection.locations_visited:
        places = ", ".join(selection.locations_visited[:3])
        return (
            f"{places}.{stats_fragment} "
            f"{selection.journey_title}, Day {selection.day_number}{date_suffix}."
        )
    return f"{selection.journey_title} \u2014 Day {selection.day_number}{date_suffix}"


def _sanitize_hashtags(raw) -> List[str]:
    """Cap hashtag count/length and drop any that fail the blocklist.

    Robust-by-construction: the PhotoCaption literally cannot carry an over-long
    list, an over-long tag, or a hype/blocklisted tag \u2014 these ship unedited in
    the Metricool first-comment/Pinterest columns.
    """
    if not isinstance(raw, list):
        return []
    clean: List[str] = []
    for tag in raw:
        if not isinstance(tag, str):
            continue
        tag = tag.lstrip("#").strip()[:_MAX_HASHTAG_LEN]
        if not tag:
            continue
        if not _CONTENT_VALIDATOR.validate(tag, "hashtag").valid:
            logger.warning(f"Dropping hashtag failing content validation: {tag!r}")
            continue
        clean.append(tag)
        if len(clean) >= _MAX_HASHTAGS:
            break
    return clean


def _validate_fields(data: dict, sel: PhotoSelection) -> dict:
    """Validate every LLM caption field for one photo.

    Returns {field_name: [issues]} for each field that failed. Covers
    main_caption (grounding + stats + blocklist), tiktok_caption and
    first_comment_hook (stats + blocklist). Any non-empty result drives
    the retry/fallback flow in generate_batch.
    """
    failed: dict = {}

    main_issues = _validate_caption(data.get("main_caption", "")[:250], sel)
    if main_issues:
        failed["main_caption"] = main_issues

    tiktok_caption = data.get("tiktok_caption", "")[:90]
    tiktok_issues = _content_issues(tiktok_caption, "tiktok_caption")
    tiktok_stats = _validate_stats_density(tiktok_caption)
    if tiktok_stats:
        tiktok_issues.append(tiktok_stats)
    if tiktok_issues:
        failed["tiktok_caption"] = tiktok_issues

    hook = data.get("first_comment_hook", "")[:100]
    hook_issues = _content_issues(hook, "first_comment_hook")
    hook_stats = _validate_stats_density(hook)
    if hook_stats:
        hook_issues.append(hook_stats)
    if hook_issues:
        failed["first_comment_hook"] = hook_issues

    return failed


class CaptionGenerator:
    """Generates photo captions via the local Claude CLI."""

    def __init__(
        self,
        endpoint: str = "",
        model: str = DEFAULT_MODEL,
        caption_dir: Optional[Path] = None,
    ):
        self.endpoint = endpoint  # unused, kept for interface compat
        self.model = model
        self.caption_dir = caption_dir or default_caption_dir()

    def ensure_week_unwritten(self, week_start: str) -> Path:
        """Refuse to clobber an existing review file; return the free path.

        Photos are marked posted at selection, so an overwrite destroys BOTH
        the hand-reviewed captions and the fresh photo set. The orchestrator
        calls this BEFORE selection (so a doomed run takes no holds and burns
        no LLM spend); generate_batch keeps it as a backstop.
        """
        cache_path = self.caption_dir / f"{week_start}_captions.yaml"
        if cache_path.exists():
            raise FileExistsError(
                f"Caption review file already exists: {cache_path}. Regenerating would "
                "destroy any hand-reviewed captions and burn a fresh photo set (photos "
                "are marked posted at selection). Move or rename it to regenerate."
            )
        return cache_path

    def generate_batch(
        self,
        selections: List[PhotoSelection],
        voice_style: VoiceStyle,
        week_start: str = "",
    ) -> CaptionBatch:
        """Generate captions for a batch of photo selections.

        Args:
            selections: Photos to caption
            voice_style: Voice style for captions
            week_start: Week label (YYYY-MM-DD) for caching

        Returns:
            CaptionBatch with generated captions (reviewed=False)
        """
        if not week_start:
            week_start = datetime.now().strftime("%Y-%m-%d")

        # No-clobber backstop for direct callers — the orchestrator checks this
        # BEFORE selection, so a doomed run never takes photo holds.
        cache_path = self.ensure_week_unwritten(week_start)

        # Filter out selections with empty/useless metadata
        valid_selections = []
        for sel in selections:
            if _is_empty_metadata(sel):
                logger.warning(f"Skipping {sel.filename}: all metadata fields empty/unknown")
            else:
                if sel.location == "Unknown Location":
                    logger.warning(
                        f"Photo {sel.filename} has 'Unknown Location' \u2014 "
                        f"using journey/summary context only"
                    )
                valid_selections.append(sel)

        if not valid_selections:
            logger.error("No valid selections after metadata filtering")
            return CaptionBatch(
                week_start=week_start,
                voice_style=voice_style.value,
                captions=[],
                reviewed=False,
                generated_at=datetime.now().isoformat(),
                model=self.model,
            )

        prompt = _build_prompt(valid_selections, voice_style)

        logger.info(
            f"Generating {len(valid_selections)} captions with voice={voice_style.value} "
            f"via {self.endpoint}"
        )

        # One batched call through the Claude CLI
        batch_call_start = time.perf_counter()
        response = self._call_llm(prompt)
        batch_call_duration = time.perf_counter() - batch_call_start
        captions_data = self._parse_response(response)

        # A short LLM batch would be silently truncated by zip(), dropping photos
        # with no caption. Fail loudly instead of shipping an under-count batch.
        if len(captions_data) != len(valid_selections):
            raise ValueError(
                f"LLM returned {len(captions_data)} caption objects for "
                f"{len(valid_selections)} selections (count mismatch)"
            )

        captions = []
        for i, (data, sel) in enumerate(zip(captions_data, valid_selections)):
            # Validate all fields (grounding + stats + blocklist).
            failed_fields = _validate_fields(data, sel)

            # Retry up to 2 times if any field fails
            retries = 0
            retry_duration = 0.0
            while failed_fields and retries < 2:
                retries += 1
                logger.info(
                    f"Retry {retries}/2 for {sel.filename}: "
                    f"{', '.join(failed_fields.keys())} failed"
                )
                retry_start = time.perf_counter()
                retry_data = self._retry_caption(sel, voice_style, failed_fields, data)
                retry_duration += time.perf_counter() - retry_start
                if retry_data:
                    # Merge retried fields, then re-validate everything.
                    for field_name in list(failed_fields.keys()):
                        if field_name in retry_data:
                            data[field_name] = retry_data[field_name]
                    failed_fields = _validate_fields(data, sel)
                else:
                    break

            main_caption = data.get("main_caption", "")[:250]
            tiktok_caption = data.get("tiktok_caption", "")[:90]
            hook = data.get("first_comment_hook", "")[:100]

            # Fallback for fields still failing after retries
            if "main_caption" in failed_fields:
                logger.warning(
                    f"Caption validation failed for {sel.filename} after retries: "
                    f"{'; '.join(failed_fields['main_caption'])}. Using fallback."
                )
                main_caption = _fallback_caption(sel)
            if "tiktok_caption" in failed_fields:
                logger.warning(f"TikTok caption fallback for {sel.filename}")
                tiktok_caption = (
                    sel.location if sel.location != "Unknown Location" else sel.journey_title
                )[:90]
            if "first_comment_hook" in failed_fields:
                logger.warning(f"Hook fallback for {sel.filename}")
                hook = ""

            fallback_used = bool(failed_fields)
            fallback_reason = (
                "; ".join(
                    f"{field_name}: {'; '.join(issues)}"
                    for field_name, issues in failed_fields.items()
                )
                if fallback_used
                else None
            )
            checked_fields = ("main_caption", "tiktok_caption", "first_comment_hook")
            provenance = GenerationProvenance(
                recorded=True,
                model=self.model,
                recipe_version=RECIPE_VERSION,
                evidence=_evidence_keys(sel),
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                attempts=1 + retries,
                duration_s=batch_call_duration + retry_duration,
                validation_passed=tuple(f for f in checked_fields if f not in failed_fields),
                validation_failed=tuple(failed_fields.keys()),
            )

            captions.append(
                PhotoCaption(
                    photo_index=i,
                    main_caption=main_caption,
                    tiktok_caption=tiktok_caption,
                    hashtags=_sanitize_hashtags(data.get("hashtags", [])),
                    journey_slug=sel.journey_slug,
                    filename=sel.filename,
                    location=sel.location,
                    first_comment_hook=hook,
                    provenance=provenance.to_dict(),
                )
            )

        batch = CaptionBatch(
            week_start=week_start,
            voice_style=voice_style.value,
            captions=captions,
            reviewed=False,
            generated_at=datetime.now().isoformat(),
            model=self.model,
        )

        # Cache to YAML
        batch.save(cache_path)

        return batch

    def _build_retry_prompt(
        self,
        selection: PhotoSelection,
        voice_style: VoiceStyle,
        failed_fields: dict,
        original_data: dict,
    ) -> str:
        """Build a single-caption retry prompt with rejection feedback."""
        style_instruction = VOICE_PROMPTS[voice_style]

        safe_journey_title = _sanitize_prompt_field(selection.journey_title, max_len=200)
        safe_location = _sanitize_prompt_field(selection.location, max_len=200)
        safe_locations_visited = [
            _sanitize_prompt_field(loc, max_len=200) for loc in selection.locations_visited
        ]
        ctx = (
            f"Journey: {safe_journey_title}\n"
            f"Location: {safe_location}\n"
            f"Day {selection.day_number} ({selection.day_date})\n"
            f"Places visited: {', '.join(safe_locations_visited)}\n"
        )
        if selection.day_summary:
            safe_day_summary = _sanitize_prompt_field(selection.day_summary, max_len=1000)
            ctx += f"Day summary: {safe_day_summary}\n"
        if selection.statistics:
            stats = selection.statistics
            if stats.get("distance_km"):
                ctx += f"Distance: {_safe_stat(stats['distance_km'])} km\n"
            if stats.get("duration_hours"):
                ctx += f"Duration: {_safe_stat(stats['duration_hours'])} hours\n"
            if stats.get("photo_count"):
                ctx += f"Photos taken: {_safe_stat(stats['photo_count'])}\n"

        rejection_lines = []
        fields_to_rewrite = []
        for field_name, issues in failed_fields.items():
            # original_data is first-pass LLM output — sanitize before re-embedding
            # so fenced/chat-template tokens can't bypass the injection guard the
            # selection-metadata fields above already pass through.
            rejected_text = _sanitize_prompt_field(original_data.get(field_name, ""), max_len=250)
            rejection_lines.append(
                f'REJECTED {field_name}: "{rejected_text}"\n  Reason: {"; ".join(issues)}'
            )
            fields_to_rewrite.append(field_name)

        rejections = "\n".join(rejection_lines)
        fields_list = ", ".join(f'"{f}"' for f in fields_to_rewrite)

        return f"""Rewrite the rejected caption field(s) for this travel photo.

VOICE STYLE: {voice_style.value}
{style_instruction}

PHOTO CONTEXT:
{ctx}

{rejections}

HARD RULES FOR REWRITE:
- Focus on the PLACE. The caption must work with zero statistics.
- At most one stat, and it must NOT lead the caption.
- Do NOT mention elevation.
- Only reference locations from the metadata above.

Return ONLY valid JSON with the field(s) {fields_list}:
{{
{chr(10).join(f'  "{f}": "..."' for f in fields_to_rewrite)}
}}"""

    def _retry_caption(
        self,
        selection: PhotoSelection,
        voice_style: VoiceStyle,
        failed_fields: dict,
        original_data: dict,
    ) -> Optional[dict]:
        """Retry generation for failed caption fields. Returns dict or None."""
        prompt = self._build_retry_prompt(selection, voice_style, failed_fields, original_data)
        try:
            response = self._call_llm(prompt)
            text = response.strip()
            # Strip markdown fences
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                text = "\n".join(lines)
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            if isinstance(result, list) and result:
                return result[0]
            logger.warning(f"Retry returned unexpected type: {type(result)}")
            return None
        except Exception as e:
            logger.warning(f"Retry failed: {e}")
            return None

    def _call_llm(self, prompt: str) -> str:
        """Call the local Claude CLI and return raw response text.

        Retries once after a 2-second backoff on failures.

        The prompt is passed via STDIN (``claude -p`` reads stdin when no
        prompt argv is supplied) rather than as an argv element, so it is
        not exposed in ``ps`` / ``/proc/<pid>/cmdline`` to other users.
        """
        import subprocess

        argv = ["claude", "-p", "--model", self.model, "--output-format", "text"]
        argv += ["--permission-mode", "default"]

        # A stray ANTHROPIC_API_KEY/AUTH_TOKEN silently flips the CLI from
        # its signed-in login to API-key auth — scrub both.
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        }

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                # Own process group so a timeout can reap claude's node/MCP
                # descendants too, not just the direct child.
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                    env=env,
                )
                try:
                    stdout, stderr = proc.communicate(input=prompt, timeout=600)
                except subprocess.TimeoutExpired:
                    # Orphaned descendants would otherwise outlive the timeout
                    # and stack with the retry below. Reap is best-effort.
                    with contextlib.suppress(Exception):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait(timeout=5)
                    raise

                if proc.returncode != 0:
                    raise RuntimeError(f"claude -p exit {proc.returncode}: {stderr[:500]}")

                content = stdout.strip()
                logger.debug(f"LLM response length: {len(content)} chars")
                return content
            except (subprocess.TimeoutExpired, RuntimeError, OSError) as e:
                last_error = e
                if attempt == 0:
                    logger.warning(f"LLM call failed ({e}), retrying in 2s...")
                    time.sleep(2)

        raise last_error  # type: ignore[misc]

    def _parse_response(self, response: str) -> List[dict]:
        """Parse JSON array from LLM response, handling markdown fences."""
        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                # A non-dict element would reach _validate_fields and die on
                # data.get(...) with a bare AttributeError. Reject it here.
                for idx, item in enumerate(result):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"LLM caption element {idx} is "
                            f"{type(item).__name__}, expected a JSON object"
                        )
                return result
            raise ValueError(f"Expected JSON array, got {type(result)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {text[:500]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
