"""Voice style definitions for photo caption A/B testing.

Four voice styles are rotated per-week to create clean A/B cohorts.
Performance is tracked via the SEO manager's ThemeRotator with
category="photo".
"""

from enum import Enum
from typing import Dict


class VoiceStyle(Enum):
    """Caption voice styles for photo posts."""

    FIRST_PERSON = "voice_first_person"
    MINIMAL_ZEN = "voice_minimal_zen"
    ENGAGING_CONVERSATIONAL = "voice_engaging"
    INTERACTIVE = "voice_interactive"


# XML-structured prompt fragments for each voice style.
# Each prompt uses semantic XML tags per Anthropic's context engineering guidance:
# - <voice_style> wraps the full prompt for a style
# - <instruction> contains the behavioral directive
# - <examples> with <example> children provide few-shot grounding
VOICE_PROMPTS: Dict[VoiceStyle, str] = {
    VoiceStyle.FIRST_PERSON: (
        "<voice_style>\n"
        "<instruction>\n"
        "Write as if recalling the moment from memory. Use past tense. "
        "Reference only the locations and facts provided. Do not invent sensory details. "
        "Be authentic and personal, not polished or promotional.\n"
        "</instruction>\n"
        "<examples>\n"
        "<example>Day three on the Connemara coast. 120 km through Clifden and Roundstone.</example>\n"
        "<example>Molas Pass on day one. Arrastra Gulch and the mountain roads.</example>\n"
        "</examples>\n"
        "</voice_style>"
    ),
    VoiceStyle.MINIMAL_ZEN: (
        "<voice_style>\n"
        "<instruction>\n"
        "Extremely short. Place name and maybe one observation. "
        "Use only the provided place name. "
        "No CTAs, no emojis, no hashtags in the caption itself. "
        "Let the photo speak. Think haiku energy.\n"
        "</instruction>\n"
        "<examples>\n"
        "<example>Molas Pass. Still air.</example>\n"
        "<example>Connemara. The green after rain.</example>\n"
        "<example>Cliffs of Moher. Nothing else to say.</example>\n"
        "</examples>\n"
        "</voice_style>"
    ),
    VoiceStyle.ENGAGING_CONVERSATIONAL: (
        "<voice_style>\n"
        "<instruction>\n"
        "Lead with a specific, surprising detail from the metadata — a distance, "
        "a place name most people haven't heard of, or an unexpected contrast. "
        "Then add one short personal reflection in past tense. "
        "The goal is to make the reader curious, not to ask them a question. "
        "No generic CTAs ('Would you?', 'Have you ever?'). No emojis. "
        "Only reference places and details from the metadata. "
        "If the detail is interesting enough, people will comment on their own.\n"
        "</instruction>\n"
        "<examples>\n"
        "<example>Arrastra Gulch doesn't show up on most Colorado maps. Day one of the San Juans and we were already lost.</example>\n"
        "<example>198 km of Donegal peninsula roads. Not a single other car for the last 40.</example>\n"
        "<example>Clifden to Roundstone is only 25 km but it took us two hours. Kept pulling over.</example>\n"
        "</examples>\n"
        "</voice_style>"
    ),
    VoiceStyle.INTERACTIVE: (
        "<voice_style>\n"
        "<instruction>\n"
        "Tell a micro-story from the road using only the metadata provided. "
        "Focus on the place — its name, what makes it distinctive, where it sits on the journey. "
        "A single statistic can add texture, but the caption should work without it. "
        "Write like a journal entry someone would actually post — not a brand. "
        "Avoid these formats: 'Rate 1-10', 'Tag someone', 'Guess the location', "
        "'Would you stop here?', 'Drop your answer', or any direct call-to-action. "
        "End with a statement, not a question. The specificity is what drives comments. "
        "Only reference places and details from the metadata.\n"
        "</instruction>\n"
        "<examples>\n"
        "<example>Day three. Connemara coast. 120 km and we still hadn't left County Galway.</example>\n"
        "<example>The road between Molas Pass and Silverton drops hard and fast. We drove it twice.</example>\n"
        "<example>January in County Clare. O'Dea's was the only place open. We stayed three hours.</example>\n"
        "<example>Four peninsula loops in one day across Donegal. The last one had no name on the map.</example>\n"
        "</examples>\n"
        "</voice_style>"
    ),
}

# Kept for backward compatibility — examples are now embedded in VOICE_PROMPTS XML.
# Consumers that read VOICE_EXAMPLES directly still work.
VOICE_EXAMPLES: Dict[VoiceStyle, list] = {
    VoiceStyle.FIRST_PERSON: [
        "Day three on the Connemara coast. 120 km through Clifden and Roundstone.",
        "Molas Pass on day one. Arrastra Gulch and the mountain roads.",
    ],
    VoiceStyle.MINIMAL_ZEN: [
        "Molas Pass. Still air.",
        "Connemara. The green after rain.",
        "Cliffs of Moher. Nothing else to say.",
    ],
    VoiceStyle.ENGAGING_CONVERSATIONAL: [
        "Arrastra Gulch doesn't show up on most Colorado maps. Day one of the San Juans and we were already lost.",
        "198 km of Donegal peninsula roads. Not a single other car for the last 40.",
        "Clifden to Roundstone is only 25 km but it took us two hours. Kept pulling over.",
    ],
    VoiceStyle.INTERACTIVE: [
        "Day three. Connemara coast. 120 km and we still hadn't left County Galway.",
        "The road between Molas Pass and Silverton drops hard and fast. We drove it twice.",
        "January in County Clare. O'Dea's was the only place open. We stayed three hours.",
        "Four peninsula loops in one day across Donegal. The last one had no name on the map.",
    ],
}

# Mapping from VoiceStyle to SEO manager theme names (for ThemeRotator category="photo")
VOICE_TO_THEME_NAME: Dict[VoiceStyle, str] = {
    VoiceStyle.FIRST_PERSON: "voice_first_person",
    VoiceStyle.MINIMAL_ZEN: "voice_minimal_zen",
    VoiceStyle.ENGAGING_CONVERSATIONAL: "voice_engaging",
    VoiceStyle.INTERACTIVE: "voice_interactive",
}
