"""Frozen legacy short-title pool -- test-side handle.

These are the ``TITLE_TEMPLATES`` / ``EXTENDED_TITLE_TEMPLATES`` /
``_pick_tiktok_title`` template strings that ``social_generator.py`` used
before composed titles replaced them. The strings themselves live in
``gomoveshift.publishing.anti_spam_gate.LEGACY_TITLE_POOL_RAW`` (the anti-spam gate refuses
any batch whose shipped title matches one of them); this module only
re-exports that tuple as a list so tests keep their historical import path.
Never a second copy of the literal -- one source of truth.
"""

from gomoveshift.publishing.anti_spam_gate import LEGACY_TITLE_POOL_RAW

LEGACY_TITLE_POOL: list[str] = list(LEGACY_TITLE_POOL_RAW)
