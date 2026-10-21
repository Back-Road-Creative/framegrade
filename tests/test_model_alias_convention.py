"""Every ``claude -p --model`` default in the service uses a model alias.

A default names an alias (``sonnet``, ``opus`` or ``haiku``) and never a dated
model id. The alias follows whichever model the logged-in session offers, while
a dated id rots the moment the alias moves on. ``caption_generator.py`` already
gets this right (``DEFAULT_MODEL = "sonnet"``).

Scope is deliberately narrow: ``tools/**.py``, ``gomoveshift/**.py`` and the
shipped root ``config.yaml``, the places a *default* lives. Provenance records
that name the exact model which produced a batch (test fixtures, audit notes)
are not defaults and are intentionally out of scope.
"""

import re
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]

MODEL_ALIASES = frozenset({"sonnet", "opus", "haiku"})

# A dated / versioned API model id: ``claude-sonnet-4-5``, ``claude-opus-4-20250514``.
_DATED_ID = re.compile(r"claude-[a-z]+-\d[\w.-]*")

# ``DEFAULT_MODEL = "x"`` / ``model: str = "x"`` (python) and ``model: "x"`` (yaml).
_MODEL_DEFAULT = re.compile(
    r"""^\s*(?:DEFAULT_MODEL|[A-Za-z_]*model)\s*(?::\s*[^=:]+)?[=:]\s*["']([^"']*)["']"""
)


def _scoped_files() -> list[Path]:
    files: list[Path] = []
    for sub in ("tools", "gomoveshift"):
        files.extend(sorted((SERVICE_ROOT / sub).rglob("*.py")))
    files.append(SERVICE_ROOT / "config.yaml")
    return files


def _hits(matcher) -> list[str]:
    found: list[str] = []
    for path in _scoped_files():
        rel = path.relative_to(SERVICE_ROOT.parents[1])
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            note = matcher(line)
            if note is not None:
                found.append(f"{rel}:{lineno}: {note}")
    return found


def test_no_dated_api_model_id_is_pinned():
    """No default anywhere in scope names a dated API model id."""

    def matcher(line: str) -> str | None:
        m = _DATED_ID.search(line)
        return None if m is None else f"pins {m.group(0)!r} — use a model alias"

    offenders = _hits(matcher)
    assert not offenders, "dated API model id pinned:\n  " + "\n  ".join(offenders)


def test_model_defaults_use_a_model_alias():
    """Every literal model default is a model alias (or empty, i.e. supplied
    at runtime)."""

    def matcher(line: str) -> str | None:
        m = _MODEL_DEFAULT.match(line)
        if m is None:
            return None
        value = m.group(1)
        if value == "" or value in MODEL_ALIASES:
            return None
        return f"model default {value!r} is not one of {sorted(MODEL_ALIASES)}"

    offenders = _hits(matcher)
    assert not offenders, "non-alias model default:\n  " + "\n  ".join(offenders)
