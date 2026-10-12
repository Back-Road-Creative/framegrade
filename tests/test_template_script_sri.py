"""Guard: external CDN <script> tags in web templates carry SRI or are a documented exception.

The review UI binds 127.0.0.1 only, but a tampered CDN response still executes in the
operator's browser. Version-pinned, SRI-capable scripts must carry an ``integrity=`` hash.
The unversioned Tailwind JIT CDN (``cdn.tailwindcss.com``) cannot have a stable hash and is
the one allowed exception. Same-origin (relative / static) scripts need no SRI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TEMPLATES = Path(__file__).resolve().parent.parent / "gomoveshift" / "web" / "templates"

# Unversioned JIT builds with no stable hash — SRI is genuinely N/A.
_SRI_EXEMPT_HOSTS = ("https://cdn.tailwindcss.com",)

_SCRIPT_SRC = re.compile(r"<script\b[^>]*\bsrc=\"([^\"]+)\"[^>]*>", re.IGNORECASE)


def _template_files() -> list[Path]:
    return sorted(_TEMPLATES.glob("*.html"))


def test_templates_exist():
    assert _template_files(), f"no templates found under {_TEMPLATES}"


@pytest.mark.parametrize("template", _template_files(), ids=lambda p: p.name)
def test_external_scripts_carry_sri_or_documented_exception(template: Path):
    html = template.read_text(encoding="utf-8")
    for match in _SCRIPT_SRC.finditer(html):
        tag, src = match.group(0), match.group(1)
        if not src.startswith(("http://", "https://")):
            continue  # same-origin / relative — SRI not required
        if any(src.startswith(host) for host in _SRI_EXEMPT_HOSTS):
            assert 'crossorigin="anonymous"' in tag, (
                f"SRI-exempt external script in {template.name} must still set crossorigin: {src}"
            )
            continue
        assert "integrity=" in tag, (
            f"external script in {template.name} lacks integrity= (SRI): {src}"
        )
        assert 'crossorigin="anonymous"' in tag, (
            f"external script in {template.name} with integrity= must set crossorigin: {src}"
        )
