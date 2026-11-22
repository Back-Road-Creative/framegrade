"""``gomoveshift queue edit-facebook-caption``: rewrite a published Reel's description.

Thin CLI over ``publishing.facebook_caption_edit``, which builds the caption from the Short's
measured county (typed text only when the record has none) and owns the platform call.
"""

from __future__ import annotations

import sys

from ...publishing.facebook_caption_edit import CaptionEditRefusedError, edit_facebook_caption


def _experiments():
    """The brand's caption experiments, so an edit keeps the Short's arm treatment."""
    try:
        from ...console.settings import current_brand
        from ...publishing.brand_policy import load_brand_policy

        return load_brand_policy(current_brand()).experiments
    except Exception:  # noqa: BLE001 - no policy readable: the edit keeps its untagged form
        return ()


def cmd_edit_facebook_caption(args, config) -> int:
    try:
        lines = edit_facebook_caption(
            config.seo.storage_path,
            getattr(config.seo, "credentials_dir", None),
            args.id,
            args.text,
            apply=args.apply,
            experiments=_experiments(),
        )
    except CaptionEditRefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0
