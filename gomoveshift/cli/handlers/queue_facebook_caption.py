"""``gomoveshift queue edit-facebook-caption``: rewrite a published Reel's description.

Thin CLI over ``publishing.facebook_caption_edit``, which builds the caption from the Short's
measured county (typed text only when the record has none) and owns the platform call.
"""

from __future__ import annotations

import sys

from ...publishing.facebook_caption_edit import CaptionEditRefusedError, edit_facebook_caption


def cmd_edit_facebook_caption(args, config) -> int:
    try:
        lines = edit_facebook_caption(
            config.seo.storage_path,
            getattr(config.seo, "credentials_dir", None),
            args.id,
            args.text,
            apply=args.apply,
        )
    except CaptionEditRefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0
