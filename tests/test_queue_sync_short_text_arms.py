"""queue sync-short-text refuses a Short whose stored Shorts arm the regenerated text drops (C5b-6).

Kept apart from ``test_queue_sync_short_text.py`` so the arm rule has its own file; the
fixtures and the world builder are that module's.
"""

from __future__ import annotations

import json

from test_queue_sync_short_text import (  # noqa: F401  (autouse fixture)
    NOW,
    OLD_HASH,
    _hashes,
    _no_published_titles,
    _short,
    _world,
)

from gomoveshift.publishing import queue_short_text as qst
from gomoveshift.publishing.queue_short_text import sync_short_text


def test_a_short_whose_stored_arm_can_no_longer_render_is_refused(tmp_path, monkeypatch):
    real = qst._generate_lf

    def drops_the_arm(*a, **kw):
        posts = real(*a, **kw)
        for p in posts:
            p.experiment_arms = {}
        return posts

    monkeypatch.setattr(qst, "_generate_lf", drops_the_arm)
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_hashtag_set": "road"}), _short(2)])
    before = json.loads(plan.read_text())["shorts"][0]
    changes = sync_short_text(conn, apply=True, now=NOW)
    assert [c.status for c in changes] == ["refused", "changed"]
    assert "yt_hashtag_set" in changes[0].reason
    assert json.loads(plan.read_text())["shorts"][0] == before
    assert _hashes(conn)[(1, "youtube")] == OLD_HASH


def test_an_unknown_stored_arm_id_never_refuses_the_short(tmp_path, monkeypatch):
    real = qst._generate_lf

    def drops_the_arm(*a, **kw):
        posts = real(*a, **kw)
        for p in posts:
            p.experiment_arms = {}
        return posts

    monkeypatch.setattr(qst, "_generate_lf", drops_the_arm)
    plan, conn = _world(tmp_path, [_short(1, arms={"yt_hashtag_set": "not-an-arm"})])
    [change] = sync_short_text(conn, apply=True, now=NOW)
    assert change.status == "changed"
