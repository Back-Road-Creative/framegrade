### Fixed

- A refused Facebook Page token now pauses Facebook when the Page photo upload or its first comment, the Reel delivery's upload or reconcile read, or the read-back of a just-posted first comment meets it. Before, each held the post as an unknown outcome, or logged the refusal and went on. Held attempts stay held, and no write is repeated.
