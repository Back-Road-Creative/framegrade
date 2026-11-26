### Fixed

- Resuming a platform that was paused for a refused access token now sends the post that hit the refusal on the next pass, instead of 6 hours later.
- A refused Facebook, Instagram or TikTok token met while reconciling parked posts or posting first comments now pauses that platform too, instead of failing each row in turn.
- An expired TikTok refresh token no longer stops the whole `dispatch-queue` pass: TikTok pauses (and `health check` names it) while YouTube, Instagram and Facebook still drain. Any other TikTok refresh failure skips only TikTok for that pass.
