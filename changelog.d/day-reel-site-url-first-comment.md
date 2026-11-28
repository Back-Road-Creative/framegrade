### Added

- Every day reel shows `gomoveshift.com` small at the bottom right for the whole reel, clear of the Instagram and YouTube button column. It is burned in by the day reel renderer, so the Drive requests and any direct enqueue both get it. The Day 12 reels went out without it.
- Every day reel gets its journal day link as the first comment on YouTube and Facebook. The route writes it on the sidecar record (`first_comment_by_platform`), and the queue's existing comment step posts it. Instagram gets no comment, because links there are not clickable. Before, a day reel had no full drive to link, so it got no first comment at all.
