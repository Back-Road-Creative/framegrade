### Fixed

- Facebook photo and carousel queue entries no longer use the Instagram media
  client or fall through to the Facebook Reels uploader. Unsupported Facebook
  photo delivery fails before reading media or calling an upload adapter;
  Instagram photo and Facebook video routes remain available.
