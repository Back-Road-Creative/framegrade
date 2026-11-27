### Fixed

- A refused token met after an upload went through now pauses the platform too: YouTube's post-upload check and playlist assign no longer log it and publish the Short unchecked, and a Facebook reel's read-back after finish pauses Facebook at once instead of a pass later. The row keeps its upload id and is reconciled, never uploaded again, once the platform is resumed.
