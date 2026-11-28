### Fixed

- A YouTube or Drive upload chunk that runs past its timeout is now stopped before the upload retries. Before, the timed-out chunk was left sending in the background, and the retry sent the same chunk again beside it on the same connection. If the chunk cannot be stopped, the upload fails instead of retrying beside it.
- YouTube and Drive uploads now send 8 MB chunks, down from 256 MB. The 60 s socket timeout bounds a chunk's whole send, so a 256 MB chunk needed about 36 Mbit/s of uplink or it timed out on every retry. An 8 MB chunk needs about 1.1 Mbit/s.
