### Fixed

- A YouTube long-form upload now gets its 5 retries for each 256 MB chunk. Before, every chunk that uploaded used up one of the 5, so an LF over 1.25 GB failed on its first network timeout with no retry. Kenmare (about 8 GB) failed this way twice.
