### Fixed

- `health heal` no longer waits for a new mirror SHA after a resume that died in drive pre-flight (for example, too little staging disk) before any encode. The attempt is recorded as `preflight_failed` with the first blocker line, the outcome notice says "failed pre-flight: <blocker>", and the next tick relaunches at the same SHA, at most 8 times per recipe per SHA (#373).
