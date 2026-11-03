### Fixed

- The pre-encode location invariant now accepts a state label, such as `North Carolina`, when that one state holds every material county. That label is the route gate's own fallback when no county can be proven. The invariant used to refuse it as "omits Henderson County", so two gates contradicted each other and the drive stopped before encode.
