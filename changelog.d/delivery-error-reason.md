### Fixed

- A failed Instagram delivery now names the Graph request that failed
  (`op=create_container` and the like) and, when Graph returns no error
  object, keeps a short redacted excerpt of the response body instead of
  "unknown Graph API error". Retry and terminal handling are unchanged.
- `deliveries.error_detail` keeps up to 2000 characters (was 500), so a
  final-text refusal keeps every reason code.
