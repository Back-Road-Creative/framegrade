### Fixed

- `health backlog --supersede` now re-renders a drive whose own passed audit cannot publish: a `failed` row whose publish retry is refused, or an `auditing` drive whose publish (asked as a dry run) is refused, is set aside and relaunched. A passed audit that could publish is still refused. On 2026-11-27 the blur-mask gate asked for Kerry-to-Cork and Old Head to Courtmacsherry to be re-rendered, and the supersede refused both.
