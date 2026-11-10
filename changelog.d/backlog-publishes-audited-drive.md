### Added

- `health backlog` publishes its region's audited drive: when the `auditing` drive's newest audit succeeded, the hourly tick runs `health publish` for it. A slot still outside Waypost's lead window is a noop ("audit passed: ... waits for its Waypost slot"), retried next tick; any other publish refusal is a refusal. Before, an audited drive held its region's slot ("audit in progress") until a person ran `health publish` inside the window.
