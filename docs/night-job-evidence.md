# Night worker outcome evidence

The console reconciles recognized `photos night --project … --workspace … --brand …`
jobs against its configured `night-projects/projects.sqlite3` after their recorded
worker disappears. Durable complete becomes done, failed/partial/blocked becomes
failed, and paused/waiting retain those nonfailure states. Return code 2 from a
reaped worker also uses a matching paused/waiting record without failure notices.

The row brand, argv brand and current console brand must agree. The interpreter,
command, three unique flags, project identifier and configured workspace must match.
The reader refuses symlinks, opens only the configured DB read-only, and caps JSON
record size. Missing, ambiguous or nonterminal evidence keeps normal interrupted
handling; a live matching worker is never overridden. Other job types are unchanged.

`outcome_evidence=night_project:<state>` distinguishes a recorded worker outcome
from an exit code observed directly by this console. No media is decoded or hashed
in the ticker. Completion here is historical worker evidence, not a claim that
output files remain present or unmodified forever. Private preview serving and
worker resume validation retain their separate integrity checks.
