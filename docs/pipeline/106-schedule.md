### Schedule (`console/schedule.py`, `GET /schedule`)

Read-only view over `pipeline.schedule_report.build_schedule_report()` -- what is booked, when
the queue drains, and the next free long-form/superlong slot per region -- mirroring
`cli/handlers/schedule.py::cmd_schedule`'s ledger+config construction (`ContentLedger.load()` +
`Config.load()`), never a second scheduler. `schedule_view()` degrades any ledger/config failure
(missing file, bad config) to an empty report plus a rendered `hint`, the same never-500 shape as
`pubqueue.queue_rows()`'s missing-DB-file case. The template renders the booked long-forms table,
the shorts count and queue-ends line, the next-available matrix (one row per region/designed day,
soonest marked), and the next-available-superlong table.
