### Fixed

- The backlog's publish gate wrote its blur-mask rows under the code checkout (the staging mirror), not the tick's `--data-root`, so the backlog never saw them and never re-rendered Kerry-to-Cork or Old Head to Courtmacsherry. `health publish` and the backlog republish now write them into the tick's own incidents dir, where the re-render reads them.
