# Pipeline Internals — implementation reference

> **This file is reference, not contract.** It documents *how* the pipeline works
> (function names, constants, thresholds, mechanism) for when you're debugging or
> extending a subsystem. The README holds the operative
> do/avoid rules; this is the detail behind them.
>
> Because this documents code internals, it drifts when code is renamed. Treat the
> source as truth; if this disagrees with the code, the code wins — fix this file.
> When you change behavior, update the one-line rule in README.md; update
> this file only if the mechanism a future debugger needs has changed.

> **Generated.** This file is rebuilt by `tools/build_internals_index.py` from its preamble and checks the heading of every file under `docs/pipeline/`. Do not hand-edit it -- edit the module file instead and re-run the generator (a test checks this file matches its output). Module boundary: each `docs/pipeline/NN-*.md` holds one `##`/`###` section.

The sections are the files under [`docs/pipeline/`](pipeline/), one per module, numbered in reading order -- browse that directory or `grep -rl '<topic>' docs/pipeline/`. This index deliberately lists none of them, so adding a module never edits this file.
