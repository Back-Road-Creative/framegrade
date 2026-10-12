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
