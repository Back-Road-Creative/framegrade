# Documentation index

Start with [install.md](install.md), then the [project README](../README.md)
for the command and environment reference. Pages are grouped by what they
cover. The numbered notes in [pipeline/](pipeline/) are the per-feature design
and behaviour records; [pipeline/_preamble.md](pipeline/_preamble.md) says how
to read them.

## Install and operate

- [install.md](install.md): local install, CPU-only install, Docker console.
- [gpu-checks.md](gpu-checks.md): what hosted CI cannot prove about GPU paths, and the checks to run on a GPU host.
- [host-worker.md](host-worker.md): the host GPU worker that runs requests the console container spools.
- [outbound.md](outbound.md): every outbound host the code names, and why.
- [console-inventory.md](console-inventory.md): what each console page and action does and what backs it.

## Pipeline reference

- [pipeline-internals.md](pipeline-internals.md): implementation reference for debugging a stage.
- [pipeline-graph.md](pipeline-graph.md): the generated stage and phase graph.
- [unattended-cadence.md](unattended-cadence.md): where publish slots come from.
- [w9c2-final-text-bind.md](w9c2-final-text-bind.md): the final-text byte-bind on publication.
- [w9c3-short-tags.md](w9c3-short-tags.md): which tags a Short may carry.
- [pipeline/](pipeline/): the numbered feature notes, from location labels (01) to the tracked Short reframe (170). Each number is used once; the GPU grade chain spike (159) and the blur composite ablation (154) are the measured spikes.

## Route Story

- [route-story-sources.md](route-story-sources.md): free public map source fetchers and their licences.
- [route-story-bundle.md](route-story-bundle.md): the offline map asset bundle.
- [route-story-planner.md](route-story-planner.md): the renderer-independent shot planner.
- [route-story-scene.md](route-story-scene.md): the scene page and capture harness.

## Photos

- [astro-quality-contract.md](astro-quality-contract.md): measure astro quality on a synthetic product, not by eye.
- [astro-natural-recipe.md](astro-natural-recipe.md): the Natural, Accent and Bold presets.
- [style-groundtruth.md](style-groundtruth.md): how the photo style picker is scored.
- [night-projects.md](night-projects.md): private night projects.
- [night-render.md](night-render.md): rendering a verified linear stack.
- [night-storage.md](night-storage.md): read-only night project storage.
- [night-library.md](night-library.md): night projects in the media library.
- [night-job-evidence.md](night-job-evidence.md): night worker outcome evidence.

## Social publishing

- [social-publishing-status.md](social-publishing-status.md): what works today, per platform.
- [social-publishing-plan-contract.md](social-publishing-plan-contract.md): the publish-plan records.
- [social-publishing-plan-builder.md](social-publishing-plan-builder.md): building a plan from Phase D inputs.
- [social-publishing-payload-mapping.md](social-publishing-payload-mapping.md): CSV row to platform payload.
- [social-publishing-queue.md](social-publishing-queue.md): the durable delivery queue.
- [social-publishing-dispatch.md](social-publishing-dispatch.md): the drain loop.
- [social-publishing-youtube.md](social-publishing-youtube.md): YouTube Shorts adapter.
- [social-publishing-meta.md](social-publishing-meta.md): Meta credentials and publishing.
- [social-publishing-facebook.md](social-publishing-facebook.md): Facebook Reels adapter.
- [social-publishing-tiktok.md](social-publishing-tiktok.md): TikTok Direct Post adapter.

## Social metrics

- [social-metrics-observations.md](social-metrics-observations.md): comparable metrics versus honestly missing ones.
- [social-metrics-long-form.md](social-metrics-long-form.md): long-form tracking.
- [social-metrics-volume.md](social-metrics-volume.md): volume and cadence are decided by Waypost.
- [social-metrics-best-times.md](social-metrics-best-times.md): the best-time table.
- [social-metrics-decline.md](social-metrics-decline.md): platform decline detection.
- [social-metrics-anomaly.md](social-metrics-anomaly.md): post-level anomaly detection.
- [facebook-metrics.md](facebook-metrics.md): Facebook Reel plays and watch time, kept apart from video views.
