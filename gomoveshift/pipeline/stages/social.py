"""Stage: Social post generation."""

import json
import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class SocialGenerationStage(BaseStage):
    name = PipelineContext.STAGE_SOCIAL_GENERATION

    def run(self, ctx):
        from .. import orchestrator as _orch

        SocialGenerator = _orch.SocialGenerator  # noqa: N806
        output_dir = ctx.paths.project_dir
        try:
            shorts_data = ctx.shorts_data
            if shorts_data is None:
                af = output_dir / "shorts" / "shorts_analysis.json"
                if af.exists():
                    with open(af) as f:
                        shorts_data = json.load(f)
            weather_results = None
            shorts_dir = output_dir / "shorts"
            wf = shorts_dir / "weather_results.json"
            if wf.exists():
                try:
                    with open(wf) as f:
                        wd = json.load(f)
                    from ..weather import WeatherCondition, WeatherResult

                    weather_results = [
                        WeatherResult(
                            condition=WeatherCondition(w["condition"]),
                            brightness=w["brightness"],
                            confidence=w["confidence"],
                            segment_start=w["segment_start"],
                            segment_end=w["segment_end"],
                        )
                        for w in wd
                    ]
                except Exception as exc:
                    logger.debug("Weather result loading failed: %s", exc)
            if weather_results is None and shorts_data:
                try:
                    from ..weather import WeatherDetector

                    # config.drive.weather_chunk_* are the knobs the
                    # degraded-brightness remediation message names;
                    # thread them so editing them actually takes effect.
                    _drive = ctx.config.drive
                    det = WeatherDetector(
                        chunk_max_sec=_drive.weather_chunk_max_sec,
                        chunk_timeout_cap_sec=_drive.weather_chunk_timeout_cap_sec,
                    )
                    weather_results = [
                        w
                        for seg in shorts_data["segments"]
                        if (
                            w := det.detect_segment(
                                ctx.current_video, seg["start_time"], seg["end_time"]
                            )
                        )
                        is not None
                    ]
                    shorts_dir.mkdir(parents=True, exist_ok=True)
                    with open(wf, "w") as f:
                        json.dump([w.to_dict() for w in weather_results], f, indent=2)
                except Exception as exc:
                    logger.debug("Weather detection failed: %s", exc)
                    weather_results = None
            rd_short = ctx.recording_date.strftime("%b %Y") if ctx.recording_date else None
            sp = None
            if ctx.metadata and ctx.metadata.get("series"):
                sp = ctx.metadata["series"].get("part")
            if sp is None:
                import re

                m = re.match(r"episode_(\d+)", ctx.project_name)
                if m:
                    sp = int(m.group(1))
            if sp is None:
                sp = 0
            _terrain = None
            if shorts_data and shorts_data.get("segments"):
                from collections import Counter

                ts = [
                    s.get("terrain_type") for s in shorts_data["segments"] if s.get("terrain_type")
                ]
                if ts:
                    _terrain = Counter(ts).most_common(1)[0][0]
            ns = ctx.num_shorts if ctx.num_shorts > 0 else 15
            sg = SocialGenerator(
                location=ctx.location,
                project_name=ctx.project_name,
                recording_date=rd_short,
                series_part=sp,
                terrain_type=_terrain,
                # Optional hand-title override at <project_dir>/social/titles.yaml.
                titles_file=output_dir / "social" / "titles.yaml",
            )
            # Each short's OWN burned label:
            # shorts_analysis.json segments carry the per-clip location_text
            # the render actually burned, so the social surfaces match the
            # pixels by construction. Falsy / drive-equal entries keep the
            # drive-wide surfaces inside generate_all (its documented
            # contract); old resume files without location_text degrade the
            # same way. Order mirrors weather_results (same segments list).
            per_short_locations = None
            if shorts_data and shorts_data.get("segments"):
                per_short_locations = [
                    s.get("location_text") or "" for s in shorts_data["segments"]
                ]
            posts = sg.generate_all(
                num_shorts=ns,
                weather_results=weather_results,
                per_short_locations=per_short_locations,
            )
            ctx.social_posts = [p.to_dict() for p in posts]
            shorts_dir.mkdir(parents=True, exist_ok=True)
            with open(shorts_dir / "social_posts.json", "w") as f:
                json.dump(ctx.social_posts, f, indent=2)
            logger.info(f"  Generated {len(ctx.social_posts)} social posts")
        except Exception as e:
            # Class name included so TitleCapacityExceeded is distinguishable.
            msg = f"Social post generation failed: {type(e).__name__}: {e}"
            logger.error(f"  {msg}")
            return StageResult(success=False, error=msg)
        return StageResult(success=True)
