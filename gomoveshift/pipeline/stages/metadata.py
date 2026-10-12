"""Stage: YouTube metadata generation."""

import json
import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)

# Floor for the SEO structural score (title/description length, tag count,
# keyword placement — see BaseSEOGenerator._calculate_seo_structure_score).
# A healthy long-form scores >= 0.5 (description >= 300 chars gives 0.3 and the
# "Scenic Drive" title keyword gives 0.2); a below-floor value means the
# title/description/tags are structurally broken. The content-safety gate
# does not check quality_score, so this is an upstream gate that refuses to
# write publish metadata before it is used.
# The degraded-SEO path carries quality_score=None and is gated separately by
# the validator's seo_quality gate, so None is not our concern here.
_MIN_METADATA_QUALITY_SCORE = 0.5


def _quality_score_error(metadata: dict) -> "str | None":
    """Return a rejection message when the SEO structural score is below the
    floor, else ``None`` (including when the score is ``None``/absent)."""
    score = metadata.get("quality_score")
    if score is not None and score < _MIN_METADATA_QUALITY_SCORE:
        return (
            f"metadata SEO quality score {score:.2f} below floor "
            f"{_MIN_METADATA_QUALITY_SCORE:.2f} — structurally deficient "
            "title/description/tags; refusing to write publish metadata"
        )
    return None


class MetadataGenerationStage(BaseStage):
    name = PipelineContext.STAGE_METADATA_GENERATION

    def run(self, ctx):
        from .. import orchestrator as _orch

        LongFormMetadataGenerator = _orch.LongFormMetadataGenerator  # noqa: N806
        props = ctx.video_properties
        output_dir = ctx.paths.project_dir
        if ctx.recording_date is None:
            ctx.recording_date = _extract_recording_date(ctx)
        if ctx.route_title is None:
            ctx.route_title = _extract_route_title(ctx)
        publish_day_names = {
            "mon": "Monday",
            "tue": "Tuesday",
            "wed": "Wednesday",
            "thu": "Thursday",
            "fri": "Friday",
            "sat": "Saturday",
            "sun": "Sunday",
        }
        try:
            meta_gen = LongFormMetadataGenerator()
            chapters, chapter_text = [], ""
            chapters_path = output_dir / f"{ctx.project_name}_chapters.txt"
            if chapters_path.exists():
                chapter_text = chapters_path.read_text()
                for line in chapter_text.strip().splitlines():
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        chapters.append({"timestamp": parts[0], "title": parts[1]})
            recording_date_long = (
                ctx.recording_date.strftime("%B %d, %Y") if ctx.recording_date else None
            )
            metadata = meta_gen.generate(
                location=ctx.location,
                publish_day=publish_day_names[ctx.publish_day],
                video_duration=props.duration,
                project_name=ctx.project_name,
                recording_date=recording_date_long,
                title_override=ctx.route_title,
            )
            if chapters:
                metadata["chapters"] = chapters
                metadata["chapter_text"] = chapter_text
            # Enforce the SEO structural score BEFORE writing the publish
            # metadata artifact (upstream gate — cheaper than a bad upload).
            _q_err = _quality_score_error(metadata)
            if _q_err:
                return StageResult(success=False, error=_q_err)
            metadata_path = output_dir / f"{ctx.project_name}_youtube.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            txt_path = output_dir / f"{ctx.project_name}_youtube.txt"
            with open(txt_path, "w") as f:
                f.write(f"TITLE:\n{metadata['title']}\n\n")
                f.write(f"DESCRIPTION:\n{metadata['description']}\n\n")
                f.write(f"TAGS:\n{', '.join(metadata['tags'])}\n\n")
                f.write(f"CATEGORY: {metadata['category']}\n")
                if metadata.get("quality_score") is not None:
                    f.write(f"QUALITY SCORE: {round(metadata['quality_score'] * 100)}%\n")
                if chapter_text:
                    f.write(f"\nCHAPTERS:\n{chapter_text}\n")
            ctx.metadata_file = metadata_path
            ctx.metadata = metadata
            logger.info(f"  Title: {metadata['title']}")
            logger.info(f"  Tags: {len(metadata['tags'])} tags generated")
        except Exception as e:
            return StageResult(success=False, error=f"Metadata generation failed: {e}")
        return StageResult(success=True)


def _extract_recording_date(ctx):
    from ..orchestrator import PipelineOrchestrator

    rd = PipelineOrchestrator._extract_recording_date(
        gpx_date=ctx.gps_date, project_name=ctx.project_name
    )
    if rd is None:
        from ..gopro_gps import resolve_corresponding_gpx

        # No blind glob("*.gpx")[0] — see chapters stage. Adopt only an
        # explicit --gps, an exact <stem>.gpx sidecar, or a GPX that passes the
        # correspondence gate, so a wrong-drive GPX can't set the recording date.
        _dur = ctx.video_properties.duration if ctx.video_properties else 0.0
        _gpx = resolve_corresponding_gpx(
            ctx.source_video, explicit_gpx=ctx.gps_path, video_duration=_dur
        )
        if _gpx:
            try:
                from ..chapters import ChapterGenerator

                _cg = ChapterGenerator(ctx.config)
                _j = _cg.gpx_parser.parse(_gpx)
                rd = _j.start_date
            except Exception:
                pass
    return rd


def _extract_route_title(ctx):
    output_dir = ctx.paths.project_dir
    cp = output_dir / f"{ctx.project_name}_chapters.txt"
    if not cp.exists():
        return None
    chs = []
    for line in cp.read_text().strip().splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            chs.append({"timestamp": parts[0], "title": parts[1]})
    if not chs:
        return None
    try:
        from ..route_context import extract_route_context

        rc = extract_route_context(ctx.location, chs)
        if rc.gps_towns:
            return rc.make_default_title()
    except Exception:
        pass
    return None
