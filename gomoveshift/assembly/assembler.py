"""Video assembler - combines analysis results into final output."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Config
from ..models import DetectedLocation, LocationType, VideoAnalysis
from ..pipeline.chapters._shared import is_valid_chapter_placename
from .encoder import Chapter, VideoEncoder


@dataclass
class AssemblyPlan:
    """Plan for assembling output videos."""

    source_path: Path
    segments_to_keep: list[tuple[float, float]]  # (start, end) in seconds
    cut_points: list[float]  # Where to split into separate files
    chapters: list[Chapter]  # Chapter markers for locations
    output_paths: list[Path]


class VideoAssembler:
    """
    Assembles final output videos based on analysis and user approvals.

    Takes analysis results and approved edits, produces clean output files.
    """

    def __init__(self, config: Config):
        self.config = config
        self.encoder = VideoEncoder(
            codec=config.output.codec,
            crf=config.output.crf,
            preset=config.output.preset,
            audio_codec=config.output.audio_codec,
            audio_bitrate=config.output.audio_bitrate,
            # Both are pinned to LONGFORM_* by OutputConfig.model_post_init and
            # had no reader at all until now — the render silently shipped
            # whatever pix_fmt/profile ffmpeg defaulted to.
            pix_fmt=config.output.pix_fmt,
            profile=config.output.profile,
        )

    def create_assembly_plan(
        self,
        analysis: VideoAnalysis,
        approved_trims: Optional[list[tuple[float, float]]] = None,
        approved_cuts: Optional[list[float]] = None,
        output_dir: Optional[Path] = None,
    ) -> AssemblyPlan:
        """
        Create a plan for video assembly.

        Args:
            analysis: Video analysis results
            approved_trims: List of (start, end) time ranges to REMOVE
                           If None, uses all suggested dead spots
            approved_cuts: List of timestamps to split video
                          If None, uses all suggested cut points
            output_dir: Directory for output files
        """
        source_path = Path(analysis.video_path)

        if output_dir is None:
            output_dir = source_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine what to trim
        if approved_trims is None:
            # Use all suggested dead spots
            trims = [(ds.start_time, ds.end_time) for ds in analysis.dead_spots]
        else:
            trims = approved_trims

        # Calculate segments to keep (inverse of trims)
        segments_to_keep = self._invert_trims(trims, analysis.duration)

        # Determine cut points
        if approved_cuts is None:
            cuts = [cp.timestamp for cp in analysis.suggested_cuts]
        else:
            cuts = approved_cuts

        cuts = sorted(cuts)

        # Generate chapter markers from locations
        chapters = self._create_chapters(analysis.locations)

        # Determine output file paths
        output_paths = self._plan_output_files(source_path, cuts, analysis.duration, output_dir)

        return AssemblyPlan(
            source_path=source_path,
            segments_to_keep=segments_to_keep,
            cut_points=cuts,
            chapters=chapters,
            output_paths=output_paths,
        )

    def _invert_trims(
        self, trims: list[tuple[float, float]], duration: float
    ) -> list[tuple[float, float]]:
        """Convert list of segments to remove into list of segments to keep."""
        if not trims:
            return [(0.0, duration)]

        # Sort trims by start time
        trims = sorted(trims, key=lambda x: x[0])

        # Merge overlapping trims
        merged = []
        for start, end in trims:
            if merged and start <= merged[-1][1]:
                # Overlapping, extend previous
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Invert to get segments to keep
        segments = []
        prev_end = 0.0

        for trim_start, trim_end in merged:
            if trim_start > prev_end:
                segments.append((prev_end, trim_start))
            prev_end = trim_end

        # Add final segment
        if prev_end < duration:
            segments.append((prev_end, duration))

        return segments

    def _create_chapters(self, locations: list[DetectedLocation]) -> list[Chapter]:
        """Create chapter markers from detected locations."""
        chapters = []

        # Start with a default chapter
        chapters.append(Chapter(start_time=0.0, title="Start"))

        for loc in locations:
            # Only create chapters for significant locations
            if loc.location_type in (
                LocationType.TOWN_ENTRY,
                LocationType.COUNTY_BOUNDARY,
                LocationType.LANDMARK,
            ):
                # OCR sign-text is unvalidated here — business names and regex
                # false-positives can masquerade as placenames. Drop anything
                # the shared chapter validator rejects before it becomes an
                # irreversible embedded MP4 chapter.
                if not is_valid_chapter_placename(loc.location_name):
                    continue

                title = loc.location_name
                if loc.location_type == LocationType.TOWN_ENTRY:
                    title = f"Entering {loc.location_name}"
                elif loc.location_type == LocationType.COUNTY_BOUNDARY:
                    # Keep a county-first sign's own form ("County Kerry"); only
                    # append " County" to a bare US name ("Kerry" -> "Kerry County").
                    if "county" in loc.location_name.lower():
                        title = loc.location_name
                    else:
                        title = f"{loc.location_name} County"

                chapters.append(Chapter(start_time=loc.timestamp, title=title))

        return chapters

    def _plan_output_files(
        self,
        source_path: Path,
        cuts: list[float],
        duration: float,
        output_dir: Path,
    ) -> list[Path]:
        """Plan output file names based on cut points."""
        stem = source_path.stem

        if not cuts:
            # Single output file
            return [output_dir / f"{stem}_cleaned.mp4"]

        # Multiple output files
        paths = []
        for i in range(len(cuts) + 1):
            paths.append(output_dir / f"{stem}_part{i + 1:02d}.mp4")

        return paths

    def execute(
        self,
        plan: AssemblyPlan,
        progress_callback: Optional[callable] = None,
    ) -> list[Path]:
        """
        Execute the assembly plan to create output files.

        Returns list of created output file paths.
        """
        created_files = []

        if not plan.cut_points:
            # Single output file
            output_path = plan.output_paths[0]

            if progress_callback:
                progress_callback("encoding", 0.0, output_path.name)

            # Adjust chapters for trims
            adjusted_chapters = self._adjust_chapters_for_trims(
                plan.chapters, plan.segments_to_keep
            )

            success = self.encoder.trim_and_encode(
                plan.source_path,
                output_path,
                plan.segments_to_keep,
                adjusted_chapters if self.config.output.embed_chapters else None,
            )

            if success:
                created_files.append(output_path)

            if progress_callback:
                progress_callback("encoding", 1.0, output_path.name)

        else:
            # Multiple output files - split at cut points
            cut_points = [0.0] + plan.cut_points + [float("inf")]

            for i, output_path in enumerate(plan.output_paths):
                segment_start = cut_points[i]
                segment_end = cut_points[i + 1]

                if progress_callback:
                    progress = i / len(plan.output_paths)
                    progress_callback("encoding", progress, output_path.name)

                # Filter segments and chapters for this output file. The keeps
                # stay ABSOLUTE: trim_and_encode seeks with them into the source
                # it is handed, and that is the full un-trimmed source. Rebasing
                # them to segment_start made every part after the first cut from
                # the opening of the drive instead of its own footage.
                segment_keeps = [
                    (max(s, segment_start), min(e, segment_end))
                    for s, e in plan.segments_to_keep
                    if s < segment_end and e > segment_start
                ]

                segment_chapters = [
                    Chapter(c.start_time - segment_start, c.title)
                    for c in plan.chapters
                    if segment_start <= c.start_time < segment_end
                ]

                # Chapters index the finished file, so they DO rebase. Shift them
                # earlier for dead-spots removed *inside* this file, in that same
                # output-relative frame the single-file adjuster expects.
                relative_keeps = [(s - segment_start, e - segment_start) for s, e in segment_keeps]
                segment_chapters = self._adjust_chapters_for_trims(segment_chapters, relative_keeps)

                if not segment_chapters:
                    segment_chapters = [Chapter(0.0, f"Part {i + 1}")]

                if segment_keeps:
                    success = self.encoder.trim_and_encode(
                        plan.source_path,
                        output_path,
                        segment_keeps,
                        segment_chapters if self.config.output.embed_chapters else None,
                    )

                    if success:
                        created_files.append(output_path)

            if progress_callback:
                progress_callback("encoding", 1.0, "Complete")

        return created_files

    def _adjust_chapters_for_trims(
        self, chapters: list[Chapter], segments_to_keep: list[tuple[float, float]]
    ) -> list[Chapter]:
        """Adjust chapter timestamps to account for trimmed segments."""
        if not segments_to_keep:
            return chapters

        adjusted = []
        _cumulative_offset = 0.0
        _segment_idx = 0

        for chapter in chapters:
            original_time = chapter.start_time

            # Calculate how much time has been trimmed before this point
            trimmed_time = 0.0
            prev_end = 0.0

            for seg_start, seg_end in segments_to_keep:
                if seg_start > original_time:
                    break
                # Accumulate every removed gap before this chapter; overwriting
                # (=) kept only the last gap, mis-placing chapters after >=2 trims.
                trimmed_time += seg_start - prev_end
                prev_end = seg_end

            # Also add time from any segments that were completely before this chapter
            new_time = original_time - trimmed_time

            # Only keep chapters that fall within kept segments
            for seg_start, seg_end in segments_to_keep:
                if seg_start <= original_time < seg_end:
                    adjusted.append(Chapter(new_time, chapter.title))
                    break

        return adjusted
