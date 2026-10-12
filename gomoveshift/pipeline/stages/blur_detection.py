"""Stage: detect faces ∪ license plates for privacy blur (blur chain).

Runs on the EXACT stops-cut file the stage-9 encode consumes (inserted after
StopRemoval, before ChapterGeneration and the PreRenderReview gate, so an
operator can preview). Produces a stabilized blur-region sidecar; the next blur stages turn it
into a mask and composite it in the encode. Off by default (``blur.enabled``);
escape ``--skip-blur``. A detection failure is FATAL when blur is on — never
ship to the irreversible encode unable to locate faces/plates.
"""

import logging
import math

from .. import blur_gate
from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class BlurDetectionStage(BaseStage):
    name = PipelineContext.STAGE_BLUR_DETECTION

    def should_run(self, ctx) -> bool:
        if not super().should_run(ctx):
            return False
        if not getattr(ctx.config.drive, "blur_enabled", False):
            return False
        if getattr(ctx.config.drive, "skip_blur", False):
            logger.warning(
                "  BLUR: skipped (--skip-blur) — footage may ship with unblurred faces/plates"
            )
            return False
        return True

    def run(self, ctx):
        try:
            from redactcam import detect_and_track
            from redactcam import timeline as br

            from ...analysis import (
                resolve_face_model,
                resolve_person_model,
                resolve_plate_model,
                resolve_vehicle_model,
            )
            from ..stop_processor import compute_file_hash

            drive = ctx.config.drive
            video = ctx.current_video
            sidecar = ctx.paths.stages_dir / f"{ctx.project_name}_blur_regions.json"
            input_hash = compute_file_hash(video)

            existing = br.load_sidecar(sidecar)
            if existing is not None and existing.source_hash == input_hash:
                logger.info("  BLUR: reusing detections from %s (input hash match)", sidecar.name)
                ctx.blur_regions_path = sidecar
                self._ensure_mask(ctx, existing, allow_reuse=True)
                return StageResult(success=True)

            fps, frame_count, w, h = br.probe_video(video)
            props = ctx.video_properties
            w = w or (props.width if props else 0)
            h = h or (props.height if props else 0)
            fps = fps or (props.fps if props else 0.0)

            # Occupant cabin blur is a signed-off privacy feature: if it's enabled but
            # the vehicle model can't resolve, drivers would ship UNBLURRED — hard-fail
            # rather than silently disable. Disable deliberately with cabin_frac=0.
            person_on = bool(getattr(drive, "blur_person_enabled", False))
            person_model = resolve_person_model(drive) if person_on else None
            vehicle_model = resolve_vehicle_model(drive)
            if drive.blur_vehicle_cabin_frac > 0 and vehicle_model is None:
                return StageResult(
                    success=False,
                    error=(
                        "BLUR: occupant cabin blur is on (blur_vehicle_cabin_frac>0) but the "
                        "vehicle model could not be resolved — drivers would ship unblurred. "
                        "Set vehicle_model_url/path, pre-cache the model, or disable with "
                        "blur_vehicle_cabin_frac=0."
                    ),
                )

            frame_boxes, decoded_count, n_faces, n_plates = detect_and_track(
                video,
                sample_fps=drive.blur_sample_fps,
                conf=drive.blur_conf,
                face_conf=drive.blur_face_conf,
                face_sample_fps=drive.blur_face_sample_fps,
                face_confirm_min_detections=drive.blur_face_confirm_min_detections,
                pre_roll_frames=drive.blur_pre_roll_frames,
                plate_model_path=resolve_plate_model(drive),
                face_model_path=resolve_face_model(drive),
                tile_px=drive.blur_tile_px,
                tile_overlap=drive.blur_tile_overlap,
                plate_min_aspect=drive.blur_plate_min_aspect,
                plate_max_aspect=drive.blur_plate_max_aspect,
                plate_max_area_frac=drive.blur_plate_max_area_frac,
                plate_boost_speed_px=drive.blur_plate_boost_speed_px,
                plate_boost_max_run=drive.blur_plate_boost_max_run,
                track_face_margin_frac=drive.blur_face_margin_frac,
                track_face_motion_gain=drive.blur_face_motion_gain,
                track_cabin_frac=drive.blur_vehicle_cabin_frac,
                vehicle_model_path=vehicle_model,
                vehicle_conf=drive.blur_vehicle_conf,
                vehicle_min_height_frac=drive.blur_vehicle_min_height_frac,
                vehicle_ignore_bottom_frac=drive.blur_vehicle_ignore_bottom_frac,
                track_person=person_on and person_model is not None,
                person_model_path=person_model,
                person_conf=getattr(drive, "blur_person_conf", 0.35),
                person_min_height_frac=getattr(drive, "blur_person_min_height_frac", 0.0),
            )
            # detect_and_track leaves frames between samples with no entry at all,
            # and the renderer reads "no entry" as "nothing to blur". Fill those
            # gaps before the timeline is built — see fill_detection_gaps.
            frame_boxes = blur_gate.fill_detection_gaps(
                frame_boxes,
                max_gap_frames=math.ceil(fps / max(drive.blur_sample_fps, 1e-6)) + 1,
            )
            # A track's first detected frame is not its first VISIBLE frame; the
            # frames just before it are what ship unmasked. Runs after the gap
            # fill, so only births that survive interpolation are extended.
            frame_boxes = blur_gate.backfill_track_births(
                frame_boxes, pre_roll_frames=drive.blur_pre_roll_frames
            )
            # The decode loop counted every frame — authoritative over probe
            # metadata (CAP_PROP_FRAME_COUNT is often a few frames off).
            frame_count = decoded_count or frame_count
            logger.info(
                "  BLUR: %d face + %d plate detection(s); flow-tracked %d frame(s)",
                n_faces,
                n_plates,
                sum(1 for boxes in frame_boxes.values() if boxes),
            )

            timeline = br.build_timeline(
                frame_boxes,
                source_hash=input_hash,
                source_fps=fps,
                source_frame_count=frame_count,
                frame_width=w,
                frame_height=h,
                box_dilation_px=drive.blur_box_dilation_px,
                sample_fps=drive.blur_sample_fps,
                conf=drive.blur_conf,
            )
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            br.write_sidecar(sidecar, timeline)
            ctx.blur_regions_path = sidecar
            self._ensure_mask(ctx, timeline, allow_reuse=False)
        except Exception as e:
            return StageResult(success=False, error=f"Blur detection failed: {e}")
        return StageResult(success=True)

    def _ensure_mask(self, ctx, timeline, *, allow_reuse: bool) -> None:
        """Materialize the frame-accurate blur mask video for the encode.

        ``allow_reuse`` is True only on the resume path (the input hash matched,
        so an existing mask is still valid); a fresh detection always re-renders.
        """
        from redactcam import materialize_mask_video

        mask_path = ctx.paths.stages_dir / f"{ctx.project_name}_blur_mask.mkv"
        if allow_reuse and mask_path.exists():
            logger.info("  BLUR: reusing mask %s", mask_path.name)
        else:
            materialize_mask_video(timeline, mask_path)
        ctx.blur_mask_path = mask_path
