"""Phase B Preview: encode 90s preview clips from the middle of each episode."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from ...config import Config
from ..loudnorm_measure import (
    build_attenuate_filter,
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
)

logger = logging.getLogger(__name__)

#: Cutting 90s out of an episode is a stream copy, so this is a stuck-decoder
#: bound rather than a work estimate.
_GRADE_WINDOW_TIMEOUT_S = 120


def _window_grade(sp_processor, subprocess, ep_path, start, duration, work_dir, label):
    """Analysed colour filter for the previewed window, plus any temp LUT dir.

    The preview is the only human visual check ahead of the multi-hour Phase C
    master encode, and it used to render ``color_filter="null"`` — so a grading
    defect was invisible at the one gate cheap enough to catch it. The grade is
    analysed from a stream-copy cut of the SAME 90s the reviewer watches: no
    extra encode, and the look shown is the one THIS
    footage produces rather than one borrowed from elsewhere in the drive.

    Returns ``("null", None)`` — the old behaviour — whenever the cut or the
    analysis cannot run. A preview is a review aid and must never be the thing
    that stops a drive.
    """
    clip = work_dir / f"{label}_grade_window.mp4"
    try:
        cut = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(ep_path),
             "-t", f"{duration:.3f}", "-c", "copy", "-an", str(clip)],
            capture_output=True, timeout=_GRADE_WINDOW_TIMEOUT_S,
        )  # fmt: skip
        if cut.returncode != 0 or not clip.exists() or clip.stat().st_size == 0:
            logger.warning("  %s: grade window cut failed — preview renders ungraded", label)
            return "null", None
        color_filter, method, temp_lut = sp_processor._run_color_step(clip, False, None)
        logger.info("  %s: preview grade analysed from the previewed window (%s)", label, method)
        return color_filter, temp_lut
    except Exception as exc:  # noqa: BLE001 — a preview never fails the drive
        logger.warning("  %s: grade analysis failed (%s) — preview renders ungraded", label, exc)
        return "null", None
    finally:
        try:
            clip.unlink(missing_ok=True)
        except OSError:
            pass


class PhaseBPreview:
    """Phase B Preview: encode 90s from middle of each episode with OST."""

    name = "phase_b_preview"

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, state) -> None:
        # Late-bind subprocess from drive_orchestrator so tests can patch it
        import sys

        orch_mod = sys.modules.get("gomoveshift.pipeline.drive_orchestrator")
        subprocess = getattr(orch_mod, "subprocess", None)
        if subprocess is None:
            import subprocess

        logger.info("Phase B Preview: encoding 90s preview clips per episode")
        episode_files = state.episode_files or []
        episode_locations = state.episode_locations or []
        drive_location = state.location or ""
        output_base = Path(state.output_base)
        preview_dir = output_base / "_stages" / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        font_path: Optional[str] = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not Path(font_path).exists():
            logger.warning(
                "  OST font not found at %s -- previews will encode without OST overlay",
                font_path,
            )
            font_path = None

        # Lazy import to avoid circular dependency
        from ..single_pass import SinglePassProcessor

        sp_processor = SinglePassProcessor(self.config)

        for i, ep_file in enumerate(episode_files):
            ep_path = Path(ep_file)

            probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width:format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(ep_path),
            ]
            video_width = 3840
            duration = 300.0
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, timeout=30)
                raw = (probe_result.stdout or b"").decode("utf-8", errors="replace")
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if lines:
                    try:
                        video_width = int(lines[0].split(",")[0])
                    except (ValueError, IndexError):
                        pass
                    try:
                        duration = float(lines[-1])
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass

            mid = duration / 2.0
            start = max(0.0, mid - 45.0)
            clip_duration = 90.0

            location = ""
            if i < len(episode_locations) and episode_locations[i]:
                location = episode_locations[i]
            elif drive_location:
                location = drive_location

            out_path = preview_dir / f"ep_{i:03d}_preview.mp4"

            # Representative audio: every shipped surface is attenuated to
            # near-silence (the operator near-silent rule). A preview that
            # played the RAW loud source would MISLEAD the reviewer about what
            # actually ships. Measure this episode's loudness and apply the
            # SAME attenuate-only filter the day-master path uses, so the
            # preview is audibly representative (near-silent). NOTE: cutting
            # from the planned LF/short windows is still NOT reproduced here —
            # that depends on the phase_bc day-master, which --preview mode
            # intentionally does not build. The colour grade IS reproduced, from
            # the previewed window itself (see ``_window_grade``).
            measured: dict = {}
            try:
                m = subprocess.run(
                    build_loudnorm_measure_cmd(ep_path), capture_output=True, timeout=300
                )
                stderr = m.stderr or b""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                if isinstance(stderr, str):
                    measured = parse_loudnorm_stats(stderr) or {}
            except Exception:
                measured = {}
            attenuate_af = build_attenuate_filter(measured)

            color_filter, temp_lut = _window_grade(
                sp_processor, subprocess, ep_path, start, clip_duration, preview_dir, f"Episode {i}"
            )
            vf_args: list[str] = []
            # Only build a chain when there is something to render: a grade, an
            # OST overlay, or both. With neither, the preview stays a bare
            # playback-verification encode exactly as before — the filter
            # builder always emits at least a range conversion, so calling it
            # unconditionally would put a -vf on every preview.
            graded = bool(color_filter) and color_filter != "null"
            if graded or (font_path and location):
                vfilter = sp_processor._build_video_filter(
                    color_filter=color_filter,
                    location=location if font_path else "",
                    font_path=font_path,
                    duration=clip_duration,
                    fade_video=False,
                    video_width=video_width,
                )
                if vfilter and vfilter != "null":
                    vf_args = ["-vf", vfilter]

            encode_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(ep_path),
                "-t",
                "90",
                *vf_args,
                "-af",
                attenuate_af,
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p4",
                "-cq",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_path),
            ]
            try:
                result = subprocess.run(encode_cmd, capture_output=True, timeout=300)
                if result.returncode == 0:
                    logger.info(
                        "  Episode %d preview (OST: %r): %s",
                        i,
                        location or "<none>",
                        out_path,
                    )
                else:
                    logger.warning(
                        "  Episode %d preview encode failed (rc=%d)",
                        i,
                        result.returncode,
                    )
            except subprocess.TimeoutExpired:
                logger.warning("  Episode %d preview encode timed out", i)
            except Exception as e:
                logger.warning("  Episode %d preview encode error: %s", i, e)
            finally:
                # The adaptive grade bakes per-span LUTs into a temp dir the
                # caller owns; N episodes would otherwise leave N of them behind.
                if temp_lut:
                    shutil.rmtree(temp_lut, ignore_errors=True)
