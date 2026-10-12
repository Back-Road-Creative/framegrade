"""GoMoveShift Pipeline — automated video processing and publishing.

Phase 1: Long Form Processing
  - Stop detection & removal with crossfade transitions
  - Color correction (LUT or auto-adjust)
  - On-screen text overlays (location + subscribe CTA)
  - Audio attenuated to a near-silent ceiling (attenuate-only; never boosted)
  - YouTube metadata generation (SEO-optimized)

Phase 2: Shorts Extraction, Upload & Scheduling
  - High-motion segment extraction from processed long form
  - Weather detection per segment (brightness-based)
  - Vertical shorts rendering (1080x1920, center crop, OST, audio norm)
  - Resume-capable Google Drive upload with manifest
  - Schedule generation (shorts posting windows from publish calendar)
  - Unique social post generation per short
  - Metricool CSV generation with golden reference validation
"""

from .cleanup import PipelineCleanup
from .csv_generator import CSVGenerator, validate_csv
from .ffmpeg_runner import FFmpegResult, VideoValidation, validate_video
from .orchestrator import PipelineOrchestrator, RunOptions

# Phase 3: Daily Photo Posts
from .photos import (
    CaptionBatch,
    CaptionGenerator,
    PhotoCleanup,
    PhotoCleanupPlan,
    PhotoCSVGenerator,
    PhotoOrchestrator,
    PhotoPost,
    PhotoResizer,
    PhotoScheduler,
    PhotoSelection,
    PhotoSelector,
    VoiceStyle,
)
from .scheduler import PipelineScheduler, ReleaseSchedule, ScheduledSlot

# Phase 2 modules
from .shorts_extractor import PipelineShortsExtractor, RenderedShort, ShortsExtractionResult
from .single_pass import SinglePassProcessor, SinglePassResult
from .social_generator import SocialGenerator, SocialPost
from .uploader import PipelineUploader, UploadManifest, VerificationResult
from .weather import WeatherCondition, WeatherDetector, WeatherResult

__all__ = [
    # Phase 1
    "PipelineOrchestrator",
    "RunOptions",
    "PipelineCleanup",
    "SinglePassProcessor",
    "SinglePassResult",
    "FFmpegResult",
    "VideoValidation",
    "validate_video",
    # Phase 2
    "PipelineShortsExtractor",
    "RenderedShort",
    "ShortsExtractionResult",
    "WeatherDetector",
    "WeatherResult",
    "WeatherCondition",
    "PipelineUploader",
    "UploadManifest",
    "PipelineScheduler",
    "ReleaseSchedule",
    "ScheduledSlot",
    "SocialGenerator",
    "SocialPost",
    "CSVGenerator",
    "validate_csv",
    # Phase 3: Daily Photo Posts
    "PhotoOrchestrator",
    "PhotoSelector",
    "PhotoSelection",
    "PhotoScheduler",
    "PhotoCSVGenerator",
    "PhotoPost",
    "PhotoResizer",
    "PhotoCleanup",
    "PhotoCleanupPlan",
    "CaptionGenerator",
    "CaptionBatch",
    "VerificationResult",
    "VoiceStyle",
]
