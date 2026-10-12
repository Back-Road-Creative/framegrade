"""Pipeline stage wrappers — BaseStage subclasses for each pipeline stage."""

from .analytics_sync import AnalyticsSyncStage
from .blur_detection import BlurDetectionStage
from .chapters import ChapterGenerationStage
from .cleanup import CleanupStage
from .csv_gen import CSVGenerationStage
from .d_drive_copy import DDriveCopyStage
from .detect_properties import PropertyDetectionStage
from .metadata import MetadataGenerationStage
from .pre_render_review import PreRenderReviewStage
from .schedule import ScheduleGenerationStage
from .shorts import ShortsAnalysisStage
from .single_pass_encode import SinglePassEncodeStage
from .social import SocialGenerationStage
from .stop_removal import StopRemovalStage
from .thumbnails import ThumbnailGenerationStage
from .upload import DriveUploadStage
from .validation import ValidationStage
from .validation_gate import ValidationGateStage
from .wsl_copy import WSLCopyStage

__all__ = [
    "AnalyticsSyncStage",
    "BlurDetectionStage",
    "ChapterGenerationStage",
    "CleanupStage",
    "CSVGenerationStage",
    "DDriveCopyStage",
    "PropertyDetectionStage",
    "MetadataGenerationStage",
    "PreRenderReviewStage",
    "ScheduleGenerationStage",
    "ShortsAnalysisStage",
    "SinglePassEncodeStage",
    "SocialGenerationStage",
    "StopRemovalStage",
    "ThumbnailGenerationStage",
    "DriveUploadStage",
    "ValidationStage",
    "ValidationGateStage",
    "WSLCopyStage",
]

STAGE_ORDER = [
    ValidationStage(),  # 0
    AnalyticsSyncStage(),  # 1
    PropertyDetectionStage(),  # 2
    WSLCopyStage(),  # 3
    StopRemovalStage(),  # 4
    BlurDetectionStage(),  # 5  — face/plate detect on the stops-cut file, before preview gate
    ChapterGenerationStage(),  # 6  — light: GPS + duration + keep_segments
    MetadataGenerationStage(),  # 7  — light: now picks up chapters from disk
    ScheduleGenerationStage(),  # 8  — light: defaults to 15 shorts if none yet
    PreRenderReviewStage(),  # 9  — human review before irreversible encode
    SinglePassEncodeStage(),  # 10  — heavy encode after all light work
    ThumbnailGenerationStage(),  # 11
    ShortsAnalysisStage(),  # 12
    SocialGenerationStage(),  # 13
    ValidationGateStage(),  # 14  — validate BEFORE the irreversible Drive upload
    DriveUploadStage(),  # 15  — upload only after validation passes
    CSVGenerationStage(),  # 16  — CSV needs the upload's Drive links
    DDriveCopyStage(),  # 17  — copy only after validation passes
    CleanupStage(),  # 18
]
