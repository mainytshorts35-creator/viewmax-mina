import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logger(name: str = "ViewMaxPro") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s standard_runner: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# ==============================================================================
# DATACLASS CONFIGURATIONS
# ==============================================================================
@dataclass
class CanvasPreset:
    width: int
    height: int
    aspect_ratio_str: str

CANVAS_PRESETS: Dict[str, CanvasPreset] = {
    "9:16 Vertical (Shorts/Reels/TikTok)": CanvasPreset(1080, 1920, "9:16"),
    "1:1 Square (Instagram Post)": CanvasPreset(1080, 1080, "1:1"),
    "16:9 Widescreen (YouTube Standard)": CanvasPreset(1920, 1080, "16:9")
}

@dataclass
class TypographyStyle:
    font_family: str
    font_size_pct: float
    active_color: Tuple[int, int, int]
    inactive_color: Tuple[int, int, int]
    stroke_color: Tuple[int, int, int]
    stroke_width: int
    glow_enabled: bool
    glow_color: Tuple[int, int, int]
    y_position_pct: float

TYPOGRAPHY_PRESETS: Dict[str, TypographyStyle] = {
    "Hormozi Active Gold": TypographyStyle(
        font_family="Impact.ttf",
        font_size_pct=0.055,
        active_color=(255, 215, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=5,
        glow_enabled=False,
        glow_color=(0, 0, 0),
        y_position_pct=0.72
    ),
    "Cyberpunk Electric Neon": TypographyStyle(
        font_family="Impact.ttf",
        font_size_pct=0.052,
        active_color=(0, 242, 254),
        inactive_color=(200, 200, 200),
        stroke_color=(10, 10, 25),
        stroke_width=6,
        glow_enabled=True,
        glow_color=(0, 242, 254),
        y_position_pct=0.75
    ),
    "Impact Red Banner": TypographyStyle(
        font_family="Impact.ttf",
        font_size_pct=0.058,
        active_color=(255, 45, 85),
        inactive_color=(240, 240, 240),
        stroke_color=(0, 0, 0),
        stroke_width=6,
        glow_enabled=False,
        glow_color=(0, 0, 0),
        y_position_pct=0.70
    )
}

@dataclass
class AudioProcessingSettings:
    sample_rate: int = 44100
    channels: int = 1
    highpass_cutoff: float = 80.0
    presence_boost_freq: float = 3200.0
    presence_boost_gain: float = 1.4
    compressor_threshold_db: float = -16.0
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 4.0
    compressor_release_ms: float = 60.0
    compressor_makeup_db: float = 3.5
    limiter_ceiling_db: float = -0.5
    bgm_ducking_attenuation_db: float = -14.0

@dataclass
class SystemPaths:
    root_dir: str = os.getcwd()
    workspace_dir: str = os.path.join(os.getcwd(), "viewmax_workspace")
    cache_dir: str = os.path.join(os.getcwd(), "viewmax_workspace", "cache")
    temp_dir: str = os.path.join(os.getcwd(), "viewmax_workspace", "temp")
    exports_dir: str = os.path.join(os.getcwd(), "viewmax_workspace", "exports")
    font_path: str = os.path.join(os.getcwd(), "Impact.ttf")

    def initialize_directories(self) -> None:
        for p in [self.workspace_dir, self.cache_dir, self.temp_dir, self.exports_dir]:
            os.makedirs(p, exist_ok=True)
            logger.info(f"Directory verified: {p}")

@dataclass
class AppSettings:
    paths: SystemPaths = field(default_factory=SystemPaths)
    audio: AudioProcessingSettings = field(default_factory=AudioProcessingSettings)
    default_canvas: str = "9:16 Vertical (Shorts/Reels/TikTok)"
    default_style: str = "Hormozi Active Gold"
    whisper_model_size: str = "base"
    compute_type: str = "int8"
    excluded_keywords: List[str] = field(default_factory=lambda: ["ufc", "mma", "wholesome compilation", "sentimental moments"])

    def validate_environment(self) -> Dict[str, bool]:
        status = {
            "ffmpeg_available": False,
            "impact_font_present": os.path.exists(self.paths.font_path)
        }
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and os.path.exists(ffmpeg_exe):
                status["ffmpeg_available"] = True
        except Exception as e:
            logger.warning(f"FFmpeg check failed: {e}")
        return status
