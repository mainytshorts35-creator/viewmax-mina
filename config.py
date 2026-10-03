"""
================================================================================
VIEWMAX STUDIO PRO — CONFIGURATION & ENVIRONMENT MANAGEMENT
FILE: config.py
DESCRIPTION: Central configuration, Pydantic/dataclass models, typography presets, 
             canvas aspect ratios, and system path initializers.
================================================================================
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple

# ==============================================================================
# 1. ENUMS & CONSTANTS
# ==============================================================================
class AspectRatioEnum(Enum):
    SHORTS_9_16 = "9:16 (YouTube Shorts / TikTok - 1080x1920)"
    LANDSCAPE_16_9 = "16:9 (YouTube Standard - 1920x1080)"
    SQUARE_1_1 = "1:1 (Instagram Feed - 1080x1080)"


class CaptionStyleEnum(Enum):
    HORMOZI_POP = "Hormozi Pop (High-Energy Yellow/White Accent)"
    CINEMATIC_GLOW = "Cinematic Glow (Clean White with Soft Shadow)"
    CYBERPUNK_NEON = "Cyberpunk Neon (Electric Cyan & Magenta)"


class VoiceProviderEnum(Enum):
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"


@dataclass
class CanvasDimension:
    width: int
    height: int
    name: str


CANVAS_PRESETS: Dict[str, CanvasDimension] = {
    AspectRatioEnum.SHORTS_9_16.value: CanvasDimension(1080, 1920, "9:16 Vertical"),
    AspectRatioEnum.LANDSCAPE_16_9.value: CanvasDimension(1920, 1080, "16:9 Landscape"),
    AspectRatioEnum.SQUARE_1_1.value: CanvasDimension(1080, 1080, "1:1 Square")
}


@dataclass
class TypographyStyle:
    name: str
    font_name: str
    font_size: int
    primary_color: str
    highlight_color: str
    stroke_color: str
    stroke_width: int
    word_chunk_size: int
    position_y: str


TYPOGRAPHY_PRESETS: Dict[str, TypographyStyle] = {
    CaptionStyleEnum.HORMOZI_POP.value: TypographyStyle(
        name="Hormozi Pop",
        font_name="Arial-Black",
        font_size=72,
        primary_color="&H00FFFFFF",
        highlight_color="&H0000FFFF",
        stroke_color="&H00000000",
        stroke_width=4,
        word_chunk_size=2,
        position_y="center"
    ),
    CaptionStyleEnum.CINEMATIC_GLOW.value: TypographyStyle(
        name="Cinematic Glow",
        font_name="Trebuchet-MS",
        font_size=64,
        primary_color="&H00FFFFFF",
        highlight_color="&H0080E5FF",
        stroke_color="&H00222222",
        stroke_width=3,
        word_chunk_size=3,
        position_y="bottom"
    ),
    CaptionStyleEnum.CYBERPUNK_NEON.value: TypographyStyle(
        name="Cyberpunk Neon",
        font_name="Impact",
        font_size=68,
        primary_color="&H00FFFF00",
        highlight_color="&H00FF00FF",
        stroke_color="&H00000000",
        stroke_width=4,
        word_chunk_size=2,
        position_y="center"
    )
}


# ==============================================================================
# 2. CONFIGURATION DATA MODELS
# ==============================================================================
@dataclass
class PathConfig:
    base_dir: str = "/mount/src/viewmax-mina" if os.path.exists("/mount/src/viewmax-mina") else os.getcwd()
    workspace_dir: str = field(init=False)
    temp_dir: str = field(init=False)
    exports_dir: str = field(init=False)
    assets_dir: str = field(init=False)

    def __post_init__(self):
        self.workspace_dir = os.path.join(self.base_dir, "viewmax_workspace")
        self.temp_dir = os.path.join(self.workspace_dir, "temp")
        self.exports_dir = os.path.join(self.workspace_dir, "exports")
        self.assets_dir = os.path.join(self.workspace_dir, "assets")

        for d in [self.workspace_dir, self.temp_dir, self.exports_dir, self.assets_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class CredentialsConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")


@dataclass
class AudioProcessingConfig:
    target_lufs: float = -14.0
    compressor_threshold: float = -18.0
    compressor_ratio: float = 3.0
    eq_high_boost_db: float = 3.0
    ducking_threshold_db: float = -22.0
    ducking_ratio: float = 4.0


@dataclass
class WhisperModelConfig:
    model_size: str = "base"
    compute_type: str = "int8"
    device: str = "cpu"


@dataclass
class SafetyGuardrails:
    strict_mode: bool = True
    excluded_keywords: list = field(default_factory=lambda: ["ufc", "fight", "combat", "violence", "blood"])


@dataclass
class GlobalConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    audio: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)
    whisper: WhisperModelConfig = field(default_factory=WhisperModelConfig)
    safety: SafetyGuardrails = field(default_factory=SafetyGuardrails)

    def update_api_keys(self, openai_key: str, elevenlabs_key: str):
        if openai_key:
            self.credentials.openai_api_key = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
        if elevenlabs_key:
            self.credentials.elevenlabs_api_key = elevenlabs_key
            os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key


# Global application configuration instance (Required by audio_dsp, voice_synthesis, and app)
global_config = GlobalConfig()


# ==============================================================================
# 3. ENVIRONMENT & DEPENDENCY VALIDATOR
# ==============================================================================
class EnvironmentValidator:
    """Validates system binaries (FFmpeg) and python environment constraints."""

    @staticmethod
    def check_ffmpeg() -> Tuple[bool, Optional[str]]:
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return True, ffmpeg_path
        
        for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]:
            if shutil.which(candidate):
                return True, candidate

        return False, None

    @staticmethod
    def run_full_diagnostics(paths: PathConfig) -> Dict[str, Any]:
        has_ffmpeg, ffmpeg_bin = EnvironmentValidator.check_ffmpeg()
        return {
            "ffmpeg_installed": has_ffmpeg,
            "ffmpeg_path": ffmpeg_bin or "Not Found",
            "workspace_ready": os.path.exists(paths.workspace_dir),
            "temp_dir_ready": os.path.exists(paths.temp_dir)
        }


# ==============================================================================
# 4. LOGGER SETUP
# ==============================================================================
def get_logger(name: str) -> logging.Logger:
    """Creates and returns a standardized module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | [%(levelname)s] | %(name)s %(funcName)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
