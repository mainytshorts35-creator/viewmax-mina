"""
================================================================================
VIEWMAX STUDIO PRO — CONFIGURATION & ENVIRONMENT MANAGEMENT
FILE: config.py
DESCRIPTION: Manages global configuration settings, environment validators,
             dataclasses for audio DSP, speech providers, typography, and logging.
================================================================================
"""

import os
import sys
import logging
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


# ==============================================================================
# 1. ENUMS & CONSTANTS
# ==============================================================================
class AspectRatioEnum(Enum):
    SHORTS_9_16 = "9:16 (YouTube Shorts / TikTok)"
    LANDSCAPE_16_9 = "16:9 (YouTube Landscape)"
    SQUARE_1_1 = "1:1 (Instagram Feed)"


class CaptionStyleEnum(Enum):
    MR_BEAST_POP = "MrBeast Dynamic Pop (Yellow/White bold punch)"
    HORMOZI_GRADIENT = "Alex Hormozi Impact (High contrast neon highlight)"
    CINEMATIC_SUBTLE = "Cinematic Minimalist (Clean lower-third)"


class VoiceProviderEnum(Enum):
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"


# ==============================================================================
# 2. CONFIGURATION DATACLASSES
# ==============================================================================
@dataclass
class PathConfig:
    workspace_dir: str = "viewmax_workspace"
    temp_dir: str = os.path.join("viewmax_workspace", "temp")
    exports_dir: str = os.path.join("viewmax_workspace", "exports")
    assets_dir: str = os.path.join("viewmax_workspace", "assets")

    def __post_init__(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)


@dataclass
class CredentialConfig:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")


@dataclass
class AudioDSPConfig:
    target_lufs: float = -14.0
    compressor_threshold: float = -16.0
    compressor_ratio: float = 3.0
    eq_high_boost_db: float = 3.0
    bgm_ducking_db: float = -18.0


@dataclass
class VoiceSettings:
    default_provider: VoiceProviderEnum = VoiceProviderEnum.ELEVENLABS
    default_voice_id: str = "pNInz6obpgDQGcFmaJgB"
    stability: float = 0.75
    similarity_boost: float = 0.85


@dataclass
class SafetyConfig:
    strict_mode: bool = True
    excluded_keywords: List[str] = field(default_factory=lambda: ["ufc", "fight", "combat", "wholesome moment", "crying"])


@dataclass
class GlobalConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    credentials: CredentialConfig = field(default_factory=CredentialConfig)
    audio: AudioDSPConfig = field(default_factory=AudioDSPConfig)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def update_api_keys(self, openai_key: str, elevenlabs_key: str):
        if openai_key:
            self.credentials.openai_api_key = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
        if elevenlabs_key:
            self.credentials.elevenlabs_api_key = elevenlabs_key
            os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key


# Global configuration instance
global_config = GlobalConfig()


# ==============================================================================
# 3. TYPOGRAPHY & CANVAS PRESETS
# ==============================================================================
@dataclass
class CanvasDimension:
    width: int
    height: int
    name: str


@dataclass
class TypographyStyle:
    font_name: str
    font_size: int
    primary_color: str  # Hex or ASS color format
    outline_color: str
    highlight_color: str
    word_chunk_size: int
    shadow_depth: int


CANVAS_PRESETS: Dict[str, CanvasDimension] = {
    AspectRatioEnum.SHORTS_9_16.value: CanvasDimension(1080, 1920, "9:16 Vertical"),
    AspectRatioEnum.LANDSCAPE_16_9.value: CanvasDimension(1920, 1080, "16:9 Landscape"),
    AspectRatioEnum.SQUARE_1_1.value: CanvasDimension(1080, 1080, "1:1 Square")
}

TYPOGRAPHY_PRESETS: Dict[str, TypographyStyle] = {
    CaptionStyleEnum.MR_BEAST_POP.value: TypographyStyle(
        font_name="Impact",
        font_size=82,
        primary_color="&H00FFFFFF&",  # White
        outline_color="&H00000000&",  # Black outline
        highlight_color="&H0000FFFF&", # Yellow
        word_chunk_size=2,
        shadow_depth=4
    ),
    CaptionStyleEnum.HORMOZI_GRADIENT.value: TypographyStyle(
        font_name="Arial Black",
        font_size=76,
        primary_color="&H00FFFFFF&",
        outline_color="&H00000000&",
        highlight_color="&H0000CCFF&", # Orange/Yellow highlight
        word_chunk_size=3,
        shadow_depth=3
    ),
    CaptionStyleEnum.CINEMATIC_SUBTLE.value: TypographyStyle(
        font_name="Trebuchet MS",
        font_size=56,
        primary_color="&H00EFEFEF&",
        outline_color="&H00222222&",
        highlight_color="&H00FFFFFF&",
        word_chunk_size=4,
        shadow_depth=2
    )
}


# ==============================================================================
# 4. ENVIRONMENT & SYSTEM DIAGNOSTICS
# ==============================================================================
class EnvironmentValidator:
    """Validates system binaries (FFmpeg) and workspace folders."""

    @staticmethod
    def check_ffmpeg() -> tuple[bool, Optional[str]]:
        """Verifies if ffmpeg is available in system PATH."""
        try:
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return True, "ffmpeg"
        except FileNotFoundError:
            pass

        # Check common paths or imageio_ffmpeg if installed
        try:
            import imageio_ffmpeg
            bin_path = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.exists(bin_path):
                return True, bin_path
        except Exception:
            pass

        return False, None

    @classmethod
    def run_full_diagnostics(cls, paths: PathConfig) -> Dict[str, Any]:
        """Runs complete system check for deployment readiness."""
        ffmpeg_ok, ffmpeg_path = cls.check_ffmpeg()
        
        report = {
            "ffmpeg_available": ffmpeg_ok,
            "ffmpeg_path": ffmpeg_path,
            "workspace_dirs_ready": os.path.exists(paths.temp_dir) and os.path.exists(paths.exports_dir)
        }
        return report


def get_logger(name: str) -> logging.Logger:
    """Returns a consistently configured logger for the application suite."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | [%(levelname)s] | %(name)s %(funcName)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    logger = get_logger("ViewMaxPro.Config")
    logger.info("Running standalone Configuration diagnostics...")
    diag = EnvironmentValidator.run_full_diagnostics(global_config.paths)
    print(f"Diagnostics Report: {diag}")
    print("Configuration module fully operational.")
