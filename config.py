"""
================================================================================
VIEWMAX STUDIO PRO — GLOBAL CONFIGURATION & SYSTEM ARCHITECTURE
FILE: config.py
DESCRIPTION: Centralized configuration management, environment validation,
             dataclass definitions, typography presets, audio DSP parameters,
             and runtime workspace setup.
================================================================================
"""

import os
import sys
import json
import logging
import platform
import shutil
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any


# ==============================================================================
# 1. ENTERPRISE LOGGING SYSTEM
# ==============================================================================
class ViewMaxLogFormatter(logging.Formatter):
    """Custom log formatter providing colorized log outputs for terminal debugging."""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    log_fmt = "%(asctime)s | [%(levelname)s] | %(name)s standard_runner | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + log_fmt + reset,
        logging.INFO: green + log_fmt + reset,
        logging.WARNING: yellow + log_fmt + reset,
        logging.ERROR: red + log_fmt + reset,
        logging.CRITICAL: bold_red + log_fmt + reset
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.log_fmt)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str = "ViewMaxPro") -> logging.Logger:
    """Instantiates or retrieves a structured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ViewMaxLogFormatter())
        logger.addHandler(console_handler)
    return logger


logger = get_logger("ViewMaxPro.Config")


# ==============================================================================
# 2. ENUMERATIONS & CORE CONSTANTS
# ==============================================================================
class AspectRatioEnum(str, Enum):
    VERTICAL_9_16 = "9:16 Vertical (Shorts/Reels/TikTok)"
    SQUARE_1_1 = "1:1 Square (Instagram Post)"
    WIDESCREEN_16_9 = "16:9 Widescreen (YouTube)"
    CINEMATIC_21_9 = "21:9 Ultrawide"


class VoiceProviderEnum(str, Enum):
    ELEVENLABS = "ElevenLabs"
    OPENAI_HD = "OpenAI_HD"


class CaptionStyleEnum(str, Enum):
    HORMOZI_GOLD = "Hormozi Active Gold"
    CYBERPUNK_NEON = "Cyberpunk Electric Neon"
    IMPACT_RED = "Impact Red Banner"
    MINIMAL_WHITE = "Minimalist Clean White"
    SYNTHWAVE_PINK = "Synthwave Neon Pink"


class TrimStrategyEnum(str, Enum):
    SMART_CLIMAX = "Smart Climax Retention"
    KEEP_FULL_FIT_SPEECH = "Keep Full Video (Fit Speech)"
    CENTER_CROP_PAD = "Center Crop with Pad"


# ==============================================================================
# 3. DATACLASS CONFIGURATION SCHEMA
# ==============================================================================
@dataclass
class SystemPaths:
    """Manages absolute and relative directory structure for file isolation."""
    root_dir: str = field(default_factory=lambda: os.getcwd())
    workspace_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "viewmax_workspace"))
    cache_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "viewmax_workspace", "cache"))
    temp_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "viewmax_workspace", "temp"))
    exports_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "viewmax_workspace", "exports"))
    bgm_library_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "viewmax_workspace", "bgm"))
    fonts_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "viewmax_workspace", "fonts"))
    font_path: str = field(default_factory=lambda: os.path.join(os.getcwd(), "Impact.ttf"))

    def initialize_directories(self) -> None:
        """Creates required system directories if they do not exist."""
        for path in [
            self.workspace_dir,
            self.cache_dir,
            self.temp_dir,
            self.exports_dir,
            self.bgm_library_dir,
            self.fonts_dir
        ]:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory verified: {path}")


@dataclass
class CanvasDimension:
    """Defines render resolution, target aspect ratio, and background blur rules."""
    width: int
    height: int
    aspect_str: str
    bg_blur_radius: int = 35
    fg_shadow_radius: int = 15


@dataclass
class TypographyStyle:
    """Design parameters for dynamic kinetic typography overlays."""
    name: str
    font_family: str
    font_size_pct: float
    active_color: Tuple[int, int, int]
    inactive_color: Tuple[int, int, int]
    stroke_color: Tuple[int, int, int]
    stroke_width: int
    glow_enabled: bool
    glow_color: Tuple[int, int, int]
    glow_radius: int
    y_position_pct: float
    word_chunk_size: int = 3
    spacing_pct: float = 0.02
    active_y_lift_pct: float = 0.008


@dataclass
class AudioProcessingSettings:
    """Parameters for broadcast-grade multi-band vocal DSP mastering."""
    sample_rate: int = 44100
    channels: int = 1
    highpass_cutoff_hz: float = 80.0
    lowpass_cutoff_hz: float = 16000.0
    presence_boost_freq_hz: float = 3200.0
    presence_boost_q: float = 1.5
    presence_boost_gain: float = 1.4
    compressor_threshold_db: float = -16.0
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 4.0
    compressor_release_ms: float = 60.0
    compressor_makeup_db: float = 3.5
    limiter_ceiling_db: float = -0.5
    bgm_ducking_attenuation_db: float = -14.0
    bgm_base_volume_scale: float = 0.25
    fade_in_duration_sec: float = 0.15
    fade_out_duration_sec: float = 0.30


@dataclass
class ElevenLabsSettings:
    """API configurations for ElevenLabs Multilingual v2 speech synthesis."""
    api_key: str = ""
    default_model_id: str = "eleven_multilingual_v2"
    stability: float = 0.40
    similarity_boost: float = 0.80
    style: float = 0.25
    use_speaker_boost: bool = True
    request_timeout_sec: int = 30


@dataclass
class OpenAISettings:
    """API configurations for OpenAI GPT-4o brain and TTS-1-HD vocal engines."""
    api_key: str = ""
    llm_model: str = "gpt-4o"
    tts_model: str = "tts-1-hd"
    temperature: float = 0.70
    max_tokens: int = 1500
    timeout_sec: int = 45


@dataclass
class WhisperSettings:
    """Local alignment engine parameters via Faster-Whisper."""
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    word_timestamps: bool = True
    language: str = "en"


@dataclass
class ContentSafetyRules:
    """Strict filtering rules to enforce branding scope and exclude unwanted topics."""
    strict_mode: bool = True
    excluded_keywords: List[str] = field(default_factory=lambda: [
        "ufc", "mma", "fighting championship", "mixed martial arts",
        "wholesome compilation", "wholesome moment", "sentimental moments",
        "cute animals compilation", "try not to cry"
    ])
    negative_prompt_directives: str = (
        "DO NOT generate content related to sports fighting, UFC, MMA, "
        "or overly sentimental wholesome compilations. Keep narrative intense, "
        "mysterious, high-stakes, or analytical."
    )


# ==============================================================================
# 4. PRESET LIBRARIES
# ==============================================================================
CANVAS_PRESETS: Dict[str, CanvasDimension] = {
    AspectRatioEnum.VERTICAL_9_16.value: CanvasDimension(
        width=1080, height=1920, aspect_str="9:16", bg_blur_radius=35
    ),
    AspectRatioEnum.SQUARE_1_1.value: CanvasDimension(
        width=1080, height=1080, aspect_str="1:1", bg_blur_radius=25
    ),
    AspectRatioEnum.WIDESCREEN_16_9.value: CanvasDimension(
        width=1920, height=1080, aspect_str="16:9", bg_blur_radius=20
    ),
    AspectRatioEnum.CINEMATIC_21_9.value: CanvasDimension(
        width=2560, height=1080, aspect_str="21:9", bg_blur_radius=15
    )
}

TYPOGRAPHY_PRESETS: Dict[str, TypographyStyle] = {
    CaptionStyleEnum.HORMOZI_GOLD.value: TypographyStyle(
        name=CaptionStyleEnum.HORMOZI_GOLD.value,
        font_family="Impact.ttf",
        font_size_pct=0.055,
        active_color=(255, 215, 0),
        inactive_color=(255, 255, 255),
        stroke_color=(0, 0, 0),
        stroke_width=5,
        glow_enabled=False,
        glow_color=(0, 0, 0),
        glow_radius=0,
        y_position_pct=0.72,
        word_chunk_size=3
    ),
    CaptionStyleEnum.CYBERPUNK_NEON.value: TypographyStyle(
        name=CaptionStyleEnum.CYBERPUNK_NEON.value,
        font_family="Impact.ttf",
        font_size_pct=0.052,
        active_color=(0, 242, 254),
        inactive_color=(200, 200, 200),
        stroke_color=(10, 10, 25),
        stroke_width=6,
        glow_enabled=True,
        glow_color=(0, 242, 254),
        glow_radius=8,
        y_position_pct=0.75,
        word_chunk_size=3
    ),
    CaptionStyleEnum.IMPACT_RED.value: TypographyStyle(
        name=CaptionStyleEnum.IMPACT_RED.value,
        font_family="Impact.ttf",
        font_size_pct=0.058,
        active_color=(255, 45, 85),
        inactive_color=(240, 240, 240),
        stroke_color=(0, 0, 0),
        stroke_width=6,
        glow_enabled=False,
        glow_color=(0, 0, 0),
        glow_radius=0,
        y_position_pct=0.70,
        word_chunk_size=2
    ),
    CaptionStyleEnum.MINIMAL_WHITE.value: TypographyStyle(
        name=CaptionStyleEnum.MINIMAL_WHITE.value,
        font_family="Impact.ttf",
        font_size_pct=0.048,
        active_color=(255, 255, 255),
        inactive_color=(140, 140, 140),
        stroke_color=(20, 20, 20),
        stroke_width=4,
        glow_enabled=False,
        glow_color=(0, 0, 0),
        glow_radius=0,
        y_position_pct=0.80,
        word_chunk_size=4
    ),
    CaptionStyleEnum.SYNTHWAVE_PINK.value: TypographyStyle(
        name=CaptionStyleEnum.SYNTHWAVE_PINK.value,
        font_family="Impact.ttf",
        font_size_pct=0.054,
        active_color=(255, 0, 127),
        inactive_color=(220, 220, 250),
        stroke_color=(25, 0, 40),
        stroke_width=5,
        glow_enabled=True,
        glow_color=(255, 0, 127),
        glow_radius=10,
        y_position_pct=0.73,
        word_chunk_size=3
    )
}


# ==============================================================================
# 5. ENVIRONMENT & DIAGNOSTIC VALIDATOR
# ==============================================================================
class EnvironmentValidator:
    """Performs deep system diagnostics to verify FFmpeg, Python dependencies, and keys."""

    @staticmethod
    def check_ffmpeg() -> Tuple[bool, str]:
        """Locates system or imageio-ffmpeg binaries."""
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return True, exe
        except Exception as e:
            logger.warning(f"imageio_ffmpeg resolution failed: {e}")

        # Fallback to system PATH lookup
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return True, sys_ffmpeg

        return False, ""

    @staticmethod
    def check_font_file(font_path: str) -> bool:
        """Verifies if the designated Impact font or suitable substitute exists."""
        if os.path.exists(font_path):
            return True
        logger.warning(f"Primary font not found at '{font_path}'. Fallback will be rendered.")
        return False

    @staticmethod
    def inspect_system_hardware() -> Dict[str, Any]:
        """Queries CPU cores, architecture, and memory availability."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_cores": os.cpu_count() or 1,
            "architecture": platform.machine()
        }

    @classmethod
    def run_full_diagnostics(cls, paths: SystemPaths) -> Dict[str, Any]:
        """Runs complete validation suit and logs environmental report."""
        ffmpeg_ok, ffmpeg_path = cls.check_ffmpeg()
        font_ok = cls.check_font_file(paths.font_path)
        hw_info = cls.inspect_system_hardware()

        report = {
            "ffmpeg_installed": ffmpeg_ok,
            "ffmpeg_binary_path": ffmpeg_path,
            "font_present": font_ok,
            "hardware": hw_info
        }

        logger.info("=== VIEWMAX ENVIRONMENT DIAGNOSTICS ===")
        logger.info(f"OS Platform: {hw_info['platform']}")
        logger.info(f"FFmpeg Status: {'AVAILABLE (' + ffmpeg_path + ')' if ffmpeg_ok else 'MISSING'}")
        logger.info(f"Primary Font Status: {'FOUND' if font_ok else 'MISSING (Using Fallback)'}")
        logger.info("=======================================")

        return report


# ==============================================================================
# 6. MASTER APP CONFIGURATION MANAGER
# ==============================================================================
class AppConfig:
    """Master configuration manager aggregating paths, settings, API credentials, and rules."""

    def __init__(self):
        self.paths = SystemPaths()
        self.audio = AudioProcessingSettings()
        self.elevenlabs = ElevenLabsSettings()
        self.openai = OpenAISettings()
        self.whisper = WhisperSettings()
        self.safety = ContentSafetyRules()
        self.default_canvas = AspectRatioEnum.VERTICAL_9_16.value
        self.default_style = CaptionStyleEnum.HORMOZI_GOLD.value
        
        # Ensure directories exist upon initialization
        self.paths.initialize_directories()

    def update_api_keys(self, openai_key: Optional[str] = None, elevenlabs_key: Optional[str] = None) -> None:
        """Dynamically injects API keys from Streamlit UI or environment vars."""
        if openai_key:
            self.openai.api_key = openai_key.strip()
        if elevenlabs_key:
            self.elevenlabs.api_key = elevenlabs_key.strip()

    def load_from_env(self) -> None:
        """Populates API credentials from OS environment variables if available."""
        env_openai = os.getenv("OPENAI_API_KEY", "")
        env_eleven = os.getenv("ELEVENLABS_API_KEY", "")
        self.update_api_keys(openai_key=env_openai, elevenlabs_key=env_eleven)

    def export_to_json(self, json_path: str) -> None:
        """Exports current non-sensitive settings to a JSON file."""
        config_dict = {
            "default_canvas": self.default_canvas,
            "default_style": self.default_style,
            "audio": asdict(self.audio),
            "whisper": asdict(self.whisper),
            "safety": {
                "strict_mode": self.safety.strict_mode,
                "excluded_keywords": self.safety.excluded_keywords
            }
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        logger.info(f"Exported configuration to: {json_path}")

    def load_from_json(self, json_path: str) -> None:
        """Loads configuration overrides from a JSON file."""
        if not os.path.exists(json_path):
            logger.warning(f"Config file not found at: {json_path}")
            return
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.default_canvas = data.get("default_canvas", self.default_canvas)
        self.default_style = data.get("default_style", self.default_style)
        
        if "audio" in data:
            for k, v in data["audio"].items():
                if hasattr(self.audio, k):
                    setattr(self.audio, k, v)
                    
        if "whisper" in data:
            for k, v in data["whisper"].items():
                if hasattr(self.whisper, k):
                    setattr(self.whisper, k, v)
                    
        logger.info(f"Successfully loaded configuration overrides from: {json_path}")


# Instantiate default global configuration singleton
global_config = AppConfig()
global_config.load_from_env()
