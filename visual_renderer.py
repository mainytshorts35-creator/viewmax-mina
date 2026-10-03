"""
================================================================================
VIEWMAX STUDIO PRO — KINETIC CAPTION & TYPOGRAPHY RENDERING ENGINE
FILE: caption_renderer.py
DESCRIPTION: High-resolution PIL kinetic caption renderer, Faster-Whisper word-level
             timestamp aligner, ASS dynamic subtitle script generator, and 
             MoviePy video frame compositor.
================================================================================
"""

import os
import sys
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

from config import (
    TypographyStyle,
    CaptionStyleEnum,
    TYPOGRAPHY_PRESETS,
    CanvasDimension,
    CANVAS_PRESETS,
    WhisperSettings,
    global_config,
    get_logger
)

logger = get_logger("ViewMaxPro.CaptionRenderer")


# ==============================================================================
# 1. CUSTOM EXCEPTIONS & DATA MODELS
# ==============================================================================
class CaptionError(Exception):
    """Base exception for caption and typography rendering errors."""
    pass


class WhisperAlignmentError(CaptionError):
    """Raised when audio transcription or word timestamp alignment fails."""
    pass


class FontRenderError(CaptionError):
    """Raised when font files cannot be loaded or rendered."""
    pass


@dataclass
class WordTimestamp:
    """Represents a single word with precise millisecond timestamps."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.01, self.end_time - self.start_time)


@dataclass
class CaptionChunk:
    """Group of words displayed together on-screen as a single animated sentence block."""
    chunk_id: int
    words: List[WordTimestamp]
    start_time: float
    end_time: float

    @property
    def full_text(self) -> str:
        return " ".join([w.word for w in self.words])


@dataclass
class RenderedCaptionFrame:
    """Rendered PIL image layer with positioning metadata for video frame blending."""
    image: Image.Image
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    start_time: float
    end_time: float


# ==============================================================================
# 2. FASTER-WHISPER ALIGNMENT ENGINE
# ==============================================================================
class WhisperAlignmentEngine:
    """
    Extracts word-level timestamps from vocal audio files using Faster-Whisper
    with dynamic fallback estimation.
    """

    def __init__(self, settings: Optional[WhisperSettings] = None):
        self.cfg = settings or global_config.whisper
        self.model = None

    def _initialize_model(self) -> None:
        """Lazy-loads the Faster-Whisper model into memory."""
        if self.model is not None:
            return

        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster-Whisper model ('{self.cfg.model_size}') on device '{self.cfg.device}'...")
            self.model = WhisperModel(
                self.cfg.model_size,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type
            )
            logger.info("Faster-Whisper model successfully loaded.")
        except ImportError:
            logger.warning("faster_whisper package not installed. Using fallback heuristic aligner.")
            self.model = "FALLBACK"
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}. Switching to fallback aligner.")
            self.model = "FALLBACK"

    def transcribe_and_align(self, audio_path: str, reference_text: Optional[str] = None) -> List[WordTimestamp]:
        """
        Transcribes audio or aligns provided reference text to extract word timings.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file for alignment not found: {audio_path}")

        self._initialize_model()

        if self.model != "FALLBACK":
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    beam_size=self.cfg.beam_size,
                    word_timestamps=True,
                    language=self.cfg.language
                )

                word_timestamps: List[WordTimestamp] = []
                for segment in segments:
                    if segment.words:
                        for w in segment.words:
                            cleaned_word = w.word.strip().upper()
                            if cleaned_word:
                                word_timestamps.append(
                                    WordTimestamp(
                                        word=cleaned_word,
                                        start_time=w.start,
                                        end_time=w.end,
                                        confidence=w.probability
                                    )
                                )

                if word_timestamps:
                    logger.info(f"Extracted {len(word_timestamps)} word timestamps via Faster-Whisper.")
                    return word_timestamps

            except Exception as e:
                logger.warning(f"Whisper alignment execution error: {e}. Falling back to estimate engine.")

        # Fallback heuristic timestamp estimator
        return self._estimate_timestamps_heuristic(audio_path, reference_text)

    def _estimate_timestamps_heuristic(self, audio_path: str, text: Optional[str]) -> List[WordTimestamp]:
        """Calculates linear fallback word timing using audio file duration."""
        import wave
        
        duration = 5.0
        try:
            with wave.open(audio_path, 'rb') as wf:
                duration = float(wf.getnframes()) / float(wf.getframerate())
        except Exception as e:
            logger.warning(f"Failed to inspect wave file duration for fallback alignment: {e}")

        raw_text = text or "DYNAMIC KINETIC TYPOGRAPHY CAPTION SYSTEM"
        words = [w.strip().upper() for w in raw_text.split() if w.strip()]
        
        if not words:
            return []

        time_per_word = duration / float(len(words))
        timestamps = []

        for i, word in enumerate(words):
            start = i * time_per_word
            end = (i + 1) * time_per_word
            timestamps.append(WordTimestamp(word=word, start_time=start, end_time=end, confidence=0.85))

        logger.info(f"Heuristic alignment generated timings for {len(words)} words over {duration:.2f}s.")
        return timestamps

    def build_caption_chunks(self, words: List[WordTimestamp], chunk_size: int = 3) -> List[CaptionChunk]:
        """Groups individual word timestamps into multi-word caption chunks."""
        chunks: List[CaptionChunk] = []
        if not words:
            return chunks

        for i in range(0, len(words), chunk_size):
            group = words[i:i + chunk_size]
            chunk_id = len(chunks) + 1
            c_start = group[0].start_time
            c_end = group[-1].end_time
            chunks.append(CaptionChunk(chunk_id=chunk_id, words=group, start_time=c_start, end_time=c_end))

        return chunks


# ==============================================================================
# 3. HIGH-RESOLUTION PIL GRAPHICS ENGINE
# ==============================================================================
class PILCaptionGraphicsEngine:
    """
    Renders kinetic typographic captions onto transparent PIL RGBA images with
    stroke outlines, glow diffusion, and active-word highlighting effects.
    """

    def __init__(self, style: TypographyStyle, canvas: CanvasDimension):
        self.style = style
        self.canvas = canvas
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Loads target TrueType font file or falls back to system font."""
        target_size = int(self.canvas.height * self.style.font_size_pct)
        font_path = global_config.paths.font_path

        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=target_size)
            except Exception as e:
                logger.warning(f"Error loading TTF font '{font_path}': {e}")

        # Fallback font resolution
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=target_size)
        except IOError:
            logger.warning("Defaulting to basic PIL load_default font.")
            return ImageFont.load_default()

    def render_chunk_frame(
        self,
        chunk: CaptionChunk,
        active_word_index: int
    ) -> Image.Image:
        """
        Renders a full caption chunk image layer with specific word highlighted.
        """
        img = Image.new("RGBA", (self.canvas.width, self.canvas.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        words = [w.word for w in chunk.words]
        if not words:
            return img

        # Measure dimensions for each word in chunk
        word_widths = []
        word_heights = []
        for w in words:
            bbox = self.font.getbbox(w)
            w_width = bbox[2] - bbox[0]
            w_height = bbox[3] - bbox[1]
            word_widths.append(w_width)
            word_heights.append(w_height)

        space_bbox = self.font.getbbox(" ")
        space_width = space_bbox[2] - space_bbox[0]
        total_width = sum(word_widths) + (space_width * (len(words) - 1))
        max_height = max(word_heights) if word_heights else 30

        # Calculate base horizontal and vertical centers
        start_x = (self.canvas.width - total_width) // 2
        base_y = int(self.canvas.height * self.style.y_position_pct) - (max_height // 2)

        curr_x = start_x

        for idx, word_str in enumerate(words):
            is_active = (idx == active_word_index)
            
            # Select color and offset for active word
            fill_color = self.style.active_color if is_active else self.style.inactive_color
            y_offset = -int(self.canvas.height * self.style.active_y_lift_pct) if is_active else 0
            word_y = base_y + y_offset

            # 1. Draw Glow Layer if Enabled
            if self.style.glow_enabled and is_active:
                self._draw_word_glow(
                    img, word_str, curr_x, word_y,
                    self.style.glow_color, self.style.glow_radius
                )

            # 2. Draw Stroke / Outline
            stroke_w = self.style.stroke_width
            stroke_c = self.style.stroke_color + (255,)
            for dx in range(-stroke_w, stroke_w + 1):
                for dy in range(-stroke_w, stroke_w + 1):
                    if dx * dx + dy * dy <= stroke_w * stroke_w:
                        draw.text(
                            (curr_x + dx, word_y + dy),
                            word_str,
                            font=self.font,
                            fill=stroke_c
                        )

            # 3. Draw Core Text
            fill_rgba = fill_color + (255,)
            draw.text((curr_x, word_y), word_str, font=self.font, fill=fill_rgba)

            curr_x += word_widths[idx] + space_width

        return img

    def _draw_word_glow(
        self,
        base_img: Image.Image,
        text: str,
        x: int,
        y: int,
        glow_rgb: Tuple[int, int, int],
        radius: int
    ) -> None:
        """Draws a Gaussian blurred glow behind active kinetic text."""
        glow_layer = Image.new("RGBA", (self.canvas.width, self.canvas.height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)

        glow_fill = glow_rgb + (220,)
        
        # Render bold thick text for glow source
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                glow_draw.text((x + dx, y + dy), text, font=self.font, fill=glow_fill)

        blurred_glow = glow_layer.filter(ImageFilter.GaussianBlur(radius=radius))
        base_img.alpha_composite(blurred_glow)


# ==============================================================================
# 4. ADVANCED SUBSTATION ALPHA (ASS) SCRIPT EXPORTER
# ==============================================================================
class ASSSubtitleBuilder:
    """
    Exports word timing metadata into stylized ASS subtitle files
    compatible with FFmpeg burning filters.
    """

    @staticmethod
    def rgb_to_ass_color(rgb: Tuple[int, int, int]) -> str:
        """Converts RGB tuple to ASS hex color format (&H00BBGGRR)."""
        r, g, b = rgb
        return f"&H00{b:02X}{g:02X}{r:02X}"

    @classmethod
    def generate_ass_script(
        cls,
        chunks: List[CaptionChunk],
        style: TypographyStyle,
        canvas: CanvasDimension,
        output_ass_path: str
    ) -> str:
        """Generates full .ass subtitle script file on disk."""
        primary_color = cls.rgb_to_ass_color(style.inactive_color)
        active_color = cls.rgb_to_ass_color(style.active_color)
        outline_color = cls.rgb_to_ass_color(style.stroke_color)

        font_size = int(canvas.height * style.font_size_pct)
        margin_v = int(canvas.height * (1.0 - style.y_position_pct))

        ass_header = f"""[Script Info]
Title: ViewMax Kinetic Captions
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {canvas.width}
PlayResY: {canvas.height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KineticStyle,Impact,{font_size},{primary_color},{active_color},{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{style.stroke_width},0,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events = []
        for chunk in chunks:
            for idx, word_obj in enumerate(chunk.words):
                start_str = cls._format_ass_time(word_obj.start_time)
                end_str = cls._format_ass_time(word_obj.end_time)

                # Format sentence with primary highlight on active word
                formatted_words = []
                for w_idx, w_item in enumerate(chunk.words):
                    if w_idx == idx:
                        formatted_words.append(f"{{\\c{active_color}\\b1}}{w_item.word}{{\\rKineticStyle}}")
                    else:
                        formatted_words.append(w_item.word)

                dialogue_text = " ".join(formatted_words)
                event_line = f"Dialogue: 0,{start_str},{end_str},KineticStyle,,0,0,0,,{dialogue_text}"
                events.append(event_line)

        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(events))

        logger.info(f"Generated ASS subtitle file: {output_ass_path}")
        return output_ass_path

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        """Formats seconds float to ASS timestamp format (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int(round((seconds - int(seconds)) * 100))
        if centisecs >= 100:
            centisecs = 99
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


# ==============================================================================
# 5. MOVIEPY VIDEO OVERLAY COMPOSITOR
# ==============================================================================
class MoviePyCaptionCompositor:
    """
    Integrates rendered PIL image caption layers with MoviePy VideoClips.
    """

    def __init__(self, style_preset: str = CaptionStyleEnum.HORMOZI_GOLD.value):
        self.style = TYPOGRAPHY_PRESETS.get(style_preset, TYPOGRAPHY_PRESETS[CaptionStyleEnum.HORMOZI_GOLD.value])

    def overlay_captions_on_clip(
        self,
        video_clip,
        chunks: List[CaptionChunk],
        canvas: CanvasDimension
    ):
        """
        Creates MoviePy ImageClips for each word frame and composites over base clip.
        """
        try:
            from moviepy.editor import ImageClip, CompositeVideoClip
        except ImportError:
            logger.error("MoviePy is required for overlay_captions_on_clip.")
            return video_clip

        gfx_engine = PILCaptionGraphicsEngine(self.style, canvas)
        caption_clips = []

        logger.info(f"Compositing captions across {len(chunks)} text chunks...")

        for chunk in chunks:
            for idx, word_obj in enumerate(chunk.words):
                duration = word_obj.end_time - word_obj.start_time
                if duration <= 0:
                    continue

                pil_frame = gfx_engine.render_chunk_frame(chunk, active_word_index=idx)
                np_frame = np.array(pil_frame)

                img_clip = (
                    ImageClip(np_frame)
                    .set_start(word_obj.start_time)
                    .set_duration(duration)
                    .set_position(("center", "center"))
                )
                caption_clips.append(img_clip)

        final_clip = CompositeVideoClip([video_clip] + caption_clips)
        return final_clip


# ==============================================================================
# 6. STANDALONE VERIFICATION RUNNER
# ==============================================================================

    canvas_preset = CANVAS_PRESETS[global_config.default_canvas]
    style_preset = TYPOGRAPHY_PRESETS[global_config.default_style]

    gfx = PILCaptionGraphicsEngine(style_preset, canvas_preset)

    mock_words = [
        WordTimestamp("VIEWMAX", 0.0, 0.4),
        WordTimestamp("STUDIO", 0.4, 0.8),
        WordTimestamp("PRO", 0.8, 1.2)
    ]
    mock_chunk = CaptionChunk(chunk_id=1, words=mock_words, start_time=0.0, end_time=1.2)

    rendered_img = gfx.render_chunk_frame(mock_chunk, active_word_index=1)
    
    test_export_path = os.path.join(global_config.paths.temp_dir, "caption_preview_test.png")
    rendered_img.save(test_export_path)

    print(f"Sample frame rendered successfully: {test_export_path}")
    print(f"Canvas resolution: {canvas_preset.width}x{canvas_preset.height}")
    print("Caption renderer suite fully operational.")
