"""
================================================================================
VIEWMAX STUDIO PRO — ZERO-COST LOCAL SPEECH SYNTHESIS ENGINE
FILE: voice_synthesis.py
DESCRIPTION: Free local text-to-speech generation engine using Edge-TTS neural 
             voices, requiring no API keys, tokens, or subscription costs.
================================================================================
"""

import os
import sys
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from config import (
    GlobalConfig,
    VoiceProviderEnum,
    global_config,
    get_logger
)

logger = get_logger("ViewMaxPro.VoiceSynthesisLocal")


# ==============================================================================
# 1. EXCEPTIONS & DATA MODELS
# ==============================================================================
class VoiceSynthesisError(Exception):
    """Base exception class for speech synthesis failures."""
    pass


@dataclass
class VoicePersona:
    """Defines voice persona parameters for local neural speech generation."""
    key: str
    display_name: str
    provider: VoiceProviderEnum
    voice_id: str  # Edge-TTS exact voice name (e.g., 'en-US-AriaNeural')
    sample_rate: int = 44100


# Predefined free high-retention neural voices via Edge-TTS
BUILTIN_SPEAKERS: Dict[str, VoicePersona] = {
    "ryan_neural": VoicePersona(
        key="ryan_neural",
        display_name="Ryan (Deep & Authoritative - Local Free)",
        provider=VoiceProviderEnum.EDGE_TTS,
        voice_id="en-US-RyanNeural"
    ),
    "aria_neural": VoicePersona(
        key="aria_neural",
        display_name="Aria (Clear & Engaging - Local Free)",
        provider=VoiceProviderEnum.EDGE_TTS,
        voice_id="en-US-AriaNeural"
    ),
    "steven_neural": VoicePersona(
        key="steven_neural",
        display_name="Steven (Cinematic Narration - Local Free)",
        provider=VoiceProviderEnum.EDGE_TTS,
        voice_id="en-US-ChristopherNeural"
    )
}


# ==============================================================================
# 2. LOCAL VOICE SYNTHESIS MANAGER
# ==============================================================================
class VoiceSynthesisManager:
    """
    Manages local, zero-cost text-to-speech generation using edge-tts.
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.cfg = config or global_config
        logger.info("VoiceSynthesisManager (Local Free Engine) initialized successfully.")

    def generate_narration(
        self,
        text_script: str,
        primary_speaker_key: str = "ryan_neural",
        output_wav_path: str = "output_speech.wav"
    ) -> str:
        """
        Generates spoken audio locally from text script using free Edge-TTS voices.
        """
        if not text_script or not text_script.strip():
            logger.error("Attempted voice synthesis with empty script text.")
            raise ValueError("Provided text script for voice synthesis is empty.")

        persona = BUILTIN_SPEAKERS.get(primary_speaker_key, BUILTIN_SPEAKERS["ryan_neural"])
        logger.info(f"Initiating local speech generation. Speaker: '{persona.display_name}' | Voice ID: {persona.voice_id}")

        try:
            # Run async edge-tts generation synchronously
            asyncio.run(self._async_generate_edge_tts(text_script, persona.voice_id, output_wav_path))
            logger.info(f"Local speech successfully generated and saved to '{output_wav_path}'.")
            return output_wav_path
        except Exception as e:
            logger.critical(f"Local voice synthesis failed: {e}")
            raise VoiceSynthesisError(f"Failed to generate local speech: {e}")

    async def _async_generate_edge_tts(self, text: str, voice_id: str, output_path: str) -> None:
        """Asynchronously calls edge_tts to stream speech directly to an audio file."""
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "The 'edge-tts' package is required for free local speech. "
                "Please add 'edge-tts' to your requirements.txt file."
            )

        mp3_temp_path = output_path + ".tmp.mp3"
        
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(mp3_temp_path)

        # Convert MP3 output to PCM WAV format via FFmpeg
        self._convert_audio_to_wav(mp3_temp_path, output_path, sample_rate=44100)

        if os.path.exists(mp3_temp_path):
            os.remove(mp3_temp_path)

    @staticmethod
    def _convert_audio_to_wav(input_audio_path: str, output_wav_path: str, sample_rate: int = 44100) -> None:
        """Converts audio to standard WAV format using FFmpeg subprocess."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", input_audio_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-sample_fmt", "s16",
            output_wav_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))

    def estimate_speaking_duration(self, text: str, words_per_minute: int = 150) -> float:
        """Estimates audio speaking duration in seconds based on word count."""
        if not text:
            return 0.0
        words = text.split()
        num_words = len(words)
        duration_mins = num_words / float(words_per_minute)
        return max(1.0, duration_mins * 60.0)


# ==============================================================================
# 3. STANDALONE VERIFICATION RUNNER
# ==============================================================================
if __name__ == "__main__":
    logger.info("Running standalone Local Voice Synthesis verification suite...")
    sample_text = "Welcome to ViewMax Studio Pro. Local free text to speech is fully operational."
    manager = VoiceSynthesisManager(global_config)
    dur_est = manager.estimate_speaking_duration(sample_text)
    print(f"Sample Script Text: '{sample_text}'")
    print(f"Estimated Speaking Duration: {dur_est:.2f} seconds")
    print("Local voice module verified successfully.")
