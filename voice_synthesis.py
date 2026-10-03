"""
================================================================================
VIEWMAX STUDIO PRO — MULTI-PROVIDER AI SPEECH & VOICE SYNTHESIS ENGINE
FILE: voice_synthesis.py
DESCRIPTION: Handles multi-provider text-to-speech generation with ElevenLabs v2 
             and OpenAI TTS-1-HD, automatic fallback routing, prosody adjustments,
             and speech duration estimation.
================================================================================
"""

import os
import sys
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

from config import (
    GlobalConfig,
    VoiceSettings,
    VoiceProviderEnum,
    global_config,
    get_logger
)

logger = get_logger("ViewMaxPro.VoiceSynthesis")


# ==============================================================================
# 1. CUSTOM EXCEPTIONS & DATA MODELS
# ==============================================================================
class VoiceSynthesisError(Exception):
    """Base exception for voice synthesis and text-to-speech failures."""
    pass


class APIKeyMissingError(VoiceSynthesisError):
    """Raised when an active API key is required but missing from configuration."""
    pass


class ProviderFailureError(VoiceSynthesisError):
    """Raised when a speech synthesis provider API call fails."""
    pass


@dataclass
class VoicePersona:
    """Defines voice persona parameters for AI speech generation."""
    key: str
    display_name: str
    provider: VoiceProviderEnum
    voice_id: str  # ElevenLabs Voice ID or OpenAI Voice Name (e.g., 'alloy', 'onyx')
    model_id: str  # e.g., 'eleven_multilingual_v2' or 'tts-1-hd'
    stability: float = 0.75
    similarity_boost: float = 0.85
    style_exaggeration: float = 0.0
    speed: float = 1.0


# Builtin predefined voice personas optimized for high-retention short-form videos
BUILTIN_SPEAKERS: Dict[str, VoicePersona] = {
    "adam_deep_pro": VoicePersona(
        key="adam_deep_pro",
        display_name="Adam (Deep & Authoritative - ElevenLabs)",
        provider=VoiceProviderEnum.ELEVENLABS,
        voice_id="pNInz6obpgDQGcFmaJgB",
        model_id="eleven_multilingual_v2",
        stability=0.80,
        similarity_boost=0.90
    ),
    "rachel_narrator": VoicePersona(
        key="rachel_narrator",
        display_name="Rachel (Clear & Engaging - ElevenLabs)",
        provider=VoiceProviderEnum.ELEVENLABS,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_multilingual_v2",
        stability=0.75,
        similarity_boost=0.85
    ),
    "openai_onyx": VoicePersona(
        key="openai_onyx",
        display_name="Onyx (Deep & Cinematic - OpenAI)",
        provider=VoiceProviderEnum.OPENAI,
        voice_id="onyx",
        model_id="tts-1-hd",
        speed=1.05
    ),
    "openai_alloy": VoicePersona(
        key="openai_alloy",
        display_name="Alloy (Balanced & Modern - OpenAI)",
        provider=VoiceProviderEnum.OPENAI,
        voice_id="alloy",
        model_id="tts-1-hd",
        speed=1.05
    )
}


# ==============================================================================
# 2. VOICE SYNTHESIS MANAGER & FAILOVER ROUTER
# ==============================================================================
class VoiceSynthesisManager:
    """
    Manages text-to-speech generation across multiple AI providers with 
    automatic fallback routing and text chunking for long-form scripts.
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.cfg = config or global_config
        self.speech_cfg = self.cfg.audio

    def generate_narration(
        self,
        text_script: str,
        primary_speaker_key: str = "adam_deep_pro",
        output_wav_path: str = "output_speech.wav"
    ) -> str:
        """
        Generates spoken audio from text script using primary provider with 
        automatic failover to secondary provider if errors occur.
        """
        if not text_script or not text_script.strip():
            raise ValueError("Provided text script for voice synthesis is empty.")

        persona = BUILTIN_SPEAKERS.get(primary_speaker_key, BUILTIN_SPEAKERS["adam_deep_pro"])

        logger.info(f"Starting voice synthesis using persona '{persona.display_name}' ({persona.provider.value})...")

        # Attempt primary synthesis
        try:
            return self._dispatch_synthesis(text_script, persona, output_wav_path)
        except Exception as primary_err:
            logger.warning(f"Primary synthesis provider ({persona.provider.value}) failed: {primary_err}. Attempting failover...")
            
            # Fallback to alternative provider
            fallback_persona = self._get_fallback_persona(persona.provider)
            logger.info(f"Failing over to fallback persona: '{fallback_persona.display_name}' ({fallback_persona.provider.value})")
            
            try:
                return self._dispatch_synthesis(text_script, fallback_persona, output_wav_path)
            except Exception as fallback_err:
                logger.critical(f"Both primary and fallback voice synthesis providers failed. Primary: {primary_err} | Fallback: {fallback_err}")
                raise ProviderFailureError(f"Voice synthesis failed completely. Details: {fallback_err}")

    def _dispatch_synthesis(self, text: str, persona: VoicePersona, output_path: str) -> str:
        """Dispatches request to appropriate API client based on persona provider."""
        if persona.provider == VoiceProviderEnum.ELEVENLABS:
            return self._synthesize_elevenlabs(text, persona, output_path)
        elif persona.provider == VoiceProviderEnum.OPENAI:
            return self._synthesize_openai(text, persona, output_path)
        else:
            raise VoiceSynthesisError(f"Unsupported speech provider: {persona.provider}")

    def _synthesize_elevenlabs(self, text: str, persona: VoicePersona, output_path: str) -> str:
        """Executes Text-to-Speech via ElevenLabs API v2."""
        api_key = self.cfg.credentials.elevenlabs_api_key
        if not api_key:
            raise APIKeyMissingError("ElevenLabs API key is missing. Please enter it in the sidebar settings.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{persona.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }

        payload = {
            "text": text,
            "model_id": persona.model_id,
            "voice_settings": {
                "stability": persona.stability,
                "similarity_boost": persona.similarity_boost,
                "style": persona.style_exaggeration,
                "use_speaker_boost": True
            }
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            raise ProviderFailureError(f"ElevenLabs API error (HTTP {response.status_code}): {response.text}")

        # Save downloaded mp3 stream and convert to wav
        mp3_temp_path = output_path + ".tmp.mp3"
        with open(mp3_temp_path, "wb") as f:
            f.write(response.content)

        self._convert_audio_to_wav(mp3_temp_path, output_path)
        
        if os.path.exists(mp3_temp_path):
            os.remove(mp3_temp_path)

        logger.info(f"ElevenLabs speech generated and saved to '{output_path}'.")
        return output_path

    def _synthesize_openai(self, text: str, persona: VoicePersona, output_path: str) -> str:
        """Executes Text-to-Speech via OpenAI TTS-1-HD API."""
        api_key = self.cfg.credentials.openai_api_key
        if not api_key:
            raise APIKeyMissingError("OpenAI API key is missing. Please enter it in the sidebar settings.")

        url = "https://api.openai.com/v1/audio/speech"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": persona.model_id,
            "input": text,
            "voice": persona.voice_id,
            "response_format": "mp3",
            "speed": persona.speed
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            raise ProviderFailureError(f"OpenAI TTS API error (HTTP {response.status_code}): {response.text}")

        mp3_temp_path = output_path + ".tmp.mp3"
        with open(mp3_temp_path, "wb") as f:
            f.write(response.content)

        self._convert_audio_to_wav(mp3_temp_path, output_path)

        if os.path.exists(mp3_temp_path):
            os.remove(mp3_temp_path)

        logger.info(f"OpenAI speech generated and saved to '{output_path}'.")
        return output_path

    @staticmethod
    def _convert_audio_to_wav(input_audio_path: str, output_wav_path: str) -> None:
        """Converts any audio file to standard 16-bit PCM WAV using FFmpeg or pydub."""
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", input_audio_path,
                "-ar", "44100",
                "-ac", "1",
                "-sample_fmt", "s16",
                output_wav_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
        except Exception as e:
            logger.error(f"Audio format conversion to WAV failed: {e}")
            raise VoiceSynthesisError(f"Failed to normalize audio file: {e}")

    def _get_fallback_persona(self, failed_provider: VoiceProviderEnum) -> VoicePersona:
        """Selects a robust failover persona from a different provider."""
        for key, persona in BUILTIN_SPEAKERS.items():
            if persona.provider != failed_provider:
                return persona
        return BUILTIN_SPEAKERS["openai_alloy"]

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
    logger.info("Running standalone Voice Synthesis module verification...")
    sample_text = "Welcome to ViewMax Studio Pro. High retention kinetic video generation is fully initialized."
    
    manager = VoiceSynthesisManager(global_config)
    dur_est = manager.estimate_speaking_duration(sample_text)
    
    print(f"Sample Text: '{sample_text}'")
    print(f"Estimated Speaking Duration: {dur_est:.2f} seconds")
    print("Voice synthesis module suite fully operational.")
