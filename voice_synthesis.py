"""
================================================================================
VIEWMAX STUDIO PRO — MULTI-PROVIDER AI VOICE SYNTHESIS ENGINE
FILE: voice_synthesis.py
DESCRIPTION: Enterprise text-to-speech engine supporting ElevenLabs Multilingual v2
             and OpenAI TTS-1-HD. Features automatic failover, exponential backoff,
             vocal profile management, duration estimation, and audio stitching.
================================================================================
"""

import os
import sys
import time
import json
import wave
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

from config import (
    AppConfig,
    ElevenLabsSettings,
    OpenAISettings,
    VoiceProviderEnum,
    global_config,
    get_logger
)

logger = get_logger("ViewMaxPro.VoiceSynthesis")


# ==============================================================================
# 1. CUSTOM EXCEPTIONS & DATA MODELS
# ==============================================================================
class VoiceSynthesisError(Exception):
    """Base exception for all voice synthesis operations."""
    pass


class ElevenLabsAPIError(VoiceSynthesisError):
    """Raised when ElevenLabs API returns an error status code or network failure."""
    pass


class OpenAITTSAPIError(VoiceSynthesisError):
    """Raised when OpenAI TTS service fails or fails authorization."""
    pass


class QuotaExceededError(VoiceSynthesisError):
    """Raised when API rate limit or character credit quota is exhausted."""
    pass


class VoiceConfigError(VoiceSynthesisError):
    """Raised when API credentials or requested voice IDs are unconfigured."""
    pass


@dataclass
class VoiceSpeakerProfile:
    """Metadata profile describing a voice voice persona."""
    speaker_id: str
    display_name: str
    provider: VoiceProviderEnum
    voice_id: str
    gender: str = "neutral"
    accent: str = "american"
    description: str = ""


@dataclass
class VoiceScriptSegment:
    """Single narrative text segment for multi-speaker or chunked generation."""
    segment_id: int
    text: str
    speaker_profile: VoiceSpeakerProfile
    estimated_duration_sec: float = 0.0


@dataclass
class SynthesisResult:
    """Metadata result returned after successfully rendering audio speech synthesis."""
    output_path: str
    duration_seconds: float
    provider_used: VoiceProviderEnum
    voice_id: str
    text_length_chars: int
    cost_estimate_usd: float


# ==============================================================================
# 2. PRESET SPEAKER REGISTRY
# ==============================================================================
BUILTIN_SPEAKERS: Dict[str, VoiceSpeakerProfile] = {
    "adam_narration": VoiceSpeakerProfile(
        speaker_id="adam_narration",
        display_name="Adam — Deep Cinematic (ElevenLabs)",
        provider=VoiceProviderEnum.ELEVENLABS,
        voice_id="pNInz6obpgDQGcFmaJgB",
        gender="male",
        accent="american",
        description="Deep, authoritative narrative tone suitable for documentary shorts."
    ),
    "rachel_storytelling": VoiceSpeakerProfile(
        speaker_id="rachel_storytelling",
        display_name="Rachel — Energetic Storyteller (ElevenLabs)",
        provider=VoiceProviderEnum.ELEVENLABS,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        gender="female",
        accent="american",
        description="Warm, engaging, and highly expressive voice for high-retention stories."
    ),
    "onyx_bold": VoiceSpeakerProfile(
        speaker_id="onyx_bold",
        display_name="Onyx — Intense & Bold (OpenAI)",
        provider=VoiceProviderEnum.OPENAI_HD,
        voice_id="onyx",
        gender="male",
        accent="american",
        description="Deep resonant male voice engineered for intense documentary hooks."
    ),
    "nova_energetic": VoiceSpeakerProfile(
        speaker_id="nova_energetic",
        display_name="Nova — Vibrant & Dynamic (OpenAI)",
        provider=VoiceProviderEnum.OPENAI_HD,
        voice_id="nova",
        gender="female",
        accent="american",
        description="Vibrant, quick-paced female voice ideal for fast YouTube Shorts."
    ),
    "alloy_neutral": VoiceSpeakerProfile(
        speaker_id="alloy_neutral",
        display_name="Alloy — Balanced Neutral (OpenAI)",
        provider=VoiceProviderEnum.OPENAI_HD,
        voice_id="alloy",
        gender="neutral",
        accent="american",
        description="Clear, balanced, neutral broadcast voice for informative clips."
    )
}


# ==============================================================================
# 3. BASE ABSTRACT VOICE PROVIDER INTERFACE
# ==============================================================================
class BaseVoiceProvider(ABC):
    """Abstract interface defining required contract for voice synthesis providers."""

    @abstractmethod
    def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_wav_path: str
    ) -> SynthesisResult:
        """Renders text script to standard WAV audio output file."""
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Verifies validity of configured API credentials."""
        pass


# ==============================================================================
# 4. ELEVENLABS MULTILINGUAL V2 PROVIDER IMPLEMENTATION
# ==============================================================================
class ElevenLabsTTSProvider(BaseVoiceProvider):
    """High-fidelity voice synthesis engine utilizing ElevenLabs REST API v1."""

    def __init__(self, settings: Optional[ElevenLabsSettings] = None):
        self.cfg = settings or global_config.elevenlabs

    def validate_credentials(self) -> bool:
        """Tests validity of current ElevenLabs API key against /v1/user endpoint."""
        if not self.cfg.api_key:
            return False

        headers = {"xi-api-key": self.cfg.api_key}
        try:
            resp = requests.get(
                "https://api.elevenlabs.io/v1/user",
                headers=headers,
                timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"ElevenLabs credential verification request failed: {e}")
            return False

    def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_wav_path: str
    ) -> SynthesisResult:
        """Queries ElevenLabs TTS endpoint with automatic retries and exponential backoff."""
        if not self.cfg.api_key:
            raise VoiceConfigError("ElevenLabs API key is missing or unconfigured.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.cfg.api_key
        }

        payload = {
            "text": text,
            "model_id": self.cfg.default_model_id,
            "voice_settings": {
                "stability": self.cfg.stability,
                "similarity_boost": self.cfg.similarity_boost,
                "style": self.cfg.style,
                "use_speaker_boost": self.cfg.use_speaker_boost
            }
        }

        max_retries = 3
        backoff_sec = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"ElevenLabs TTS rendering attempt {attempt}/{max_retries} for voice '{voice_id}'...")
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.cfg.request_timeout_sec,
                    stream=True
                )

                if response.status_code == 200:
                    temp_mp3 = output_wav_path + ".tmp.mp3"
                    with open(temp_mp3, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Convert streaming MP3 to PCM WAV using FFmpeg
                    from audio_dsp import VocalMasteringProcessor
                    self._convert_mp3_to_wav(temp_mp3, output_wav_path)
                    
                    if os.path.exists(temp_mp3):
                        os.remove(temp_mp3)

                    # Measure duration
                    duration = self._get_wav_duration(output_wav_path)
                    cost = (len(text) / 1000.0) * 0.030  # Estimated cost per 1k chars

                    return SynthesisResult(
                        output_path=output_wav_path,
                        duration_seconds=duration,
                        provider_used=VoiceProviderEnum.ELEVENLABS,
                        voice_id=voice_id,
                        text_length_chars=len(text),
                        cost_estimate_usd=cost
                    )

                elif response.status_code == 429:
                    logger.warning("ElevenLabs rate limit exceeded (429). Retrying after delay...")
                    time.sleep(backoff_sec)
                    backoff_sec *= 2.0
                elif response.status_code in [401, 403]:
                    raise ElevenLabsAPIError("Invalid ElevenLabs API Key or unauthorized quota access.")
                else:
                    raise ElevenLabsAPIError(f"ElevenLabs API request failed with status code {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error during ElevenLabs request: {e}")
                if attempt == max_retries:
                    raise ElevenLabsAPIError(f"Exhausted retries connecting to ElevenLabs: {e}")
                time.sleep(backoff_sec)

        raise ElevenLabsAPIError("ElevenLabs voice synthesis failed after multiple attempts.")

    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> None:
        """Utility wrapper executing FFmpeg conversion from MP3 to WAV format."""
        import subprocess
        from config import EnvironmentValidator, global_config
        
        _, ffmpeg_bin = EnvironmentValidator.check_ffmpeg()
        if not ffmpeg_bin:
            ffmpeg_bin = "ffmpeg"

        cmd = [
            ffmpeg_bin, "-y",
            "-i", mp3_path,
            "-ar", "44100",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise VoiceSynthesisError(f"FFmpeg MP3-to-WAV conversion failed: {result.stderr.decode()}")

    def _get_wav_duration(self, wav_path: str) -> float:
        """Extracts total playback duration in seconds from WAV header."""
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return float(frames) / float(rate)


# ==============================================================================
# 5. OPENAI TTS-1-HD PROVIDER IMPLEMENTATION
# ==============================================================================
class OpenAITTSProvider(BaseVoiceProvider):
    """High-speed vocal synthesis provider utilizing OpenAI /v1/audio/speech REST API."""

    def __init__(self, settings: Optional[OpenAISettings] = None):
        self.cfg = settings or global_config.openai

    def validate_credentials(self) -> bool:
        """Validates OpenAI API key presence and format standard."""
        if not self.cfg.api_key or not self.cfg.api_key.startswith("sk-"):
            return False
        return True

    def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        output_wav_path: str
    ) -> SynthesisResult:
        """Generates vocal speech via OpenAI Audio API."""
        if not self.validate_credentials():
            raise VoiceConfigError("OpenAI API key is unconfigured or malformed.")

        # Default fallback voice if voice_id is not native to OpenAI
        valid_openai_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        selected_voice = voice_id.lower() if voice_id.lower() in valid_openai_voices else "onyx"

        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.cfg.tts_model,
            "input": text,
            "voice": selected_voice,
            "response_format": "wav",
            "speed": 1.05  # Slight speed boost for punchy YouTube Shorts retention
        }

        logger.info(f"Rendering OpenAI TTS audio with voice '{selected_voice}'...")
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.cfg.timeout_sec
            )

            if response.status_code == 200:
                with open(output_wav_path, "wb") as f:
                    f.write(response.content)

                with wave.open(output_wav_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = float(frames) / float(rate)

                cost = (len(text) / 1000.0) * 0.015  # OpenAI TTS-1-HD cost rate

                return SynthesisResult(
                    output_path=output_wav_path,
                    duration_seconds=duration,
                    provider_used=VoiceProviderEnum.OPENAI_HD,
                    voice_id=selected_voice,
                    text_length_chars=len(text),
                    cost_estimate_usd=cost
                )

            elif response.status_code == 429:
                raise QuotaExceededError("OpenAI API rate limit or quota exceeded.")
            else:
                raise OpenAITTSAPIError(f"OpenAI TTS API error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            raise OpenAITTSAPIError(f"Connection failure to OpenAI TTS API: {e}")


# ==============================================================================
# 6. MASTER ORCHESTRATOR & FALLBACK MANAGER
# ==============================================================================
class VoiceSynthesisManager:
    """
    Master speech orchestrator managing primary and secondary provider failover,
    script chunking, audio concatenations, and duration checks.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.app_cfg = config or global_config
        self.elevenlabs_provider = ElevenLabsTTSProvider(self.app_cfg.elevenlabs)
        self.openai_provider = OpenAITTSProvider(self.app_cfg.openai)

    def estimate_speaking_duration(self, text: str, wpm: float = 150.0) -> float:
        """Estimates spoken speech duration in seconds given target words-per-minute rate."""
        word_count = len(text.strip().split())
        words_per_second = wpm / 60.0
        return round(word_count / words_per_second, 2)

    def generate_narration(
        self,
        text_script: str,
        primary_speaker_key: str = "adam_narration",
        output_wav_path: str = ""
    ) -> SynthesisResult:
        """
        Synthesizes text script using chosen primary provider with automatic failover 
        to secondary backup provider upon error.
        """
        if not output_wav_path:
            output_wav_path = os.path.join(
                self.app_cfg.paths.temp_dir,
                f"narration_{int(time.time())}.wav"
            )

        speaker = BUILTIN_SPEAKERS.get(primary_speaker_key, BUILTIN_SPEAKERS["adam_narration"])
        logger.info(f"Initiating narration synthesis for script ({len(text_script)} chars) using speaker: {speaker.display_name}")

        # Primary Execution Attempt
        try:
            if speaker.provider == VoiceProviderEnum.ELEVENLABS and self.elevenlabs_provider.validate_credentials():
                return self.elevenlabs_provider.synthesize_speech(
                    text=text_script,
                    voice_id=speaker.voice_id,
                    output_wav_path=output_wav_path
                )
            elif speaker.provider == VoiceProviderEnum.OPENAI_HD and self.openai_provider.validate_credentials():
                return self.openai_provider.synthesize_speech(
                    text=text_script,
                    voice_id=speaker.voice_id,
                    output_wav_path=output_wav_path
                )
        except Exception as primary_error:
            logger.error(f"Primary voice synthesis provider failed: {primary_error}. Triggering automatic failover...")

        # Failover Execution Attempt
        try:
            logger.info("Executing failover attempt via OpenAI TTS-1-HD provider...")
            return self.openai_provider.synthesize_speech(
                text=text_script,
                voice_id="onyx",
                output_wav_path=output_wav_path
            )
        except Exception as failover_error:
            logger.critical(f"Failover voice synthesis attempt also failed: {failover_error}")
            raise VoiceSynthesisError("All voice synthesis providers exhausted or unconfigured.")

    def stitch_audio_segments(self, segment_paths: List[str], output_combined_wav: str) -> str:
        """Concatenates multiple audio WAV segments into a seamless master track."""
        if not segment_paths:
            raise VoiceSynthesisError("No audio segments provided for stitching.")

        data = []
        params = None

        for path in segment_paths:
            with wave.open(path, 'rb') as wf:
                if params is None:
                    params = wf.getparams()
                data.append(wf.readframes(wf.getnframes()))

        with wave.open(output_combined_wav, 'wb') as wf:
            wf.setparams(params)
            for frame_chunk in data:
                wf.writeframes(frame_chunk)

        logger.info(f"Successfully stitched {len(segment_paths)} audio segments into '{output_combined_wav}'")
        return output_combined_wav


# ==============================================================================
# 7. STANDALONE VERIFICATION RUNNER
# ==============================================================================
if __name__ == "__main__":
    logger.info("Running standalone Voice Synthesis Engine test...")
    manager = VoiceSynthesisManager()

    sample_text = (
        "In the darkest depths of the ocean, strange luminescence illuminates "
        "creatures that have never seen the light of day. Here is what science discovered."
    )

    dur_est = manager.estimate_speaking_duration(sample_text)
    print(f"Sample Script Word Count: {len(sample_text.split())} words")
    print(f"Estimated Speaking Time: {dur_est} seconds")
    print("Voice providers initialized and ready for deployment.")
