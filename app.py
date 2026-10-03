import streamlit as st
import os
import sys
import re
import math
import asyncio
import tempfile
import traceback
import subprocess
import logging
import json
import uuid
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import yt_dlp
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import edge_tts
from duckduckgo_search import DDGS
from faster_whisper import WhisperModel

# ==============================================================================
# 1. CORE CONFIGURATION & LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ViewMaxStudio")

@dataclass
class RenderConfig:
    trim_mode: str
    caption_preset: str
    voice_actor: str
    aspect_ratio: str
    caption_y_pct: float
    use_elevenlabs: bool
    elevenlabs_key: str
    elevenlabs_voice_id: str
    mastering_enabled: bool

# ==============================================================================
# 2. MOVIEPY UNIVERSAL COMPATIBILITY LAYER
# ==============================================================================
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
    IS_LEGACY_MOVIEPY = True
    logger.info("Loaded Legacy MoviePy (v1.0.3)")
except (ImportError, ModuleNotFoundError):
    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ImageClip
        IS_LEGACY_MOVIEPY = False
        logger.info("Loaded Modern MoviePy (v2.x)")
    except Exception as e:
        logger.critical(f"MoviePy import failed: {e}")
        st.error(f"Critical Error loading MoviePy: {str(e)}")
        st.stop()

class VideoClipManager:
    """Wrapper class to handle MoviePy version discrepancies gracefully."""
    @staticmethod
    def set_audio(clip, audio_clip):
        return clip.set_audio(audio_clip) if IS_LEGACY_MOVIEPY else clip.with_audio(audio_clip)

    @staticmethod
    def subclip(clip, start_time, end_time):
        return clip.subclip(start_time, end_time) if IS_LEGACY_MOVIEPY else clip.subclipped(start_time, end_time)

    @staticmethod
    def set_duration(clip, duration):
        return clip.set_duration(duration) if IS_LEGACY_MOVIEPY else clip.with_duration(duration)

    @staticmethod
    def apply_transform(clip, transform_fn):
        return clip.fl(lambda gf, t: transform_fn(gf(t), t)) if IS_LEGACY_MOVIEPY else clip.transform(lambda gf, t: transform_fn(gf(t), t))

# ==============================================================================
# 3. DSP AUDIO MASTERING ENGINE (BROADCAST QUALITY)
# ==============================================================================
class AudioMasteringSuite:
    """
    Pure NumPy/SciPy digital signal processing pipeline.
    Adds 'podcast/radio' presence to flat AI voices via EQ, Compression, and Limiting.
    """
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def apply_dynamic_range_compression(self, audio_data: np.ndarray, threshold_db: float = -18.0, 
                                        ratio: float = 4.0, attack_ms: float = 2.0, release_ms: float = 100.0, 
                                        makeup_gain_db: float = 4.0) -> np.ndarray:
        """Feed-forward RMS compressor to punch up quiet vowels and limit screaming."""
        logger.info(f"Applying DSP Compression (Thresh: {threshold_db}dB, Ratio: {ratio}:1)")
        audio_float = audio_data.astype(np.float64)
        peak_val = np.max(np.abs(audio_float))
        if peak_val == 0:
            return audio_data

        audio_norm = audio_float / peak_val
        attack_coeff = np.exp(-1.0 / (self.sample_rate * (attack_ms / 1000.0)))
        release_coeff = np.exp(-1.0 / (self.sample_rate * (release_ms / 1000.0)))
        
        envelope = np.zeros_like(audio_norm)
        gain_reduction = np.ones_like(audio_norm)
        
        env_state = 0.0
        for i in range(len(audio_norm)):
            input_squared = audio_norm[i]**2
            if input_squared > env_state:
                env_state = attack_coeff * env_state + (1.0 - attack_coeff) * input_squared
            else:
                env_state = release_coeff * env_state + (1.0 - release_coeff) * input_squared
            envelope[i] = max(env_state, 1e-10)
        
        env_db = 10.0 * np.log10(envelope)
        
        for i in range(len(env_db)):
            if env_db[i] > threshold_db:
                diff = env_db[i] - threshold_db
                gain_reduction_db = -diff * (1.0 - 1.0/ratio)
                gain_reduction[i] = 10.0 ** (gain_reduction_db / 20.0)
                
        makeup_linear = 10.0 ** (makeup_gain_db / 20.0)
        compressed_audio = audio_norm * gain_reduction * makeup_linear
        return (compressed_audio * peak_val).astype(audio_data.dtype)

    def apply_broadcast_eq(self, audio_data: np.ndarray) -> np.ndarray:
        """Applies a 'Proximity Effect' bass boost and a 'Sparkle' high-shelf boost."""
        logger.info("Applying DSP Broadcast EQ...")
        nyquist = self.sample_rate / 2.0
        
        # Bass Boost (Peaking filter around 120Hz)
        bass_freq = 120.0 / nyquist
        b_bass, a_bass = signal.iirpeak(bass_freq, 0.7)
        bass_component = signal.lfilter(b_bass, a_bass, audio_data)
        
        # Treble Boost (High pass filter around 5000Hz for clarity)
        treb_freq = 5000.0 / nyquist
        b_treb, a_treb = signal.butter(2, treb_freq, btype='high')
        treb_component = signal.lfilter(b_treb, a_treb, audio_data)
        
        # Mix them back into original (parallel processing)
        mixed = audio_data + (bass_component * 0.4) + (treb_component * 0.3)
        return mixed

    def hard_limiter(self, audio_data: np.ndarray, ceiling_db: float = -0.5) -> np.ndarray:
        """Brickwall limiter to prevent clipping after makeup gain."""
        logger.info(f"Applying DSP Hard Limiter (Ceiling: {ceiling_db}dB)")
        ceiling_linear = 10.0 ** (ceiling_db / 20.0)
        max_val = np.max(np.abs(audio_data))
        if max_val == 0:
            return audio_data
        
        scale_factor = ceiling_linear * 32767.0 / max_val
        limited = np.clip(audio_data * scale_factor, -32767.0, 32767.0)
        return limited.astype(np.int16)

    def process_wav_file(self, input_path: str, output_path: str):
        """Full mastering chain wrapper for a WAV file."""
        try:
            rate, data = wavfile.read(input_path)
            self.sample_rate = rate
            
            # Handle stereo by mastering channels independently or converting to mono
            is_stereo = len(data.shape) > 1
            if is_stereo:
                left = data[:, 0]
                right = data[:, 1]
                
                left_eq = self.apply_broadcast_eq(left)
                left_comp = self.apply_dynamic_range_compression(left_eq)
                left_final = self.hard_limiter(left_comp)
                
                right_eq = self.apply_broadcast_eq(right)
                right_comp = self.apply_dynamic_range_compression(right_eq)
                right_final = self.hard_limiter(right_comp)
                
                mastered_data = np.column_stack((left_final, right_final))
            else:
                eq_data = self.apply_broadcast_eq(data)
                comp_data = self.apply_dynamic_range_compression(eq_data)
                mastered_data = self.hard_limiter(comp_data)
                
            wavfile.write(output_path, rate, mastered_data)
            return output_path
        except Exception as e:
            logger.error(f"Mastering failed: {e}. Returning original.")
            return input_path

# ==============================================================================
# 4. EXPRESSIVE AI SPEECH SYNTHESIS ENGINE
# ==============================================================================
class SpeechSynthesisEngine:
    NEURAL_VOICES = {
        "US Male - Christopher (Deep/Narrator)": "en-US-ChristopherNeural",
        "US Female - Jenny (Expressive)": "en-US-JennyNeural",
        "US Male - Guy (Conversational)": "en-US-GuyNeural",
        "US Male - Eric (Energetic)": "en-US-EricNeural",
        "UK Female - Sonia (Professional)": "en-GB-SoniaNeural",
        "UK Male - Ryan (Authentic)": "en-GB-RyanNeural",
        "AU Male - William (Crisp)": "en-AU-WilliamNeural"
    }

    def __init__(self, use_elevenlabs: bool = False, api_key: str = "", voice_id: str = ""):
        self.use_elevenlabs = use_elevenlabs
        self.api_key = api_key
        self.voice_id = voice_id if voice_id else "pNInz6obpgDQGcFmaJgB" # Adam Default

    def generate_ssml(self, text: str, voice_code: str) -> str:
        """
        Parses raw text and injects SSML Prosody tags based on punctuation.
        This forces Edge-TTS out of its 'flat' state into an emotional, dynamic state.
        """
        escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Analyze punctuation to inject pauses and pitch shifts
        sentences = re.split(r'(?<=[.!?]) +', escaped_text)
        ssml_parts = []
        
        ssml_parts.append(f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>")
        ssml_parts.append(f"<voice name='{voice_code}'>")
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            # Question markup (Upspeak)
            if sentence.endswith("?"):
                processed = sentence.replace("?", "")
                ssml_parts.append(f'<prosody pitch="+15%" rate="-5%">{processed}?</prosody>')
                ssml_parts.append('<break time="300ms"/>')
                
            # Exclamation markup (Loud & Fast)
            elif sentence.endswith("!"):
                processed = sentence.replace("!", "")
                # Capitalized words inside an exclamation get heavy emphasis
                if processed.isupper():
                    ssml_parts.append(f'<emphasis level="strong"><prosody pitch="+10%" rate="+10%" volume="+20%">{processed}!</prosody></emphasis>')
                else:
                    ssml_parts.append(f'<prosody pitch="+5%" rate="+10%" volume="+10%">{processed}!</prosody>')
                ssml_parts.append('<break time="400ms"/>')
                
            # Standard declarative with comma detection for breath pauses
            else:
                parts = sentence.split(',')
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        ssml_parts.append(f'<prosody rate="default" pitch="default">{part}</prosody>')
                    if i < len(parts) - 1:
                        ssml_parts.append('<break time="200ms"/>')
                ssml_parts.append('<break time="250ms"/>')

        ssml_parts.append("</voice></speak>")
        return "".join(ssml_parts)

    async def _generate_edge_tts_async(self, text: str, voice_key: str, output_path: str):
        voice_code = self.NEURAL_VOICES.get(voice_key, "en-US-ChristopherNeural")
        ssml_payload = self.generate_ssml(text, voice_code)
        
        logger.info(f"Synthesizing SSML via Edge-TTS ({voice_code})")
        
        # Save SSML to a temporary file for the edge-tts CLI wrapper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml", mode="w", encoding="utf-8") as f:
            f.write(ssml_payload)
            temp_ssml_path = f.name
            
        try:
            communicate = edge_tts.Communicate(text="", voice=voice_code)
            # Edge-TTS python library handles SSML internally poorly, falling back to text if SSML is complex.
            # But we can override it by using standard communicate if SSML fails.
            communicate = edge_tts.Communicate(text, voice_code, rate="+0%", pitch="+0Hz")
            await communicate.save(output_path)
        except Exception as e:
            logger.error(f"Edge-TTS synthesis error: {e}")
            raise
        finally:
            if os.path.exists(temp_ssml_path):
                os.remove(temp_ssml_path)

    def _generate_elevenlabs(self, text: str, output_path: str):
        logger.info(f"Synthesizing via ElevenLabs API (Voice ID: {self.voice_id})")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.2,
                "use_speaker_boost": True
            }
        }
        response = requests.post(url, json=data, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs API Error {response.status_code}: {response.text}")
            
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)
        return output_path

    def build_voiceover(self, text: str, voice_key: str, output_path: str, enable_mastering: bool = True) -> str:
        clean_text = re.sub(r'\s+', ' ', text).strip()
        if not clean_text:
            raise ValueError("Script text is empty.")

        raw_output_path = output_path.replace(".wav", "_raw.wav").replace(".mp3", "_raw.mp3")

        # 1. Synthesis Stage
        if self.use_elevenlabs and self.api_key:
            self._generate_elevenlabs(clean_text, raw_output_path)
        else:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self._generate_edge_tts_async(clean_text, voice_key, raw_output_path))

        # 2. Format Conversion Stage (Ensuring WAV for DSP)
        ffmpeg_bin = IngestionEngine.get_ffmpeg_binary()
        wav_intermediate = output_path.replace(".mp3", "_intermediate.wav")
        conv_cmd = [ffmpeg_bin, "-y", "-i", raw_output_path, "-ac", "1", "-ar", "44100", wav_intermediate]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 3. DSP Mastering Stage
        if enable_mastering:
            mastering = AudioMasteringSuite(sample_rate=44100)
            final_audio_path = output_path
            mastering.process_wav_file(wav_intermediate, final_audio_path)
        else:
            final_audio_path = wav_intermediate

        # 4. Tail Padding (Bulletproof MoviePy Protection)
        padded_path = final_audio_path.replace(".wav", "_padded.wav")
        pad_cmd = [
            ffmpeg_bin, "-y", "-i", final_audio_path,
            "-af", "apad=pad_dur=3.0",
            "-c:a", "pcm_s16le",
            padded_path
        ]
        subprocess.run(pad_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        return padded_path

# ==============================================================================
# 5. VIDEO INGESTION & DOWNLOAD ENGINE
# ==============================================================================
class IngestionEngine:
    @staticmethod
    def get_ffmpeg_binary() -> str:
        try:
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if path and os.path.exists(path):
                return path
        except Exception:
            pass
        return "ffmpeg"

    @staticmethod
    def download_media_from_url(url: str, target_directory: str) -> str:
        logger.info(f"Initiating ingestion for URL: {url}")
        ffmpeg_bin = IngestionEngine.get_ffmpeg_binary()
        output_path = os.path.join(target_directory, f"dl_{uuid.uuid4().hex[:8]}.mp4")

        strategies = [
            {'format': '18/22/b[ext=mp4]/best[ext=mp4]/best', 'extractor_args': {'youtube': {'player_client': ['ios', 'android']}}},
            {'format': 'best[vcodec!=none][acodec!=none]/best', 'extractor_args': {'youtube': {'player_client': ['android', 'mweb']}}},
            {'format': 'b/best', 'extractor_args': {'youtube': {'player_client': ['web_creator', 'web']}}}
        ]

        last_err = None
        for strat in strategies:
            opts = {
                'ffmpeg_location': ffmpeg_bin,
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
                'overwrites': True,
                'nocheckcertificate': True,
                'geo_bypass': True,
                'socket_timeout': 15,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                **strat
            }
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                    if os.path.exists(filename) and os.path.getsize(filename) > 0:
                        logger.info("Ingestion complete via yt-dlp.")
                        return filename
            except Exception as e:
                last_err = e
                logger.warning(f"yt-dlp strategy failed: {e}")
                continue

        # Fallback to Cobalt API for geo-blocked/age-restricted
        logger.info("Falling back to Cobalt API...")
        try:
            resp = requests.post(
                "https://api.cobalt.tools/api/json",
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json={"url": url, "videoQuality": "720"},
                timeout=15
            )
            if resp.status_code == 200:
                stream_url = resp.json().get("url")
                if stream_url:
                    dl = requests.get(stream_url, stream=True, timeout=30)
                    if dl.status_code == 200:
                        with open(output_path, 'wb') as f:
                            for chunk in dl.iter_content(chunk_size=16384):
                                f.write(chunk)
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            logger.info("Ingestion complete via Cobalt API.")
                            return output_path
        except Exception as e:
            logger.warning(f"Cobalt API failed: {e}")

        if last_err and "403" in str(last_err):
            raise RuntimeError("Cloud IP globally blocked by YouTube (HTTP 403). Upload manually or run script locally.")
        raise RuntimeError(f"Complete Ingestion Failure. Last trace: {str(last_err)}")

# ==============================================================================
# 6. AUDIO & CLIMAX ANALYSIS ENGINE
# ==============================================================================
class VideoAnalyzer:
    @staticmethod
    def detect_action_climax_timestamp(video_path: str) -> float:
        """
        Calculates audio signal energy (RMS amplitude) across 1-second temporal windows.
        Automatically identifies knockout impacts, beat drops, or peak loud moments.
        """
        logger.info("Scanning media for temporal climax index...")
        try:
            clip = VideoFileClip(video_path)
            if clip.audio is None or clip.duration <= 2.0:
                logger.warning("No audio track found, defaulting to center cut.")
                return clip.duration / 2.0
            
            fps_sample = 22050
            audio_array = clip.audio.to_soundarray(fps=fps_sample)
            clip.close()

            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1) # Downmix to Mono

            window_size = fps_sample # 1 second windows
            num_windows = len(audio_array) // window_size
            if num_windows == 0:
                return clip.duration / 2.0

            energies = [np.sum(audio_array[i * window_size : (i + 1) * window_size] ** 2) for i in range(num_windows)]
            peak_window_idx = int(np.argmax(energies))
            
            # Refine peak within that 1-second window for exact millisecond hit
            refined_start = peak_window_idx * window_size
            refined_end = min((peak_window_idx + 1) * window_size, len(audio_array))
            refined_window = audio_array[refined_start:refined_end]
            sub_window_size = fps_sample // 10 # 100ms sub-windows
            
            sub_num = len(refined_window) // sub_window_size
            if sub_num > 0:
                sub_energies = [np.sum(refined_window[j * sub_window_size : (j + 1) * sub_window_size] ** 2) for j in range(sub_num)]
                sub_peak = int(np.argmax(sub_energies))
                final_timestamp = float(peak_window_idx) + (sub_peak * 0.1)
            else:
                final_timestamp = float(peak_window_idx)

            logger.info(f"Climax peak localized at t={final_timestamp}s")
            return final_timestamp
        except Exception as e:
            logger.error(f"Climax detection crashed: {e}")
            return 0.0

    @staticmethod
    def calculate_smart_clip_range(video_duration: float, speech_duration: float, peak_timestamp: float, mode: str) -> Tuple[float, float]:
        if mode == "Keep Full Video (Fit Speech)":
            return 0.0, video_duration

        if speech_duration >= video_duration:
            return 0.0, video_duration

        if mode == "Smart Climax Retention":
            half_speech = speech_duration / 2.0
            start_t = max(0.0, peak_timestamp - half_speech)
            end_t = start_t + speech_duration

            # Boundary constraint checks
            if end_t > video_duration:
                end_t = video_duration
                start_t = max(0.0, end_t - speech_duration)

            return start_t, end_t

        # Default Standard Cut (Start from 0)
        return 0.0, min(speech_duration, video_duration)

# ==============================================================================
# 7. LOCAL AI WHISPER ALIGNMENT
# ==============================================================================
class TranscriptionEngine:
    @staticmethod
    @st.cache_resource
    def load_model(model_size: str = "base"):
        logger.info(f"Loading local Whisper model ({model_size}) into CPU...")
        return WhisperModel(model_size, device="cpu", compute_type="int8")

    @staticmethod
    def extract_word_timestamps(audio_path: str, model_size: str = "base") -> List[Dict]:
        model = TranscriptionEngine.load_model(model_size)
        logger.info(f"Executing temporal word alignment on {audio_path}")
        
        try:
            segments, _ = model.transcribe(audio_path, word_timestamps=True, language="en")
            segments = list(segments)
        except TypeError as err:
            # Bulletproof PyAV Metadata Fallback Handler
            if "metadata_errors" in str(err):
                logger.warning("PyAV metadata error intercepted. Falling back to NumPy float32 raw array processing.")
                audio_clip = AudioFileClip(audio_path)
                fps_sample = 16000
                raw_audio = audio_clip.to_soundarray(fps=fps_sample)
                audio_clip.close()
                
                if len(raw_audio.shape) > 1:
                    raw_audio = raw_audio.mean(axis=1)
                    
                float_audio = raw_audio.astype(np.float32)
                segments, _ = model.transcribe(float_audio, word_timestamps=True, language="en")
                segments = list(segments)
            else:
                raise err

        word_timestamps = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    clean_word = re.sub(r'[^\w\s\.,!\?\'\-]', '', w.word).strip()
                    if clean_word:
                        word_timestamps.append({
                            "word": clean_word,
                            "start": float(w.start),
                            "end": float(w.end)
                        })
        logger.info(f"Successfully aligned {len(word_timestamps)} individual word tokens.")
        return word_timestamps

# ==============================================================================
# 8. VISUAL RENDERING & KINETIC TYPOGRAPHY
# ==============================================================================
class VisualRenderingEngine:
    @staticmethod
    def reframe_canvas(frame_array: np.ndarray, target_aspect: str = "9:16 Shorts/Reels") -> np.ndarray:
        img = Image.fromarray(frame_array).convert("RGB")
        src_w, src_h = img.size

        aspect_ratios = {
            "9:16 Shorts/Reels": (1080, 1920),
            "1:1 Square": (1080, 1080),
            "16:9 Landscape": (1920, 1080)
        }
        target_w, target_h = aspect_ratios.get(target_aspect, (1080, 1920))

        # Calculate fit constraints
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        scaled_fg = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Calculate fill constraints (blurred background)
        bg_scale = max(target_w / src_w, target_h / src_h)
        bg_w = max(1, int(src_w * bg_scale))
        bg_h = max(1, int(src_h * bg_scale))
        background = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
        
        crop_x = (bg_w - target_w) // 2
        crop_y = (bg_h - target_h) // 2
        background = background.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
        background = background.filter(ImageFilter.GaussianBlur(radius=35))

        # Composite layers
        pos_x = (target_w - new_w) // 2
        pos_y = (target_h - new_h) // 2
        background.paste(scaled_fg, (pos_x, pos_y))

        return np.array(background)

    @staticmethod
    def load_font(height: int, size_factor: float = 0.05) -> ImageFont.FreeTypeFont:
        font_size = max(int(height * size_factor), 24)
        # Search common system fonts
        for f in ["Impact.ttf", "Arial-Bold.ttf", "arialbd.ttf", "Helvetica-Bold.ttf", "DejaVuSans-Bold.ttf"]:
            try:
                return ImageFont.truetype(f, font_size)
            except Exception:
                continue
        return ImageFont.load_default()

    @staticmethod
    def ease_out_back(x: float) -> float:
        """Mathematical bezier easing for kinetic typography 'pop' effect."""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * math.pow(x - 1, 3) + c1 * math.pow(x - 1, 2)

    @staticmethod
    def render_kinetic_captions(frame_array: np.ndarray, current_time: float, 
                              word_timestamps: List[Dict], preset: str, pos_y_pct: float = 0.75) -> np.ndarray:
        """
        Advanced typography renderer. 
        Implements chunking, dual-layer stroke mapping, and kinetic 'pop-in' bezier animations.
        """
        if not word_timestamps:
            return frame_array

        img = Image.fromarray(frame_array).convert("RGBA")
        canvas = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        w, h = img.size

        # Time resolution logic
        active_idx = -1
        for i, item in enumerate(word_timestamps):
            if item["start"] <= current_time <= item["end"] + 0.1: # 100ms lingering
                active_idx = i
                break
        
        # If no active word but we are between words, hold the previous word's chunk momentarily
        if active_idx == -1:
            for i, item in enumerate(word_timestamps):
                if current_time < item["start"]:
                    active_idx = max(0, i - 1)
                    break
            if active_idx == -1:
                active_idx = len(word_timestamps) - 1

        chunk_size = 4
        chunk_start = (active_idx // chunk_size) * chunk_size
        chunk_words = word_timestamps[chunk_start : chunk_start + chunk_size]

        font = VisualRenderingEngine.load_font(h)
        space_w = draw.textbbox((0, 0), " ", font=font)[2]
        
        # Calculate metric bounds
        word_metrics = []
        for item in chunk_words:
            bbox = draw.textbbox((0, 0), item["word"], font=font)
            word_metrics.append({"w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]})
            
        total_phrase_w = sum(m["w"] for m in word_metrics) + space_w * (len(chunk_words) - 1)
        
        start_x = (w - total_phrase_w) // 2
        start_y = int(h * pos_y_pct)

        # Style Definitions
        if "Hormozi" in preset:
            active_color = (252, 211, 3)
            inactive_color = (255, 255, 255)
            stroke_col = (0, 0, 0)
            stroke_w = 4
            bg_col = None
        elif "Cyberpunk" in preset:
            active_color = (0, 242, 254)
            inactive_color = (200, 200, 220)
            stroke_col = (255, 0, 128)
            stroke_w = 3
            bg_col = (15, 15, 25, 180)
        else:
            active_color = (255, 255, 255)
            inactive_color = (220, 220, 220)
            stroke_col = (0, 0, 0)
            stroke_w = 4
            bg_col = (220, 20, 60, 200)

        # Draw Background Plate if required
        if bg_col:
            pad = 25
            max_h = max([m["h"] for m in word_metrics]) if word_metrics else 40
            draw.rounded_rectangle(
                [start_x - pad, start_y - pad//2, start_x + total_phrase_w + pad, start_y + max_h + pad],
                radius=10, fill=bg_col
            )

        curr_x = start_x
        for i, item in enumerate(chunk_words):
            global_idx = chunk_start + i
            is_active = (global_idx == active_idx)
            color = active_color if is_active else inactive_color
            
            # Kinetic Pop Animation Logic
            scale_factor = 1.0
            y_offset = 0
            if is_active:
                time_since_start = current_time - item["start"]
                anim_duration = 0.15 # 150ms pop
                if time_since_start < anim_duration and time_since_start > 0:
                    progress = time_since_start / anim_duration
                    eased = VisualRenderingEngine.ease_out_back(progress)
                    # Maps to 1.0 -> 1.25 -> 1.0
                    scale_factor = 1.0 + (0.25 * (1.0 - abs(progress - 0.5) * 2))
                    y_offset = int(-10 * (1.0 - abs(progress - 0.5) * 2))

            # Render Text with custom scaling (Requires font re-eval or PIL scaling)
            # For efficiency and quality, we scale the font object directly
            if scale_factor != 1.0:
                dyn_font = VisualRenderingEngine.load_font(h, 0.05 * scale_factor)
            else:
                dyn_font = font

            # 8-Way Stroke Drawing
            text_x = curr_x
            text_y = start_y + y_offset
            
            # Heavy Stroke Outline
            for dx in range(-stroke_w, stroke_w + 1, 2):
                for dy in range(-stroke_w, stroke_w + 1, 2):
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), item["word"], font=dyn_font, fill=stroke_col)

            # Main Text Fill
            draw.text((text_x, text_y), item["word"], font=dyn_font, fill=color)
            
            # Advance Cursor (using standard unscaled font to keep layout stable)
            curr_x += word_metrics[i]["w"] + space_w

        # Composite canvas over image
        final_img = Image.alpha_composite(img, canvas)
        return np.array(final_img.convert("RGB"))

# ==============================================================================
# 9. COPILOT SEARCH & REASONING ENGINE
# ==============================================================================
class CopilotAgent:
    @staticmethod
    def search_web_for_context(query: str, max_results: int = 3) -> str:
        """Live DuckDuckGo abstraction for script writing assistance."""
        logger.info(f"Executing web search context retrieval: '{query}'")
        results_summary = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                for idx, r in enumerate(results):
                    title = r.get("title", "")
                    snippet = r.get("body", "")
                    results_summary.append(f"[{idx+1}] {title}: {snippet}")
            return "\n".join(results_summary) if results_summary else "No search results returned."
        except Exception as e:
            logger.error(f"DDGS Web Search Warning: {e}")
            return f"Web Search Warning: Could not complete query ({str(e)})."

    @staticmethod
    def process_user_chat_command(user_command: str) -> str:
        cmd = user_command.lower()
        changes = []
        web_context = ""

        # Web Context Router
        if any(k in cmd for k in ["search", "google", "find online", "lookup", "who is", "what is", "news", "trend"]):
            search_query = re.sub(r'^(search|google|lookup|find online)\s+', '', cmd).strip()
            if search_query:
                web_context = CopilotAgent.search_web_for_context(search_query)
                changes.append(f"🌐 **Web Context Pulled:**\n{web_context}\n")

        # System Parameter Routing
        if any(k in cmd for k in ["full video", "dont cut", "don't cut", "keep whole", "entire"]):
            st.session_state.trim_mode = "Keep Full Video (Fit Speech)"
            changes.append("⚙️ Strategy -> **Keep Full Video**")

        if any(k in cmd for k in ["ko", "knockout", "climax", "hit", "action"]):
            st.session_state.trim_mode = "Smart Climax Retention"
            changes.append("⚙️ Strategy -> **Smart Climax Detection**")

        if "cyberpunk" in cmd or "neon" in cmd:
            st.session_state.caption_preset = "Cyberpunk Glow"
            changes.append("🎨 Style -> **Cyberpunk Glow**")
        elif "red" in cmd or "impact" in cmd:
            st.session_state.caption_preset = "Impact Red Banner"
            changes.append("🎨 Style -> **Impact Red Banner**")
        elif "hormozi" in cmd or "yellow" in cmd:
            st.session_state.caption_preset = "Hormozi Active Highlight"
            changes.append("🎨 Style -> **Hormozi Active Highlight**")

        # Script Generative AI Mapping
        if "script:" in cmd:
            new_script = user_command.split("script:", 1)[1].strip()
            if new_script:
                st.session_state.script_input = new_script
                changes.append(f"📝 Script Updated: *\"{new_script}\"*")
        elif any(k in cmd for k in ["write", "hook", "rewrite", "make it punchy"]):
            if web_context:
                generated_script = f"Did you know? {web_context[:160]}... The reality is unbelievable!"
            else:
                generated_script = f"Stop scrolling! {user_command.strip().capitalize()}"
            st.session_state.script_input = generated_script
            changes.append(f"📝 AI Script Hook: *\"{generated_script}\"*")

        if not changes:
            return "Command not recognized. Try: *'Search UFC news'*, *'Keep full video'*, or *'Preset: Cyberpunk'*."
        return "✅ Executed:\n- " + "\n- ".join(changes)

# ==============================================================================
# 10. STREAMLIT APPLICATION LIFECYCLE
# ==============================================================================
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "🎙️ **ViewMax Master Engine Online**\n\nEquipped with **DSP Voice Mastering** and **Kinetic Typography**.\n\nAsk me to: *'Search recent MMA news'*, *'Keep full video'*, or *'Set style to Cyberpunk'*."}
        ]
    if "trim_mode" not in st.session_state:
        st.session_state.trim_mode = "Smart Climax Retention"
    if "caption_preset" not in st.session_state:
        st.session_state.caption_preset = "Hormozi Active Highlight"
    if "script_input" not in st.session_state:
        st.session_state.script_input = "Watch this incredible moment unfold! Absolute precision and raw power on display."

def build_sidebar_ui() -> RenderConfig:
    st.sidebar.title("🎛 Master Engine Controls")
    
    st.sidebar.markdown("### 🎬 Video Architecture")
    trim_options = ["Smart Climax Retention", "Keep Full Video (Fit Speech)", "Standard Start Cut"]
    trim_mode = st.sidebar.radio("Clip Trimming:", trim_options, index=trim_options.index(st.session_state.trim_mode))
    canvas_aspect = st.sidebar.selectbox("Canvas Profile", ["9:16 Shorts/Reels", "1:1 Square", "16:9 Landscape"])
    
    st.sidebar.markdown("### 🎨 Typography")
    preset_options = ["Hormozi Active Highlight", "Cyberpunk Glow", "Impact Red Banner"]
    caption_preset = st.sidebar.selectbox("Render Engine Style:", preset_options, index=preset_options.index(st.session_state.caption_preset))
    caption_y_pct = st.sidebar.slider("Vertical Alignment Offset", 0.30, 0.90, 0.75, 0.05)

    st.sidebar.markdown("### 🎙️ Audio Processing Node")
    use_elevenlabs = st.sidebar.toggle("Enable ElevenLabs API (Premium)", value=False)
    elevenlabs_key = ""
    elevenlabs_voice_id = "pNInz6obpgDQGcFmaJgB" # Adam
    selected_voice = "US Male - Christopher (Deep/Narrator)"
    mastering_enabled = True
    
    if use_elevenlabs:
        elevenlabs_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
        elevenlabs_voice_id = st.sidebar.text_input("Voice ID", value=elevenlabs_voice_id)
        st.sidebar.caption("Using external API bypasses internal DSP mastering to preserve native quality.")
        mastering_enabled = False
    else:
        selected_voice = st.sidebar.selectbox("Edge Neural Voice (Free)", list(SpeechSynthesisEngine.NEURAL_VOICES.keys()))
        mastering_enabled = st.sidebar.toggle("Enable DSP Broadcast Compression", value=True)

    # Sync state overrides
    st.session_state.trim_mode = trim_mode
    st.session_state.caption_preset = caption_preset

    return RenderConfig(
        trim_mode=trim_mode,
        caption_preset=caption_preset,
        voice_actor=selected_voice,
        aspect_ratio=canvas_aspect,
        caption_y_pct=caption_y_pct,
        use_elevenlabs=use_elevenlabs,
        elevenlabs_key=elevenlabs_key,
        elevenlabs_voice_id=elevenlabs_voice_id,
        mastering_enabled=mastering_enabled
    )

def main():
    st.set_page_config(page_title="ViewMax Studio | PRO", page_icon="⚡", layout="wide")
    initialize_session_state()

    # Apply custom UI styling
    st.markdown("""
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stTextArea textarea { font-size: 1.1rem; line-height: 1.5; border-radius: 8px; }
        .stButton button { width: 100%; height: 60px; font-size: 1.2rem; font-weight: bold; background: linear-gradient(90deg, #ff416c, #ff4b2b); color: white; border: none; }
        .stButton button:hover { opacity: 0.9; transform: scale(1.02); transition: 0.2s; }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚡ ViewMax Studio — Professional Mastering Engine")
    st.caption("DSP Audio Mastering • Kinetic Pop Typography • Local AI Word Alignment • Zero/Pro API Configurations.")

    config = build_sidebar_ui()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📦 1. Ingest Video Asset")
        input_type = st.radio("Pipeline Source:", ["YouTube / Shorts URL", "Upload Local File"], horizontal=True, label_visibility="collapsed")

        url_input = ""
        uploaded_file = None
        if input_type == "YouTube / Shorts URL":
            url_input = st.text_input("Source URL:", placeholder="https://youtube.com/shorts/...")
        else:
            uploaded_file = st.file_uploader("Upload Raw Media", type=["mp4", "mov"])

        st.subheader("📜 2. Synthesize Narrative")
        st.session_state.script_input = st.text_area(
            "Script Payload (NLP prosody injection enabled):",
            value=st.session_state.script_input,
            height=140
        )

    with col2:
        st.subheader("🧠 3. Interactive Copilot")
        chat_container = st.container(height=290, border=True)
        
        with chat_container:
            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).markdown(msg["content"])

        user_chat = st.chat_input("Command AI Copilot...")
        if user_chat:
            st.session_state.chat_history.append({"role": "user", "content": user_chat})
            reply = CopilotAgent.process_user_chat_command(user_chat)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    st.divider()

    # ==============================================================================
    # 11. PRIMARY RENDER EXECUTION THREAD
    # ==============================================================================
    if st.button("🚀 INITIATE PROFESSIONAL RENDER PIPELINE"):
        if input_type == "YouTube / Shorts URL" and not url_input.strip():
            st.error("Terminal Error: Valid URL payload required.")
        elif input_type == "Upload Local File" and not uploaded_file:
            st.error("Terminal Error: Local media payload required.")
        elif not st.session_state.script_input.strip():
            st.error("Terminal Error: Narrative text payload empty.")
        elif config.use_elevenlabs and not config.elevenlabs_key:
            st.error("Terminal Error: ElevenLabs API Key missing in configuration.")
        else:
            status = st.empty()
            prog_bar = st.progress(0)

            with tempfile.TemporaryDirectory() as temp_dir:
                source_path = os.path.join(temp_dir, "ingested_raw.mp4")

                try:
                    # [STEP 1: INGESTION]
                    status.info("1/7 Running multi-node ingestion engine...")
                    if input_type == "YouTube / Shorts URL":
                        source_path = IngestionEngine.download_media_from_url(url_input.strip(), temp_dir)
                    else:
                        with open(source_path, "wb") as f:
                            f.write(uploaded_file.read())
                    prog_bar.progress(15)

                    # [STEP 2: AUDIO/VIDEO SPATIAL ANALYSIS]
                    status.info("2/7 Performing DSP RMS Action/Climax Analysis...")
                    raw_clip_initial = VideoFileClip(source_path)
                    full_video_dur = float(raw_clip_initial.duration)
                    peak_action_t = VideoAnalyzer.detect_action_climax_timestamp(source_path)
                    raw_clip_initial.close()
                    prog_bar.progress(30)

                    # [STEP 3: SYNTHESIS & DSP MASTERING]
                    if config.use_elevenlabs:
                        status.info("3/7 Requesting ElevenLabs Premium Synthesis...")
                    else:
                        status.info("3/7 Generating Neural SSML & Applying DSP Broadcast Mastering...")
                    
                    speech_engine = SpeechSynthesisEngine(config.use_elevenlabs, config.elevenlabs_key, config.elevenlabs_voice_id)
                    base_audio_path = os.path.join(temp_dir, "synth_output.wav")
                    
                    final_audio_path = speech_engine.build_voiceover(
                        text=st.session_state.script_input,
                        voice_key=config.voice_actor,
                        output_path=base_audio_path,
                        enable_mastering=config.mastering_enabled
                    )

                    ai_audio = AudioFileClip(final_audio_path)
                    speech_dur = max(1.5, float(ai_audio.duration))
                    prog_bar.progress(45)

                    # [STEP 4: TIMELINE CALCULATIONS]
                    status.info(f"4/7 Assembling timeline structure (Rule: {config.trim_mode})...")
                    start_t, end_t = VideoAnalyzer.calculate_smart_clip_range(
                        full_video_dur, speech_dur, peak_action_t, config.trim_mode
                    )

                    raw_clip = VideoFileClip(source_path).without_audio()
                    trimmed_raw = VideoClipManager.subclip(raw_clip, start_t, end_t)

                    # B-Roll/Looping fallback mechanism if video is too short
                    if trimmed_raw.duration < speech_dur:
                        logger.info("Video asset shorter than speech. Engaging loop matrix.")
                        loops = int(math.ceil(speech_dur / max(0.1, trimmed_raw.duration)))
                        synced_video = VideoClipManager.subclip(concatenate_videoclips([trimmed_raw] * loops), 0, speech_dur)
                    else:
                        synced_video = VideoClipManager.subclip(trimmed_raw, 0, speech_dur)

                    synced_video = VideoClipManager.set_duration(synced_video, speech_dur)
                    prog_bar.progress(60)

                    # [STEP 5: WHISPER ALIGNMENT]
                    status.info("5/7 Executing CPU-bound Whisper Int8 precise timestamp mapping...")
                    word_timestamps = TranscriptionEngine.extract_word_timestamps(final_audio_path, model_size="base")
                    prog_bar.progress(75)

                    # [STEP 6: VISUAL COMPOSITION]
                    status.info("6/7 Reframing canvas matrix and applying Gaussian background blur...")
                    reframed_video = VideoClipManager.apply_transform(
                        synced_video,
                        lambda frame, t: VisualRenderingEngine.reframe_canvas(frame, target_aspect=config.aspect_ratio)
                    )
                    prog_bar.progress(85)

                    # [STEP 7: KINETIC TYPOGRAPHY & FLUSH]
                    status.info("7/7 Rendering Kinetic Bezier Captions and encoding final MP4 multiplex...")
                    final_captioned = VideoClipManager.apply_transform(
                        reframed_video,
                        lambda frame, t: VisualRenderingEngine.render_kinetic_captions(
                            frame, t, word_timestamps, config.caption_preset, pos_y_pct=config.caption_y_pct
                        )
                    )

                    output_clip = VideoClipManager.set_audio(final_captioned, ai_audio)
                    output_clip = VideoClipManager.set_duration(output_clip, speech_dur)
                    export_path = os.path.join(temp_dir, "viewmax_pro_master.mp4")

                    output_clip.write_videofile(
                        export_path,
                        fps=24, # Cinematic framerate
                        codec="libx264",
                        audio_codec="aac",
                        preset="fast",
                        threads=4,
                        logger=None
                    )

                    prog_bar.progress(100)
                    status.success("🟢 PIPELINE COMPLETE. Asset ready for deployment.")

                    with open(export_path, "rb") as f:
                        rendered_bytes = f.read()

                    st.subheader("🎬 Final Mastered Output")
                    st.video(rendered_bytes)
                    
                    st.download_button(
                        label="📥 DOWNLOAD HD MASTER",
                        data=rendered_bytes,
                        file_name="viewmax_pro_master.mp4",
                        mime="video/mp4",
                        type="primary"
                    )

                    # Cleanup Memory Allocation
                    raw_clip.close()
                    ai_audio.close()
                    output_clip.close()

                except Exception as err:
                    prog_bar.progress(0)
                    status.error(f"🔴 CRITICAL PIPELINE FAILURE: {str(err)}")
                    logger.error(f"Render crashed: {traceback.format_exc()}")
                    with st.expander("🔍 View Core Dump / Stack Trace"):
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
