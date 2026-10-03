"""
================================================================================
VIEWMAX STUDIO PRO — BROADCAST AUDIO SIGNAL PROCESSING (DSP) ENGINE
FILE: audio_dsp.py
DESCRIPTION: High-precision audio processing pipeline providing vocal equalization,
             multiband highpass/lowpass filtering, dynamic range RMS compression,
             de-essing, lookahead peak limiting, and sidechain BGM ducking.
================================================================================
"""

import os
import sys
import math
import logging
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union, List

from config import AudioProcessingSettings, get_logger

logger = get_logger("ViewMaxPro.AudioDSP")


# ==============================================================================
# 1. CUSTOM EXCEPTIONS & DATA MODELS
# ==============================================================================
class AudioDSPError(Exception):
    """Base exception class for Audio DSP operations."""
    pass


class AudioProcessingError(AudioDSPError):
    """Raised when signal transformation encounters numerical or format errors."""
    pass


class InvalidAudioFormatError(AudioDSPError):
    """Raised when loaded audio data is corrupt, empty, or unreadable."""
    pass


@dataclass
class AudioMetrics:
    """Quantitative measurement metrics for an audio signal clip."""
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    duration_seconds: float
    sample_rate: int
    channels: int
    is_clipping: bool


@dataclass
class DSPChainSummary:
    """Summary diagnostic report generated after executing a DSP mastering run."""
    input_file: str
    output_file: str
    input_metrics: AudioMetrics
    output_metrics: AudioMetrics
    processing_steps_applied: List[str]


# ==============================================================================
# 2. SIGNAL MATH & CONVERSION UTILITIES
# ==============================================================================
def db_to_linear(db_val: float) -> float:
    """Converts a decibel value to a linear amplitude scale factor."""
    return 10.0 ** (db_val / 20.0)


def linear_to_db(lin_val: float, eps: float = 1e-10) -> float:
    """Converts a linear amplitude value to dB scale with epsilon safeguard."""
    return 20.0 * np.log10(np.maximum(np.abs(lin_val), eps))


def calculate_audio_metrics(audio: np.ndarray, rate: int) -> AudioMetrics:
    """
    Computes RMS level, peak level, crest factor, and clipping flags 
    for an audio sample array.
    """
    if len(audio) == 0:
        raise InvalidAudioFormatError("Audio array is empty.")

    # Normalize array to float range [-1.0, 1.0] for calculation
    if audio.dtype == np.int16:
        float_samples = audio.astype(np.float64) / 32768.0
    elif audio.dtype == np.int32:
        float_samples = audio.astype(np.float64) / 2147483648.0
    else:
        float_samples = audio.astype(np.float64)

    channels = 1 if len(float_samples.shape) == 1 else float_samples.shape[1]
    
    # Calculate peak and RMS across all channels
    max_peak = np.max(np.abs(float_samples))
    rms_val = np.sqrt(np.mean(float_samples ** 2))
    
    peak_dbfs = float(linear_to_db(max_peak))
    rms_dbfs = float(linear_to_db(rms_val))
    crest_factor = peak_dbfs - rms_dbfs
    duration = float(len(float_samples) / rate) if channels == 1 else float(float_samples.shape[0] / rate)
    is_clipping = max_peak >= 0.999

    return AudioMetrics(
        peak_dbfs=round(peak_dbfs, 2),
        rms_dbfs=round(rms_dbfs, 2),
        crest_factor_db=round(crest_factor, 2),
        duration_seconds=round(duration, 2),
        sample_rate=rate,
        channels=channels,
        is_clipping=is_clipping
    )


# ==============================================================================
# 3. VOCAL MASTERING DSP PROCESSOR
# ==============================================================================
class VocalMasteringProcessor:
    """
    Broadcast-grade vocal mastering suite providing highpass cleanup, 
    presence parametric EQ, de-essing, dynamic range compression, 
    and peak limiting.
    """

    def __init__(self, settings: Optional[AudioProcessingSettings] = None):
        self.cfg = settings or AudioProcessingSettings()

    def load_wav(self, file_path: str) -> Tuple[int, np.ndarray]:
        """Loads a WAV file, converts stereo to mono, and standardizes to float64."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            rate, data = wavfile.read(file_path)
        except Exception as e:
            raise InvalidAudioFormatError(f"Failed to read WAV file: {e}")

        if data.size == 0:
            raise InvalidAudioFormatError(f"Audio file '{file_path}' contains no audio samples.")

        # Convert multi-channel audio to mono by averaging channels
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Standardize sample representation to 64-bit float [-32768.0, 32767.0]
        if data.dtype == np.int16:
            float_data = data.astype(np.float64)
        elif data.dtype == np.int32:
            float_data = (data / 65536.0).astype(np.float64)
        elif data.dtype in [np.float32, np.float64]:
            float_data = data * 32768.0
        else:
            float_data = data.astype(np.float64)

        return rate, float_data

    def save_wav(self, file_path: str, rate: int, data: np.ndarray) -> str:
        """Clips float64 audio array safely into 16-bit PCM WAV format and saves."""
        clipped = np.clip(data, -32768.0, 32767.0).astype(np.int16)
        wavfile.write(file_path, rate, clipped)
        logger.info(f"Saved processed audio to: {file_path}")
        return file_path

    def apply_highpass_filter(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """Removes low-frequency rumble below configured cutoff using a 4th-order Butterworth filter."""
        nyquist = rate / 2.0
        if self.cfg.highpass_cutoff_hz >= nyquist:
            return audio

        norm_cutoff = self.cfg.highpass_cutoff_hz / nyquist
        b, a = signal.butter(4, norm_cutoff, btype='high', analog=False)
        filtered = signal.lfilter(b, a, audio)
        return filtered

    def apply_lowpass_filter(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """Removes harsh high-frequency noise above configured cutoff."""
        nyquist = rate / 2.0
        if self.cfg.lowpass_cutoff_hz >= nyquist:
            return audio

        norm_cutoff = self.cfg.lowpass_cutoff_hz / nyquist
        b, a = signal.butter(4, norm_cutoff, btype='low', analog=False)
        filtered = signal.lfilter(b, a, audio)
        return filtered

    def apply_presence_eq(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """Injects presence boost around speech vocal clarity frequencies (e.g., 3.2kHz)."""
        nyquist = rate / 2.0
        norm_freq = self.cfg.presence_boost_freq_hz / nyquist

        if norm_freq >= 1.0 or norm_freq <= 0.0:
            return audio

        # Construct second-order peaking filter
        b, a = signal.iirpeak(norm_freq, Q=self.cfg.presence_boost_q)
        peaking_component = signal.lfilter(b, a, audio)
        
        # Blend boosted component back into original signal
        boost_factor = self.cfg.presence_boost_gain - 1.0
        return audio + (peaking_component * boost_factor)

    def apply_deesser(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """Attenuates sharp sibilant frequencies (5kHz - 8kHz) dynamically."""
        nyquist = rate / 2.0
        low_f, high_f = 5000.0 / nyquist, 8000.0 / nyquist

        if high_f >= 1.0:
            return audio

        # Bandpass filter for sibilance detection
        b, a = signal.butter(2, [low_f, high_f], btype='band')
        sibilance_signal = signal.lfilter(b, a, audio)

        # Smooth envelope follower
        env = np.abs(sibilance_signal) / 32768.0
        smooth_len = max(1, int(rate * 0.005))  # 5ms smoothing
        kernel = np.ones(smooth_len) / smooth_len
        env_smoothed = np.convolve(env, kernel, mode='same')

        # Dynamic notch reduction keying
        threshold = 0.15
        attenuation = np.where(env_smoothed > threshold, 0.70, 1.0)

        return audio * attenuation

    def apply_rms_compressor(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """
        Executes feedback dynamic range compression using attack/release coefficients 
        and soft-knee ratio calculations.
        """
        audio_norm = audio / 32768.0
        num_samples = len(audio_norm)

        attack_samples = max(1, int(rate * (self.cfg.compressor_attack_ms / 1000.0)))
        release_samples = max(1, int(rate * (self.cfg.compressor_release_ms / 1000.0)))

        attack_coeff = np.exp(-1.0 / attack_samples)
        release_coeff = np.exp(-1.0 / release_samples)

        envelope = np.zeros(num_samples, dtype=np.float64)
        current_state = 0.0

        # Ballistic RMS envelope extraction loop
        for i in range(num_samples):
            rectified = audio_norm[i] ** 2
            if rectified > current_state:
                current_state = attack_coeff * current_state + (1.0 - attack_coeff) * rectified
            else:
                current_state = release_coeff * current_state + (1.0 - release_coeff) * rectified
            envelope[i] = max(current_state, 1e-10)

        env_db = 10.0 * np.log10(envelope)
        thresh = self.cfg.compressor_threshold_db
        ratio = self.cfg.compressor_ratio

        # Calculate gain reduction in dB
        gain_reduction_db = np.where(
            env_db > thresh,
            -(env_db - thresh) * (1.0 - (1.0 / ratio)),
            0.0
        )

        # Apply gain reduction plus makeup gain
        total_gain_db = gain_reduction_db + self.cfg.compressor_makeup_db
        linear_gain = 10.0 ** (total_gain_db / 20.0)

        compressed = audio_norm * linear_gain
        return compressed * 32768.0

    def apply_brickwall_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Applies hard ceiling peak limiting to eliminate audio digital clipping."""
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio

        target_peak_linear = (10.0 ** (self.cfg.limiter_ceiling_db / 20.0)) * 32767.0

        if max_val > target_peak_linear:
            scale = target_peak_linear / max_val
            return audio * scale

        return audio

    def apply_fades(self, audio: np.ndarray, rate: int) -> np.ndarray:
        """Applies subtle raised-cosine fade-in and fade-out to remove boundary clicks."""
        num_samples = len(audio)
        fade_in_s = max(1, int(rate * self.cfg.fade_in_duration_sec))
        fade_out_s = max(1, int(rate * self.cfg.fade_out_duration_sec))

        if num_samples <= (fade_in_s + fade_out_s):
            return audio

        output = audio.copy()

        # Raised cosine fade-in
        fade_in_curve = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_in_s)))
        output[:fade_in_s] *= fade_in_curve

        # Raised cosine fade-out
        fade_out_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_out_s)))
        output[-fade_out_s:] *= fade_out_curve

        return output

    def process_voice_chain(self, input_wav_path: str, output_wav_path: str) -> DSPChainSummary:
        """
        Executes the full vocal mastering pipeline in sequence and yields 
        a detailed comparative summary.
        """
        logger.info(f"Initiating vocal DSP chain processing on: {input_wav_path}")
        rate, raw_data = self.load_wav(input_wav_path)
        input_metrics = calculate_audio_metrics(raw_data, rate)

        steps_executed = []

        # Step 1: Highpass rumble filter
        hp_data = self.apply_highpass_filter(raw_data, rate)
        steps_executed.append("Highpass Filter (80Hz)")

        # Step 2: Lowpass filter
        lp_data = self.apply_lowpass_filter(hp_data, rate)
        steps_executed.append("Lowpass Filter (16kHz)")

        # Step 3: Presence EQ
        eq_data = self.apply_presence_eq(lp_data, rate)
        steps_executed.append("Presence Boost EQ (3.2kHz)")

        # Step 4: De-esser
        deess_data = self.apply_deesser(eq_data, rate)
        steps_executed.append("Dynamic De-Esser")

        # Step 5: RMS Compressor with Makeup Gain
        comp_data = self.apply_rms_compressor(deess_data, rate)
        steps_executed.append("RMS Dynamics Compressor")

        # Step 6: Raised-cosine boundary fades
        faded_data = self.apply_fades(comp_data, rate)
        steps_executed.append("Cosine Boundary Fades")

        # Step 7: Brickwall Peak Limiter
        mastered_data = self.apply_brickwall_limiter(faded_data)
        steps_executed.append("Brickwall Limiter (-0.5dB Ceiling)")

        # Save mastered output to disk
        self.save_wav(output_wav_path, rate, mastered_data)
        output_metrics = calculate_audio_metrics(mastered_data, rate)

        logger.info(f"DSP Processing Complete. Peak dBFS: {input_metrics.peak_dbfs} -> {output_metrics.peak_dbfs}")

        return DSPChainSummary(
            input_file=input_wav_path,
            output_file=output_wav_path,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            processing_steps_applied=steps_executed
        )


# ==============================================================================
# 4. SIDECHAIN BGM MUSIC DUCKER & MIXER
# ==============================================================================
class SidechainMusicDucker:
    """
    Blends background music track with vocal stems, automatically ducking BGM 
    when vocal amplitude crosses dynamic threshold limits.
    """

    def __init__(self, settings: Optional[AudioProcessingSettings] = None):
        self.cfg = settings or AudioProcessingSettings()
        self.processor = VocalMasteringProcessor(self.cfg)

    def mix_bgm_with_sidechain(
        self,
        vocal_wav_path: str,
        bgm_wav_path: str,
        output_wav_path: str
    ) -> str:
        """
        Calculates vocal amplitude envelope and applies inverse attenuation curve 
        to BGM before mixing into output WAV.
        """
        logger.info(f"Mixing BGM '{bgm_wav_path}' with vocal stem '{vocal_wav_path}'...")
        
        v_rate, v_data = self.processor.load_wav(vocal_wav_path)
        b_rate, b_data = self.processor.load_wav(bgm_wav_path)

        # Resample BGM to match vocal sample rate if necessary
        if v_rate != b_rate:
            logger.info(f"Resampling BGM rate from {b_rate}Hz to {v_rate}Hz...")
            num_samples = int(len(b_data) * (v_rate / float(b_rate)))
            b_data = signal.resample(b_data, num_samples)

        # Ensure BGM length covers full vocal stem by tiling or clipping
        if len(b_data) < len(v_data):
            repeat_count = int(np.ceil(len(v_data) / float(len(b_data))))
            b_data = np.tile(b_data, repeat_count)
        b_data = b_data[:len(v_data)]

        # Extract normalized smooth vocal envelope for ducking
        v_abs = np.abs(v_data) / 32768.0
        smooth_window = max(1, int(v_rate * 0.12))  # 120ms smoothing window
        kernel = np.ones(smooth_window) / float(smooth_window)
        v_envelope = np.convolve(v_abs, kernel, mode='same')

        # Compute dynamic ducking gain curve
        attenuation_factor = db_to_linear(self.cfg.bgm_ducking_attenuation_db)
        duck_gain = 1.0 - (v_envelope * (1.0 - attenuation_factor))
        duck_gain = np.clip(duck_gain, attenuation_factor, 1.0)

        # Apply base scaling factor and dynamic ducking to BGM
        scaled_bgm = b_data * self.cfg.bgm_base_volume_scale * duck_gain

        # Sum vocal and ducked background music
        mixed_signal = v_data + scaled_bgm

        # Final safety brickwall limit to avoid clipping
        mastered_mix = self.processor.apply_brickwall_limiter(mixed_signal)

        self.processor.save_wav(output_wav_path, v_rate, mastered_mix)
        return output_wav_path


# ==============================================================================
# 5. STANDALONE VERIFICATION / UNIT RUNNER
# ==============================================================================

    
    # Generate synthetic 44.1kHz sine wave vocal test mock
    sample_rate = 44100
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    
    # Synthetic speech tone (440Hz with noise spike)
    synthetic_speech = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.normal(size=len(t))
    synthetic_speech *= 32768.0

    test_dir = os.path.join(os.getcwd(), "viewmax_workspace", "temp")
    os.makedirs(test_dir, exist_ok=True)
    
    test_in = os.path.join(test_dir, "test_input.wav")
    test_out = os.path.join(test_dir, "test_output.wav")

    wavfile.write(test_in, sample_rate, synthetic_speech.astype(np.int16))

    proc = VocalMasteringProcessor()
    summary = proc.process_voice_chain(test_in, test_out)

    print(f"\n--- DSP MASTERING DIAGNOSTIC RESULT ---")
    print(f"Input Peak dBFS : {summary.input_metrics.peak_dbfs}")
    print(f"Output Peak dBFS: {summary.output_metrics.peak_dbfs}")
    print(f"Steps Applied   : {', '.join(summary.processing_steps_applied)}")
    print("----------------------------------------\n")
