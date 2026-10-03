import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import logging

logger = logging.getLogger("ViewMaxPro.DSP")

class AudioMasteringRack:
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def apply_highpass(self, audio: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        nyq = self.sr / 2.0
        b, a = signal.butter(4, cutoff / nyq, btype='high')
        return signal.lfilter(b, a, audio)

    def apply_parametric_eq(self, audio: np.ndarray) -> np.ndarray:
        nyq = self.sr / 2.0
        # Presence boost around 3kHz for vocal punch
        b_pres, a_pres = signal.iirpeak(3000.0 / nyq, 1.5)
        presence = signal.lfilter(b_pres, a_pres, audio)
        return audio + (presence * 0.35)

    def apply_rms_compressor(self, audio: np.ndarray, thresh_db: float = -16.0, 
                             ratio: float = 4.0, attack_ms: float = 5.0, 
                             release_ms: float = 50.0, makeup_db: float = 3.0) -> np.ndarray:
        audio_f = audio.astype(np.float64) / 32768.0
        attack_c = np.exp(-1.0 / (self.sr * (attack_ms / 1000.0)))
        release_c = np.exp(-1.0 / (self.sr * (release_ms / 1000.0)))
        
        env = np.zeros_like(audio_f)
        state = 0.0
        for i in range(len(audio_f)):
            sq = audio_f[i] ** 2
            state = attack_c * state + (1 - attack_c) * sq if sq > state else release_c * state + (1 - release_c) * sq
            env[i] = max(state, 1e-10)
            
        env_db = 10.0 * np.log10(env)
        gain_db = np.where(env_db > thresh_db, -(env_db - thresh_db) * (1.0 - 1.0 / ratio), 0.0)
        gain_lin = 10.0 ** ((gain_db + makeup_db) / 20.0)
        
        return np.clip(audio_f * gain_lin * 32767.0, -32768, 32767).astype(np.int16)

    def process_vocal_chain(self, input_path: str, output_path: str):
        rate, data = wavfile.read(input_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        filtered = self.apply_highpass(data)
        eq_data = self.apply_parametric_eq(filtered)
        comp_data = self.apply_rms_compressor(eq_data)
        
        wavfile.write(output_path, rate, comp_data)
        logger.info("Audio mastering complete.")
        return output_path
