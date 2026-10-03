import os
import requests
import subprocess
import logging
import openai

logger = logging.getLogger("ViewMaxPro.Voice")

class HumanVoiceEngine:
    def __init__(self, elevenlabs_key: str = "", openai_key: str = ""):
        self.elevenlabs_key = elevenlabs_key
        self.openai_client = openai.OpenAI(api_key=openai_key) if openai_key else None

    def synthesize_elevenlabs(self, text: str, voice_id: str, output_path: str):
        if not self.elevenlabs_key:
            raise ValueError("ElevenLabs API key is missing.")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_key
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "style": 0.20,
                "use_speaker_boost": True
            }
        }
        res = requests.post(url, json=data, headers=headers)
        if res.status_code != 200:
            raise RuntimeError(f"ElevenLabs API Error: {res.text}")
            
        with open(output_path, 'wb') as f:
            f.write(res.content)

    def synthesize_openai_hd(self, text: str, voice: str, output_path: str):
        if not self.openai_client:
            raise ValueError("OpenAI API key is missing.")
            
        response = self.openai_client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            response_format="mp3"
        )
        response.stream_to_file(output_path)

    def generate(self, engine: str, text: str, voice_id: str, ffmpeg_bin: str, output_path: str) -> str:
        raw_mp3 = output_path.replace(".wav", "_raw.mp3")
        
        if engine == "ElevenLabs":
            self.synthesize_elevenlabs(text, voice_id, raw_mp3)
        elif engine == "OpenAI_HD":
            self.synthesize_openai_hd(text, voice_id, raw_mp3)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        subprocess.run([
            ffmpeg_bin, "-y", "-i", raw_mp3, "-ac", "1", "-ar", "44100", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return output_path
