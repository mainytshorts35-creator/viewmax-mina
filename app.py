import streamlit as st
import os
import sys
import re
import math
import tempfile
import traceback
import subprocess
import requests
import numpy as np
import yt_dlp
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from gtts import gTTS
from faster_whisper import WhisperModel

# ==========================================
# 1. MOVIEPY UNIVERSAL COMPATIBILITY LAYER
# ==========================================
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    IS_LEGACY_MOVIEPY = True
except (ImportError, ModuleNotFoundError):
    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
        IS_LEGACY_MOVIEPY = False
    except Exception as e:
        st.error(f"Critical Error loading MoviePy: {str(e)}")
        st.stop()

def safe_set_audio(clip, audio_clip):
    return clip.set_audio(audio_clip) if IS_LEGACY_MOVIEPY else clip.with_audio(audio_clip)

def safe_subclip(clip, start_time, end_time):
    return clip.subclip(start_time, end_time) if IS_LEGACY_MOVIEPY else clip.subclipped(start_time, end_time)

def safe_set_duration(clip, duration):
    return clip.set_duration(duration) if IS_LEGACY_MOVIEPY else clip.with_duration(duration)

def safe_apply_transform(clip, transform_fn):
    return clip.fl(lambda gf, t: transform_fn(gf(t), t)) if IS_LEGACY_MOVIEPY else clip.transform(lambda gf, t: transform_fn(gf(t), t))

# ==========================================
# 2. SYSTEM BINARIES & AUDIO PADDING ENGINE
# ==========================================
def get_ffmpeg_binary():
    try:
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return "ffmpeg"

def pad_audio_file_with_silence(input_path, output_path, pad_seconds=3.0):
    """
    Pads trailing silence to the audio file using FFmpeg.
    Prevents MoviePy duration mismatch/indexing crashes during rendering.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-af", f"apad=pad_dur={pad_seconds}",
        "-c:a", "pcm_s16le",
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except Exception:
        return input_path

# ==========================================
# 3. LOCAL AI ENGINE & PYAV SAFE TRANSCRIPTION
# ==========================================
@st.cache_resource
def load_local_whisper_model(model_size="base"):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def extract_ai_word_timestamps_safe(audio_path, model_size="base"):
    model = load_local_whisper_model(model_size)
    
    try:
        segments, _ = model.transcribe(audio_path, word_timestamps=True, language="en")
        segments = list(segments)
    except TypeError as err:
        if "metadata_errors" in str(err):
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
    return word_timestamps

# ==========================================
# 4. ACTION & CLIMAX DETECTION ENGINE
# ==========================================
def detect_action_climax_timestamp(video_path):
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None or clip.duration <= 2.0:
            return clip.duration / 2.0
        
        fps_sample = 22050
        audio_array = clip.audio.to_soundarray(fps=fps_sample)
        clip.close()

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        window_size = fps_sample
        num_windows = len(audio_array) // window_size
        if num_windows == 0:
            return clip.duration / 2.0

        energies = [np.sum(audio_array[i * window_size : (i + 1) * window_size] ** 2) for i in range(num_windows)]
        peak_window_idx = int(np.argmax(energies))
        
        return float(peak_window_idx)
    except Exception:
        return 0.0

def calculate_smart_clip_range(video_duration, speech_duration, peak_timestamp, mode="Smart Climax Retention"):
    if mode == "Keep Full Video (Fit Speech)":
        return 0.0, video_duration

    if speech_duration >= video_duration:
        return 0.0, video_duration

    if mode == "Smart Climax Retention":
        half_speech = speech_duration / 2.0
        start_t = max(0.0, peak_timestamp - half_speech)
        end_t = start_t + speech_duration

        if end_t > video_duration:
            end_t = video_duration
            start_t = max(0.0, end_t - speech_duration)

        return start_t, end_t

    return 0.0, min(speech_duration, video_duration)

# ==========================================
# 5. INGESTION PIPELINE
# ==========================================
def download_media_from_url(url, target_directory):
    ffmpeg_bin = get_ffmpeg_binary()
    output_path = os.path.join(target_directory, 'downloaded_source.mp4')

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
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
                'Accept-Language': 'en-US,en;q=0.9',
            },
            **strat
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    return filename
                for f in os.listdir(target_directory):
                    fp = os.path.join(target_directory, f)
                    if os.path.getsize(fp) > 0 and not f.endswith('.part'):
                        return fp
        except Exception as e:
            last_err = e
            continue

    try:
        resp = requests.post(
            "https://api.cobalt.tools/api/json",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json={"url": url, "videoQuality": "720"},
            timeout=12
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
                        return output_path
    except Exception:
        pass

    if last_err and "403" in str(last_err):
        raise RuntimeError("Cloud IP blocked by YouTube (HTTP 403). Switch to 'Upload Video File' or run locally.")
    raise RuntimeError(f"Ingestion Engine Error: {str(last_err)}")

# ==========================================
# 6. SPEECH SYNTHESIS & REFRAMING ENGINE
# ==========================================
def sanitize_script(text):
    if not text or not text.strip():
        return ""
    clean = re.sub(r'[^\w\s\.,!\?\'\-]', '', text)
    return re.sub(r'\s+', ' ', clean).strip()

def build_voiceover(text, accent_key, output_path):
    clean_text = sanitize_script(text)
    if not clean_text:
        raise ValueError("Script text is empty.")

    tld_map = {"US": "com", "UK": "co.uk", "Australia": "com.au", "India": "co.in", "Canada": "ca"}
    selected_tld = tld_map.get(accent_key, "com")

    tts = gTTS(text=clean_text, lang="en", tld=selected_tld, slow=False)
    tts.save(output_path)
    return output_path

def reframe_canvas(frame_array, target_aspect="9:16 Shorts/Reels"):
    img = Image.fromarray(frame_array).convert("RGB")
    src_w, src_h = img.size

    aspect_ratios = {
        "9:16 Shorts/Reels": (1080, 1920),
        "1:1 Square": (1080, 1080),
        "16:9 Landscape": (1920, 1080)
    }
    target_w, target_h = aspect_ratios.get(target_aspect, (1080, 1920))

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    scaled_fg = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    bg_scale = max(target_w / src_w, target_h / src_h)
    bg_w = max(1, int(src_w * bg_scale))
    bg_h = max(1, int(src_h * bg_scale))
    background = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
    
    crop_x = (bg_w - target_w) // 2
    crop_y = (bg_h - target_h) // 2
    background = background.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    background = background.filter(ImageFilter.GaussianBlur(radius=25))

    pos_x = (target_w - new_w) // 2
    pos_y = (target_h - new_h) // 2
    background.paste(scaled_fg, (pos_x, pos_y))

    return np.array(background)

# ==========================================
# 7. DYNAMIC TIMED CAPTION RENDERER
# ==========================================
def load_font(height, size_factor=0.045):
    font_size = max(int(height * size_factor), 18)
    for f in ["DejaVuSans-Bold.ttf", "Arial-Bold.ttf", "arial.ttf", "Helvetica-Bold.ttf"]:
        try:
            return ImageFont.truetype(f, font_size)
        except Exception:
            continue
    return ImageFont.load_default()

def render_ai_timed_captions(frame_array, current_time, word_timestamps, preset, pos_y_pct=0.75):
    img = Image.fromarray(frame_array).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if not word_timestamps:
        return np.array(img)

    active_idx = 0
    for i, item in enumerate(word_timestamps):
        if current_time >= item["start"]:
            active_idx = i
        if item["start"] <= current_time <= item["end"]:
            active_idx = i
            break

    chunk_size = 3
    chunk_start = (active_idx // chunk_size) * chunk_size
    chunk_words = word_timestamps[chunk_start : chunk_start + chunk_size]

    font = load_font(h)
    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    word_widths = [draw.textbbox((0, 0), item["word"], font=font)[2] for item in chunk_words]
    total_phrase_w = sum(word_widths) + space_w * (len(chunk_words) - 1)
    
    try:
        sample_h = draw.textbbox((0, 0), "Hg", font=font)[3]
    except Exception:
        sample_h = 30

    start_x = (w - total_phrase_w) // 2
    start_y = int(h * pos_y_pct)

    if "Hormozi" in preset:
        active_color = (255, 230, 0)
        inactive_color = (255, 255, 255)
        bg_fill = (0, 0, 0)
        border_col = (0, 0, 0)
    elif "Cyberpunk" in preset:
        active_color = (0, 242, 254)
        inactive_color = (180, 180, 200)
        bg_fill = (15, 15, 30)
        border_col = (255, 0, 128)
    else:
        active_color = (255, 255, 255)
        inactive_color = (220, 220, 220)
        bg_fill = (220, 20, 60)
        border_col = (0, 0, 0)

    pad_x, pad_y = 20, 12
    draw.rectangle(
        [start_x - pad_x, start_y - pad_y, start_x + total_phrase_w + pad_x, start_y + sample_h + pad_y],
        fill=bg_fill,
        outline=border_col if "Cyberpunk" in preset else None,
        width=2
    )

    curr_x = start_x
    for i, item in enumerate(chunk_words):
        global_idx = chunk_start + i
        is_active = (global_idx == active_idx)
        color = active_color if is_active else inactive_color

        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 2), (0, -2)]:
            draw.text((curr_x + dx, start_y + dy), item["word"], font=font, fill=(0, 0, 0))

        draw.text((curr_x, start_y), item["word"], font=font, fill=color)
        curr_x += word_widths[i] + space_w

    return np.array(img)

# ==========================================
# 8. INTERACTIVE AI COPILOT HANDLER
# ==========================================
def process_user_chat_command(user_command):
    cmd = user_command.lower()
    changes = []

    if any(k in cmd for k in ["full video", "dont cut", "don't cut", "keep whole", "entire"]):
        st.session_state.trim_mode = "Keep Full Video (Fit Speech)"
        changes.append("Set strategy to **Keep Full Video Duration**.")

    if any(k in cmd for k in ["ko", "knockout", "climax", "hit", "action"]):
        st.session_state.trim_mode = "Smart Climax Retention"
        changes.append("Activated **Smart Climax Detection**.")

    if "cyberpunk" in cmd or "neon" in cmd:
        st.session_state.caption_preset = "Cyberpunk Glow"
        changes.append("Changed caption style to **Cyberpunk Glow**.")
    elif "red" in cmd or "impact" in cmd:
        st.session_state.caption_preset = "Impact Red Banner"
        changes.append("Changed caption style to **Impact Red Banner**.")
    elif "hormozi" in cmd or "yellow" in cmd:
        st.session_state.caption_preset = "Hormozi Active Highlight"
        changes.append("Changed caption style to **Hormozi Active Highlight**.")

    if "script:" in cmd:
        new_script = user_command.split("script:", 1)[1].strip()
        if new_script:
            st.session_state.script_input = new_script
            changes.append(f"Updated voiceover script to: *\"{new_script}\"*")
    elif any(k in cmd for k in ["write", "hook", "rewrite", "make it punchy"]):
        generated_script = f"Watch this incredible moment! {user_command.strip().capitalize()}"
        st.session_state.script_input = generated_script
        changes.append(f"Generated new viral script: *\"{generated_script}\"*")

    if not changes:
        return "I heard you! You can tell me to: *'Keep full video'*, *'Find KO moment'*, *'Change preset to Cyberpunk'*, or *'Script: [your text]'*."

    return "✅ " + " ".join(changes)

# ==========================================
# 9. STREAMLIT APPLICATION & UI
# ==========================================
st.set_page_config(page_title="ViewMax AI Studio - Intelligent Engine", page_icon="⚡", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI Video Copilot. Tell me what changes to make (e.g., 'Keep full video', 'Find KO moment', 'Change preset to Cyberpunk', or 'Script: [your text]')."}
    ]

if "trim_mode" not in st.session_state:
    st.session_state.trim_mode = "Smart Climax Retention"
if "caption_preset" not in st.session_state:
    st.session_state.caption_preset = "Hormozi Active Highlight"
if "script_input" not in st.session_state:
    st.session_state.script_input = "Watch this incredible knockout moment unfold right before your eyes. Unbelievable precision and power!"

st.title("⚡ ViewMax Studio — Intelligent Local AI Engine")
st.caption("Local Whisper Word Timing + Audio Peak KO Detection + Zero API Key AI Chat Copilot.")

st.sidebar.title("🎛 AI Pipeline Controls")

trim_options = ["Smart Climax Retention", "Keep Full Video (Fit Speech)", "Standard Start Cut"]
st.session_state.trim_mode = st.sidebar.radio(
    "Clip Trimming Strategy:",
    trim_options,
    index=trim_options.index(st.session_state.trim_mode) if st.session_state.trim_mode in trim_options else 0
)

preset_options = ["Hormozi Active Highlight", "Cyberpunk Glow", "Impact Red Banner"]
st.session_state.caption_preset = st.sidebar.selectbox(
    "Caption Preset Style:",
    preset_options,
    index=preset_options.index(st.session_state.caption_preset) if st.session_state.caption_preset in preset_options else 0
)

selected_accent = st.sidebar.selectbox("AI Voice Accent", ["US", "UK", "Australia", "India", "Canada"])
canvas_aspect = st.sidebar.selectbox("Canvas Aspect Ratio", ["9:16 Shorts/Reels", "1:1 Square", "16:9 Landscape"])
caption_y_pos = st.sidebar.slider("Caption Vertical Alignment", 0.50, 0.85, 0.75, 0.02)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Source Video Asset")
    input_type = st.radio("Source Type:", ["YouTube / Shorts URL", "Upload Video File"], horizontal=True)

    url_input = ""
    uploaded_file = None

    if input_type == "YouTube / Shorts URL":
        url_input = st.text_input("YouTube Shorts URL:", placeholder="https://www.youtube.com/shorts/...")
    else:
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi"])

    st.subheader("2. AI Voiceover Script")
    st.session_state.script_input = st.text_area(
        "Script Text (Editable or generated via AI chat):",
        value=st.session_state.script_input,
        height=120
    )

with col2:
    st.subheader("💬 AI Copilot Assistant")
    chat_container = st.container(height=260)
    
    with chat_container:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])

    user_chat = st.chat_input("Tell AI: 'Keep full video', 'Change preset to Cyberpunk'...")
    if user_chat:
        st.session_state.chat_history.append({"role": "user", "content": user_chat})
        reply = process_user_chat_command(user_chat)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

st.divider()

# ==========================================
# 10. RENDERING EXECUTION ENGINE
# ==========================================
if st.button("🚀 Render Video with AI Timing & Action Detection"):
    cleaned = sanitize_script(st.session_state.script_input)
    
    if input_type == "YouTube / Shorts URL" and not url_input.strip():
        st.error("Please enter a YouTube video URL.")
    elif input_type == "Upload Video File" and not uploaded_file:
        st.error("Please upload a video file.")
    elif not cleaned:
        st.error("Script text cannot be empty.")
    else:
        status_box = st.empty()
        progress = st.progress(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "raw_source.mp4")

            try:
                # 1. Ingest Asset
                status_box.info("1/7 Ingesting video asset...")
                if input_type == "YouTube / Shorts URL":
                    source_path = download_media_from_url(url_input.strip(), temp_dir)
                else:
                    with open(source_path, "wb") as f:
                        f.write(uploaded_file.read())
                progress.progress(15)

                # 2. Peak Audio Action / KO Climax Analysis
                status_box.info("2/7 Scanning audio waveform for action climax peak...")
                raw_clip_initial = VideoFileClip(source_path)
                full_video_duration = float(raw_clip_initial.duration)
                peak_action_t = detect_action_climax_timestamp(source_path)
                raw_clip_initial.close()
                progress.progress(30)

                # 3. Speech Audio Synthesis with Tail Padding
                status_box.info("3/7 Synthesizing voiceover audio track with zero-drop padding...")
                raw_audio_path = os.path.join(temp_dir, "raw_voice.mp3")
                padded_audio_path = os.path.join(temp_dir, "padded_voice.wav")

                build_voiceover(cleaned, selected_accent, raw_audio_path)

                temp_speech = AudioFileClip(raw_audio_path)
                raw_speech_dur = float(temp_speech.duration)
                temp_speech.close()

                # Ensure a minimum 1.5s duration threshold
                speech_dur = max(1.5, raw_speech_dur)

                # Append 3 seconds of silent audio padding to protect against MoviePy time crashes
                pad_audio_file_with_silence(raw_audio_path, padded_audio_path, pad_seconds=3.0)
                ai_audio = AudioFileClip(padded_audio_path)
                progress.progress(45)

                # 4. Smart Subclip Calculation
                status_box.info(f"4/7 Smart clipping (Strategy: {st.session_state.trim_mode})...")
                start_t, end_t = calculate_smart_clip_range(
                    full_video_duration, speech_dur, peak_action_t, mode=st.session_state.trim_mode
                )

                raw_clip = VideoFileClip(source_path).without_audio()
                trimmed_raw = safe_subclip(raw_clip, start_t, end_t)

                if trimmed_raw.duration < speech_dur:
                    loops = int(math.ceil(speech_dur / max(0.1, trimmed_raw.duration)))
                    synced_video = safe_subclip(concatenate_videoclips([trimmed_raw] * loops), 0, speech_dur)
                else:
                    synced_video = safe_subclip(trimmed_raw, 0, speech_dur)

                synced_video = safe_set_duration(synced_video, speech_dur)
                progress.progress(60)

                # 5. Local Safe Whisper Word Alignment
                status_box.info("5/7 Local Whisper AI aligning word timestamps...")
                word_timestamps = extract_ai_word_timestamps_safe(raw_audio_path, model_size="base")
                progress.progress(75)

                # 6. Canvas Reframing
                status_box.info("6/7 Reframing canvas with blurred background...")
                reframed_video = safe_apply_transform(
                    synced_video,
                    lambda frame, t: reframe_canvas(frame, target_aspect=canvas_aspect)
                )
                progress.progress(85)

                # 7. Render Timed Captions & Export
                status_box.info("7/7 Rendering dynamic captions and encoding MP4...")
                final_captioned = safe_apply_transform(
                    reframed_video,
                    lambda frame, t: render_ai_timed_captions(
                        frame, t, word_timestamps, st.session_state.caption_preset, pos_y_pct=caption_y_pos
                    )
                )

                output_clip = safe_set_audio(final_captioned, ai_audio)
                output_clip = safe_set_duration(output_clip, speech_dur)
                export_path = os.path.join(temp_dir, "viewmax_intelligent.mp4")

                output_clip.write_videofile(
                    export_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    logger=None
                )

                progress.progress(100)
                status_box.success("🎉 Video rendering complete!")

                with open(export_path, "rb") as f:
                    rendered_bytes = f.read()

                st.subheader("🎬 Final Output Video")
                st.video(rendered_bytes)
                st.download_button(
                    label="📥 Download Repurposed MP4",
                    data=rendered_bytes,
                    file_name="viewmax_intelligent.mp4",
                    mime="video/mp4"
                )

                raw_clip.close()
                ai_audio.close()

            except Exception as err:
                progress.progress(0)
                status_box.error(f"Execution Error: {str(err)}")
                with st.expander("🔍 Stack Trace"):
                    st.code(traceback.format_exc())
