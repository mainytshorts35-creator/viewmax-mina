import streamlit as st
import os
import sys
import io
import re
import math
import time
import shutil
import tempfile
import traceback
import numpy as np
import yt_dlp
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# ==========================================
# 1. MOVIEPY COMPATIBILITY & ABSTRACTION LAYER
# ==========================================
# MoviePy underwent a breaking API rewrite between v1.x and v2.x.
# This layer detects the installed version and provides universal wrapper functions.

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    IS_LEGACY_MOVIEPY = True
except (ImportError, ModuleNotFoundError):
    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
        IS_LEGACY_MOVIEPY = False
    except Exception as e:
        st.error(f"Failed to import MoviePy library: {str(e)}")
        st.stop()

def safe_set_audio(clip, audio_clip):
    """Sets audio track on a VideoClip across all MoviePy versions."""
    try:
        if IS_LEGACY_MOVIEPY:
            return clip.set_audio(audio_clip)
        else:
            return clip.with_audio(audio_clip)
    except Exception as e:
        raise RuntimeError(f"Failed to attach audio to video clip: {str(e)}")

def safe_subclip(clip, start_time, end_time):
    """Trims a clip between start_time and end_time safely."""
    try:
        if IS_LEGACY_MOVIEPY:
            return clip.subclip(start_time, end_time)
        else:
            return clip.subclipped(start_time, end_time)
    except Exception as e:
        raise RuntimeError(f"Failed to trim video clip: {str(e)}")

def safe_apply_transform(clip, transform_fn):
    """Applies frame-by-frame transformation across MoviePy versions."""
    try:
        if IS_LEGACY_MOVIEPY:
            return clip.fl(lambda gf, t: transform_fn(gf(t), t))
        else:
            return clip.transform(lambda gf, t: transform_fn(gf(t), t))
    except Exception as e:
        raise RuntimeError(f"Failed to apply frame caption transform: {str(e)}")

# ==========================================
# 2. FFMPEG SYSTEM BINARY RESOLUTION
# ==========================================
def get_ffmpeg_binary_path():
    """Locates an operational FFmpeg executable path."""
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        pass
    
    # Fallback to system PATH environment
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    raise FileNotFoundError(
        "FFmpeg binary could not be located. Ensure imageio-ffmpeg is installed or packages.txt contains 'ffmpeg'."
    )

# ==========================================
# 3. ADVANCED YT-DLP DOWNLOAD ENGINE
# ==========================================
def download_media_from_url(url, target_directory):
    """
    Downloads source video clips from YouTube Shorts, TikTok, or direct URLs.
    Includes anti-403 client spoofing and fallback headers.
    """
    ffmpeg_binary = get_ffmpeg_binary_path()
    output_template = os.path.join(target_directory, 'source_clip.%(ext)s')

    # Comprehensive anti-bot spoofing parameters
    yt_options = {
        'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'ffmpeg_location': ffmpeg_binary,
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'overwrites': True,
        'nocheckcertificate': True,
        'ignoreerrors': False,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Sec-Fetch-Mode': 'navigate',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb', 'web']
            }
        }
    }

    try:
        with yt_dlp.YoutubeDL(yt_options) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info_dict)

            # Verify downloaded file exists
            if os.path.exists(downloaded_file):
                return downloaded_file

            # Secondary search if extension shifted during download
            for file_name in os.listdir(target_directory):
                if file_name.startswith("source_clip"):
                    return os.path.join(target_directory, file_name)

            raise FileNotFoundError("Video downloaded successfully but target file path could not be resolved.")

    except Exception as error:
        raise RuntimeError(f"URL Download Engine Error: {str(error)}")

# ==========================================
# 4. TEXT PREPROCESSING & TTS SYNTHESIS
# ==========================================
def clean_script_text(raw_text):
    """Sanitizes user input for audio synthesis and word wrapping."""
    if not raw_text or not raw_text.strip():
        return ""
    # Remove unsupported special characters/emojis while maintaining basic punctuation
    sanitized = re.sub(r'[^\w\s\.,!\?\'\-]', '', raw_text)
    # Collapse consecutive whitespace
    return re.sub(r'\s+', ' ', sanitized).strip()

def generate_voiceover_audio(text, accent_country, output_filepath):
    """Generates an MP3 audio file using Google Text-to-Speech (gTTS)."""
    clean_text = clean_script_text(text)
    if not clean_text:
        raise ValueError("Script text is empty after sanitization.")

    tld_mapping = {
        "US English": "com",
        "UK English": "co.uk",
        "Australian English": "com.au",
        "Indian English": "co.in",
        "Canadian English": "ca"
    }
    selected_tld = tld_mapping.get(accent_country, "com")

    try:
        tts_engine = gTTS(text=clean_text, lang="en", tld=selected_tld, slow=False)
        tts_engine.save(output_filepath)

        if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) == 0:
            raise RuntimeError("Generated audio file is corrupted or zero bytes.")

        return output_filepath
    except Exception as error:
        raise RuntimeError(f"Voiceover Synthesis Error: {str(error)}")

# ==========================================
# 5. ANIMATED CAPTION & OVERLAY ENGINE
# ==========================================
def load_scalable_font(frame_height, user_size_factor=0.05):
    """Attempts to load true-type system fonts with graceful fallback."""
    calculated_size = max(int(frame_height * user_size_factor), 18)
    
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "Arial-Bold.ttf",
        "arial.ttf",
        "Helvetica-Bold.ttf",
        "LiberationSans-Bold.ttf"
    ]

    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, calculated_size)
        except Exception:
            continue

    return ImageFont.load_default()

def render_caption_overlay(frame_array, current_timestamp, script_text, total_duration, style_preset, pos_y_percent=0.72):
    """
    Renders high-contrast, time-synchronized subtitle boxes over raw video frames.
    """
    pil_image = Image.fromarray(frame_array).convert("RGB")
    draw_context = ImageDraw.Draw(pil_image)
    frame_width, frame_height = pil_image.size

    words_list = script_text.split()
    if not words_list:
        return np.array(pil_image)

    # Divide text into short readable phrase chunks (3-4 words)
    words_per_chunk = 4
    phrase_chunks = [" ".join(words_list[i:i + words_per_chunk]) for i in range(0, len(words_list), words_per_chunk)]

    # Compute current active phrase based on timestamp
    time_per_chunk = total_duration / max(len(phrase_chunks), 1)
    active_chunk_idx = min(int(current_timestamp // time_per_chunk), len(phrase_chunks) - 1)
    active_phrase = phrase_chunks[max(0, active_chunk_idx)]

    font = load_scalable_font(frame_height, user_size_factor=0.05)

    # Compute phrase bounding dimensions for centering
    try:
        bounding_box = draw_context.textbbox((0, 0), active_phrase, font=font)
        text_w = bounding_box[2] - bounding_box[0]
        text_h = bounding_box[3] - bounding_box[1]
    except Exception:
        text_w = len(active_phrase) * 14
        text_h = 32

    pos_x = (frame_width - text_w) // 2
    pos_y = int(frame_height * pos_y_percent)

    # Color Palette Mapping
    if "Hormozi" in style_preset:
        text_color = (255, 225, 0)      # Yellow
        bg_box_color = (0, 0, 0)        # Black
        border_color = (0, 0, 0)
    elif "Neon" in style_preset:
        text_color = (0, 242, 254)      # Cyan
        bg_box_color = (15, 15, 25)     # Dark Slate
        border_color = (255, 0, 128)    # Neon Magenta Outline
    elif "Impact Red" in style_preset:
        text_color = (255, 255, 255)    # White
        bg_box_color = (220, 20, 60)    # Crimson Red
        border_color = (0, 0, 0)
    else:
        text_color = (255, 255, 255)    # Clean White
        bg_box_color = (0, 0, 0)
        border_color = (0, 0, 0)

    padding_x = 18
    padding_y = 12

    # Render Contrast Background Rectangle
    draw_context.rectangle(
        [pos_x - padding_x, pos_y - padding_y, pos_x + text_w + padding_x, pos_y + text_h + padding_y],
        fill=bg_box_color
    )

    # Render Text Outline / Drop Shadow
    for stroke_x, stroke_y in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 3), (0, -3), (3, 0), (-3, 0)]:
        draw_context.text((pos_x + stroke_x, pos_y + stroke_y), active_phrase, font=font, fill=border_color)

    # Render Primary Text Body
    draw_context.text((pos_x, pos_y), active_phrase, font=font, fill=text_color)

    return np.array(pil_image)

# ==========================================
# 6. STREAMLIT APPLICATION INTERFACE
# ==========================================
st.set_page_config(
    page_title="ViewMax AI - Video Studio Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Dark Theme CSS
st.markdown("""
    <style>
        .stApp { background-color: #0b0e14; color: #e6edf3; font-family: 'Inter', sans-serif; }
        .main-title {
            font-size: 2.5rem; font-weight: 900;
            background: linear-gradient(90deg, #FF416C, #8A2BE2, #00F2FE);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF416C, #8A2BE2);
            color: white; font-weight: 800; border: none;
            border-radius: 8px; padding: 0.85rem 1.5rem; width: 100%;
            font-size: 1.1rem; transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(138, 43, 226, 0.4); }
        .status-box { padding: 1rem; border-radius: 8px; background-color: #161b22; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 ViewMax Repurposer Pro</div>', unsafe_allow_html=True)
st.caption("Convert YouTube Shorts or raw footage into automated short videos with AI voiceovers and synced captions.")

# Sidebar Controls
st.sidebar.title("⚙️ Engine Configuration")
selected_accent = st.sidebar.selectbox("AI Voice Accent", ["US English", "UK English", "Australian English", "Indian English", "Canadian English"])
selected_preset = st.sidebar.selectbox("Caption Preset", ["Alex Hormozi Yellow Box", "Neon Cyberpunk", "Impact Red Banner", "Classic Clean White"])
caption_position_y = st.sidebar.slider("Caption Vertical Alignment", 0.50, 0.85, 0.72, 0.02, help="Lower values move text up; higher values move text down.")

st.sidebar.divider()
st.sidebar.subheader("System Status")
st.sidebar.code(f"MoviePy Mode: {'1.x (Legacy)' if IS_LEGACY_MOVIEPY else '2.x (Modern)'}\nPython: {sys.version.split()[0]}")

# Layout Columns
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("1. Source Video Clip")
    input_method = st.radio("Select Video Input Source:", ["YouTube / Shorts URL", "Upload Local Video File"], horizontal=True)

    input_url = ""
    uploaded_file = None

    if input_method == "YouTube / Shorts URL":
        input_url = st.text_input("Paste Video URL:", placeholder="https://www.youtube.com/shorts/...")
    else:
        uploaded_file = st.file_uploader("Upload Raw MP4/MOV File:", type=["mp4", "mov", "avi"])

    st.subheader("2. Voiceover Script")
    user_script = st.text_area(
        "Enter AI Voiceover Script:",
        value="This simple psychological habit doubles your focus in less than two minutes. Stop multitasking and try this right now!",
        height=140
    )

with right_col:
    st.subheader("3. Source Media Status")
    if input_method == "YouTube / Shorts URL" and input_url.strip():
        st.info(f"🔗 Target Link Configured:\n`{input_url.strip()}`")
    elif input_method == "Upload Local Video File" and uploaded_file is not None:
        st.video(uploaded_file)
        st.success("File uploaded successfully.")
    else:
        st.info("Provide a video URL or upload a file to begin processing.")

st.divider()

# ==========================================
# 7. EXECUTION ENGINE & RENDERING PIPELINE
# ==========================================
if st.button("🚀 Process & Export Repurposed MP4 Video"):
    # Validation checks
    cleaned_script = clean_script_text(user_script)
    
    if input_method == "YouTube / Shorts URL" and not input_url.strip():
        st.error("Please enter a valid YouTube or video URL.")
    elif input_method == "Upload Local Video File" and uploaded_file is None:
        st.error("Please upload a local video file.")
    elif not cleaned_script:
        st.error("Please provide a valid script text.")
    else:
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        # Execute processing inside a managed temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            source_video_path = os.path.join(temp_dir, "input_source.mp4")
            
            try:
                # STEP 1: Acquire Source Video File
                status_box.info("Step 1/5: Fetching source video asset...")
                if input_method == "YouTube / Shorts URL":
                    source_video_path = download_media_from_url(input_url.strip(), temp_dir)
                else:
                    with open(source_video_path, "wb") as f:
                        f.write(uploaded_file.read())
                progress_bar.progress(20)

                # STEP 2: Strip Original Audio (Raw Footage)
                status_box.info("Step 2/5: Stripping original audio into raw footage...")
                raw_clip_obj = VideoFileClip(source_video_path)
                raw_footage_clip = raw_clip_obj.without_audio()
                progress_bar.progress(35)

                # STEP 3: Generate AI Voiceover Audio
                status_box.info("Step 3/5: Synthesizing AI voiceover track...")
                voice_audio_path = os.path.join(temp_dir, "generated_speech.mp3")
                generate_voiceover_audio(cleaned_script, selected_accent, voice_audio_path)
                
                ai_audio_clip = AudioFileClip(voice_audio_path)
                speech_duration = ai_audio_clip.duration
                progress_bar.progress(55)

                # STEP 4: Duration Synchronization (Loop or Trim)
                status_box.info("Step 4/5: Synchronizing video length with voiceover duration...")
                if raw_footage_clip.duration < speech_duration:
                    loop_count = int(math.ceil(speech_duration / raw_footage_clip.duration))
                    synced_video = concatenate_videoclips([raw_footage_clip] * loop_count)
                    synced_video = safe_subclip(synced_video, 0, speech_duration)
                else:
                    synced_video = safe_subclip(raw_footage_clip, 0, speech_duration)
                progress_bar.progress(70)

                # STEP 5: Apply Captions & Render MP4 Export
                status_box.info("Step 5/5: Burning animated captions and encoding final MP4...")
                
                captioned_video = safe_apply_transform(
                    synced_video,
                    lambda frame, t: render_caption_overlay(
                        frame, t, cleaned_script, speech_duration, selected_preset, pos_y_percent=caption_position_y
                    )
                )

                final_output_clip = safe_set_audio(captioned_video, ai_audio_clip)
                export_file_path = os.path.join(temp_dir, "viewmax_final_export.mp4")

                final_output_clip.write_videofile(
                    export_file_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    logger=None
                )

                progress_bar.progress(100)
                status_box.success("🎉 Video Processing & Rendering Complete!")

                # Display Output and Download Option
                with open(export_file_path, "rb") as rendered_file:
                    final_video_bytes = rendered_file.read()

                st.subheader("🎬 Final Repurposed Output")
                st.video(final_video_bytes)
                st.download_button(
                    label="📥 Download Repurposed MP4 Video",
                    data=final_video_bytes,
                    file_name="viewmax_repurposed_video.mp4",
                    mime="video/mp4"
                )

                # Cleanup MoviePy handles
                raw_clip_obj.close()
                ai_audio_clip.close()

            except Exception as unhandled_error:
                progress_bar.progress(0)
                status_box.error(f"Execution Error: {str(unhandled_error)}")
                with st.expander("🔍 Stack Trace & Debug Information"):
                    st.code(traceback.format_exc())
