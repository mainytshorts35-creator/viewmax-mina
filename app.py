import streamlit as st
import os
import tempfile
import numpy as np
import yt_dlp
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# Universal MoviePy Compatibility Layer (Supports MoviePy v1 and v2+)
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    LEGACY_MOVIEPY = True
except (ImportError, ModuleNotFoundError):
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
    LEGACY_MOVIEPY = False

# Cross-version MoviePy Helper Functions
def set_clip_audio(clip, audio):
    return clip.set_audio(audio) if LEGACY_MOVIEPY else clip.with_audio(audio)

def trim_clip(clip, start_t, end_t):
    return clip.subclip(start_t, end_t) if LEGACY_MOVIEPY else clip.subclipped(start_t, end_t)

def apply_frame_transform(clip, transform_fn):
    """Applies a frame-by-frame transformation across both MoviePy 1.x and 2.x."""
    if LEGACY_MOVIEPY:
        return clip.fl(lambda gf, t: transform_fn(gf(t), t))
    else:
        return clip.transform(lambda gf, t: transform_fn(gf(t), t))

# Download Helper for Shorts / Video URLs
def download_clip_from_url(url, output_dir):
    """Downloads a video clip from YouTube Shorts, TikTok, or direct video URLs."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, 'downloaded_clip.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'overwrites': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename

# Streamlit Page Setup
st.set_page_config(
    page_title="ViewMax - Viral Clip Repurposer",
    page_icon="✂️",
    layout="wide"
)

# Dark SaaS Custom Styling
st.markdown("""
    <style>
        .stApp { background-color: #0d1117; color: #ffffff; font-family: 'Inter', sans-serif; }
        .main-header {
            font-size: 2.3rem; font-weight: 900;
            background: linear-gradient(90deg, #FF416C, #8A2BE2, #00F2FE);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF416C, #8A2BE2);
            color: white; font-weight: bold; border: none;
            border-radius: 8px; padding: 0.8rem 1.5rem; width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">✂️ ViewMax Viral Clip Repurposer</div>', unsafe_allow_html=True)
st.caption("Paste a YouTube Shorts URL or Upload a Clip $\\rightarrow$ Strip original audio $\\rightarrow$ Add AI Voiceover & Burned Captions.")

# --- Frame Caption Overlay Generator ---
def render_animated_caption(frame_np, current_time, text, total_duration, style_preset):
    """Calculates active words based on current playback time and draws animated captions."""
    img = Image.fromarray(frame_np).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    words = text.split()
    if not words:
        return np.array(img)

    # Calculate phrase segments (chunks of 3-4 words)
    chunk_size = 4
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    time_per_chunk = total_duration / max(len(chunks), 1)
    active_index = min(int(current_time // time_per_chunk), len(chunks) - 1)
    active_phrase = chunks[active_index] if active_index >= 0 else chunks[0]

    # Font Configuration
    try:
        font_size = int(h * 0.05)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), active_phrase, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w = len(active_phrase) * 12
        text_h = 30

    x = (w - text_w) // 2
    y = int(h * 0.72)

    text_color = (255, 220, 0) if "Hormozi" in style_preset else (0, 242, 254) if "Neon" in style_preset else (255, 255, 255)

    # Draw Background Box for Contrast
    pad_x, pad_y = 16, 10
    draw.rectangle([x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y], fill=(0, 0, 0))

    # Text Outlines
    for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 3)]:
        draw.text((x + dx, y + dy), active_phrase, font=font, fill=(0, 0, 0))
    draw.text((x, y), active_phrase, font=font, fill=text_color)

    return np.array(img)

# --- Sidebar Options ---
st.sidebar.title("🎛️ Repurposer Settings")
voice_accent = st.sidebar.selectbox("AI Voice Accent", ["US English", "UK English", "Australian English", "Indian English"])
caption_style = st.sidebar.selectbox("Caption Style Preset", ["Alex Hormozi Yellow Box", "Neon Cyberpunk", "Classic Clean White"])

# --- Main Studio Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Source Video Clip")
    source_method = st.radio("Choose Input Method:", ["YouTube / Shorts URL", "Upload Local Video File"], horizontal=True)

    shorts_url = ""
    uploaded_video = None

    if source_method == "YouTube / Shorts URL":
        shorts_url = st.text_input("Paste YouTube Shorts or Video URL:", placeholder="https://www.youtube.com/shorts/...")
    else:
        uploaded_video = st.file_uploader("Upload MP4, MOV, or AVI video file", type=["mp4", "mov", "avi"])

    st.subheader("2. New AI Script & Voiceover")
    script_text = st.text_area(
        "Enter new voiceover script:",
        value="This simple psychological trick will instantly double your productivity. Stop multitasking right now!",
        height=120
    )

with col2:
    st.subheader("3. Video Status & Output Preview")
    if shorts_url:
        st.info(f"🔗 Target URL set: `{shorts_url}`")
    elif uploaded_video:
        st.video(uploaded_video)
        st.success("Local viral clip uploaded successfully!")
    else:
        st.info("Provide a Shorts URL or upload a file on the left to get started.")

st.divider()

# --- Execution Engine ---
if st.button("🚀 Process & Repurpose Viral Clip"):
    if source_method == "YouTube / Shorts URL" and not shorts_url.strip():
        st.error("Please paste a valid YouTube Shorts URL!")
    elif source_method == "Upload Local Video File" and not uploaded_video:
        st.error("Please upload a video file!")
    elif not script_text.strip():
        st.error("Please enter a voiceover script!")
    else:
        status = st.empty()
        progress = st.progress(0)

        temp_dir = tempfile.mkdtemp()
        input_video_path = os.path.join(temp_dir, "input_clip.mp4")

        try:
            # Step 1: Obtain source video file
            if source_method == "YouTube / Shorts URL":
                status.info("1/5: Downloading clip from YouTube Shorts URL...")
                input_video_path = download_clip_from_url(shorts_url.strip(), temp_dir)
            else:
                status.info("1/5: Reading uploaded video file...")
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_video.read())

            progress.progress(20)

            # Step 2: Strip original audio into raw footage
            status.info("2/5: Stripping original audio to create raw footage...")
            raw_video = VideoFileClip(input_video_path)
            raw_footage = raw_video.without_audio()

            # Step 3: Generate TTS AI Voiceover
            status.info("3/5: Synthesizing AI voiceover track...")
            tld = "co.uk" if "UK" in voice_accent else "co.in" if "Indian" in voice_accent else "com.au" if "Australian" in voice_accent else "com"
            tts = gTTS(text=script_text, lang="en", tld=tld)
            audio_path = os.path.join(temp_dir, "ai_voice.mp3")
            tts.save(audio_path)

            ai_audio = AudioFileClip(audio_path)
            audio_duration = ai_audio.duration
            progress.progress(50)

            # Step 4: Time sync raw video to match speech length
            status.info("4/5: Syncing video duration with AI voiceover...")
            if raw_footage.duration < audio_duration:
                repeat_count = int(np.ceil(audio_duration / raw_footage.duration))
                timed_video = concatenate_videoclips([raw_footage] * repeat_count)
                timed_video = trim_clip(timed_video, 0, audio_duration)
            else:
                timed_video = trim_clip(raw_footage, 0, audio_duration)

            progress.progress(70)

            # Step 5: Burn animated captions & export
            status.info("5/5: Burning animated captions and exporting final MP4...")
            captioned_video = apply_frame_transform(
                timed_video,
                lambda frame, t: render_animated_caption(frame, t, script_text, audio_duration, caption_style)
            )

            final_clip = set_clip_audio(captioned_video, ai_audio)
            output_mp4_path = os.path.join(temp_dir, "repurposed_viral_clip.mp4")

            final_clip.write_videofile(
                output_mp4_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None
            )

            progress.progress(100)
            status.success("🎉 Video Repurposing Complete!")

            # Display Output & Download
            with open(output_mp4_path, "rb") as f:
                output_bytes = f.read()

            st.subheader("🎬 Final Repurposed Video")
            st.video(output_bytes)
            st.download_button(
                "📥 Download Repurposed MP4 Video",
                data=output_bytes,
                file_name="viewmax_repurposed.mp4",
                mime="video/mp4"
            )

        except Exception as e:
            status.error(f"Processing Error: {str(e)}")
