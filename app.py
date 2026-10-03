import streamlit as st
import os
import sys
import re
import math
import tempfile
import traceback
import requests
import numpy as np
import yt_dlp
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from gtts import gTTS

# ==========================================
# 1. UNIVERSAL MOVIEPY ENGINE ABSTRACTION
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

def safe_apply_transform(clip, transform_fn):
    return clip.fl(lambda gf, t: transform_fn(gf(t), t)) if IS_LEGACY_MOVIEPY else clip.transform(lambda gf, t: transform_fn(gf(t), t))

# ==========================================
# 2. SYSTEM BINARY & INGESTION ENGINE
# ==========================================
def get_ffmpeg_binary():
    try:
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return "ffmpeg"

def download_media_from_url(url, target_directory):
    """Multi-stage download engine designed for cloud IP resilience."""
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
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Safari/604.1',
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

    # External Proxy Tunnel Fallback
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
        raise RuntimeError("Cloud IP blocked by YouTube (HTTP 403). Switch to 'Upload Local Video File' or run locally.")
    raise RuntimeError(f"Ingestion Engine Error: {str(last_err)}")

# ==========================================
# 3. TEXT SANITIZATION & TTS ENGINE
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
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("TTS Output generation failed.")
    return output_path

# ==========================================
# 4. VIEWMAX CANVAS & REFRAMING ENGINE
# ==========================================
def reframe_canvas(frame_array, target_aspect="9:16"):
    """
    Reframes footage into target aspect ratios with professional blurred background fill.
    """
    img = Image.fromarray(frame_array).convert("RGB")
    src_w, src_h = img.size

    aspect_ratios = {
        "9:16 Shorts/Reels": (1080, 1920),
        "1:1 Square": (1080, 1080),
        "16:9 Landscape": (1920, 1080)
    }
    target_w, target_h = aspect_ratios.get(target_aspect, (1080, 1920))

    # Calculate scaling to fit inside target without distortion
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    scaled_foreground = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Generate Blurred Canvas Background
    bg_scale = max(target_w / src_w, target_h / src_h)
    bg_w = max(1, int(src_w * bg_scale))
    bg_h = max(1, int(src_h * bg_scale))
    background = img.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
    
    # Center-crop background and apply heavy gaussian blur
    crop_x = (bg_w - target_w) // 2
    crop_y = (bg_h - target_h) // 2
    background = background.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    background = background.filter(ImageFilter.GaussianBlur(radius=25))

    # Composite sharp centered video over blurred canvas
    pos_x = (target_w - new_w) // 2
    pos_y = (target_h - new_h) // 2
    background.paste(scaled_foreground, (pos_x, pos_y))

    return np.array(background)

# ==========================================
# 5. DYNAMIC WORD-BY-WORD CAPTION ENGINE
# ==========================================
def load_font(height, size_factor=0.045):
    font_size = max(int(height * size_factor), 18)
    font_names = ["DejaVuSans-Bold.ttf", "Arial-Bold.ttf", "arial.ttf", "Helvetica-Bold.ttf"]
    for f in font_names:
        try:
            return ImageFont.truetype(f, font_size)
        except Exception:
            continue
    return ImageFont.load_default()

def render_word_level_captions(frame_array, t, script_text, total_dur, preset, pos_y_pct=0.75):
    """
    Renders word-by-word active highlighting over video frames.
    """
    img = Image.fromarray(frame_array).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    words = script_text.split()
    if not words:
        return np.array(img)

    # Calculate exact word timing
    time_per_word = total_dur / len(words)
    active_word_idx = min(int(t // time_per_word), len(words) - 1)

    # Chunk words into sets of 3 for vertical shorts readability
    chunk_size = 3
    active_chunk_idx = active_word_idx // chunk_size
    chunk_start = active_chunk_idx * chunk_size
    chunk_words = words[chunk_start : chunk_start + chunk_size]

    font = load_font(h)
    
    # Measure total phrase width
    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    word_widths = [draw.textbbox((0, 0), w_text, font=font)[2] for w_text in chunk_words]
    total_phrase_w = sum(word_widths) + space_w * (len(chunk_words) - 1)
    
    try:
        sample_h = draw.textbbox((0, 0), "Hg", font=font)[3]
    except Exception:
        sample_h = 30

    start_x = (w - total_phrase_w) // 2
    start_y = int(h * pos_y_pct)

    # Preset Styles Definition
    if "Hormozi Active" in preset:
        active_color = (255, 230, 0)     # High-vis Yellow
        inactive_color = (255, 255, 255) # Pure White
        bg_fill = (0, 0, 0)
        border_col = (0, 0, 0)
    elif "Cyberpunk Glow" in preset:
        active_color = (0, 242, 254)     # Neon Cyan
        inactive_color = (180, 180, 200) # Muted Silver
        bg_fill = (15, 15, 30)
        border_col = (255, 0, 128)       # Magenta Stroke
    else: # Impact Red
        active_color = (255, 255, 255)   # White
        inactive_color = (220, 220, 220)
        bg_fill = (220, 20, 60)          # Crimson Red Box
        border_col = (0, 0, 0)

    # Draw Contrast Background Box behind the active phrase chunk
    pad_x, pad_y = 20, 12
    draw.rectangle(
        [start_x - pad_x, start_y - pad_y, start_x + total_phrase_w + pad_x, start_y + sample_h + pad_y],
        fill=bg_fill,
        outline=border_col if "Cyberpunk" in preset else None,
        width=2
    )

    # Render Words sequentially across X axis with Active Word Highlight
    curr_x = start_x
    for i, word_str in enumerate(chunk_words):
        global_word_idx = chunk_start + i
        is_active = (global_word_idx == active_word_idx)
        color = active_color if is_active else inactive_color

        # Outline Stroke
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 2), (0, -2)]:
            draw.text((curr_x + dx, start_y + dy), word_str, font=font, fill=(0, 0, 0))

        # Active Word Scale Effect / Shadow Box
        if is_active:
            draw.text((curr_x, start_y), word_str, font=font, fill=color)
        else:
            draw.text((curr_x, start_y), word_str, font=font, fill=color)

        curr_x += word_widths[i] + space_w

    return np.array(img)

# ==========================================
# 6. STUDIO SAAS DASHBOARD UI
# ==========================================
st.set_page_config(page_title="ViewMax Studio - Viral Clip Engine", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #080a0f; color: #f0f4f8; font-family: 'Inter', sans-serif; }
        .hero-title {
            font-size: 2.8rem; font-weight: 900;
            background: linear-gradient(90deg, #FF416C, #8A2BE2, #00F2FE);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF416C, #8A2BE2);
            color: white; font-weight: 800; border: none;
            border-radius: 10px; padding: 1rem 1.5rem; font-size: 1.15rem;
            box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3); transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(138, 43, 226, 0.5); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">⚡ ViewMax Studio Engine</div>', unsafe_allow_html=True)
st.caption("AI-Powered Short-Form Video Repurposer: Auto-Reframing, Voiceovers, and Word-Level Active Highlighting.")

# Sidebar Configurations
st.sidebar.title("🎛️️ Studio Controls")
selected_accent = st.sidebar.selectbox("AI Voice Accent", ["US", "UK", "Australia", "India", "Canada"])
canvas_aspect = st.sidebar.selectbox("Canvas Reframing Aspect Ratio", ["9:16 Shorts/Reels", "1:1 Square", "16:9 Landscape"])
caption_preset = st.sidebar.selectbox("Caption Style Preset", ["Hormozi Active Highlight", "Cyberpunk Glow", "Impact Red Banner"])
caption_y_pos = st.sidebar.slider("Caption Vertical Alignment", 0.50, 0.85, 0.75, 0.02)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Media Asset")
    input_type = st.radio("Source Type:", ["YouTube / Shorts URL", "Upload Video File"], horizontal=True)

    url_input = ""
    uploaded_file = None

    if input_type == "YouTube / Shorts URL":
        url_input = st.text_input("YouTube Shorts / Video URL:", placeholder="https://www.youtube.com/shorts/...")
    else:
        uploaded_file = st.file_uploader("Upload MP4 / MOV Video", type=["mp4", "mov", "avi"])

    st.subheader("2. AI Voiceover Script")
    script_input = st.text_area(
        "Enter Script Text:",
        value="This single psychological trigger will instantly double your content retention. Stop scrolling and test this right now!",
        height=130
    )

with col2:
    st.subheader("3. Asset Workspace Status")
    if input_type == "YouTube / Shorts URL" and url_input.strip():
        st.info(f"🔗 Targeted Media Stream:\n`{url_input.strip()}`")
    elif input_type == "Upload Video File" and uploaded_file:
        st.video(uploaded_file)
        st.success("Media file loaded into memory workspace.")
    else:
        st.info("Provide a video link or upload footage on the left to activate processing.")

st.divider()

# ==========================================
# 7. RENDERING & PIPELINE EXECUTION
# ==========================================
if st.button("🚀 Render Repurposed Video Clip"):
    cleaned = sanitize_script(script_input)
    
    if input_type == "YouTube / Shorts URL" and not url_input.strip():
        st.error("Please enter a YouTube video URL.")
    elif input_type == "Upload Video File" and not uploaded_file:
        st.error("Please upload a local video file.")
    elif not cleaned:
        st.error("Script text cannot be empty.")
    else:
        status_box = st.empty()
        progress = st.progress(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "raw_source.mp4")

            try:
                # Step 1: Ingestion
                status_box.info("1/5 Ingesting source video asset...")
                if input_type == "YouTube / Shorts URL":
                    source_path = download_media_from_url(url_input.strip(), temp_dir)
                else:
                    with open(source_path, "wb") as f:
                        f.write(uploaded_file.read())
                progress.progress(20)

                # Step 2: Strip Audio & Reframe Canvas
                status_box.info("2/5 Stripping audio & reframing canvas layout...")
                raw_clip = VideoFileClip(source_path).without_audio()
                
                # Apply Canvas Reframing (Blur background fill)
                reframed_clip = safe_apply_transform(
                    raw_clip,
                    lambda frame, t: reframe_canvas(frame, target_aspect=canvas_aspect)
                )
                progress.progress(40)

                # Step 3: AI Speech Synthesis
                status_box.info("3/5 Synthesizing AI voiceover track...")
                audio_path = os.path.join(temp_dir, "voice.mp3")
                build_voiceover(cleaned, selected_accent, audio_path)
                
                ai_audio = AudioFileClip(audio_path)
                speech_dur = ai_audio.duration
                progress.progress(60)

                # Step 4: Duration Synchronization
                status_box.info("4/5 Synchronizing video loop to speech duration...")
                if reframed_clip.duration < speech_dur:
                    loops = int(math.ceil(speech_dur / reframed_clip.duration))
                    synced_video = safe_subclip(concatenate_videoclips([reframed_clip] * loops), 0, speech_dur)
                else:
                    synced_video = safe_subclip(reframed_clip, 0, speech_dur)
                progress.progress(75)

                # Step 5: Word-Level Captions & Export
                status_box.info("5/5 Burning word-level active captions & encoding MP4...")
                final_captioned = safe_apply_transform(
                    synced_video,
                    lambda frame, t: render_word_level_captions(
                        frame, t, cleaned, speech_dur, caption_preset, pos_y_pct=caption_y_pos
                    )
                )

                output_clip = safe_set_audio(final_captioned, ai_audio)
                export_path = os.path.join(temp_dir, "viewmax_final.mp4")

                output_clip.write_videofile(
                    export_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    preset="fast",
                    logger=None
                )

                progress.progress(100)
                status_box.success("🎉 Processing complete! Your video is ready.")

                with open(export_path, "rb") as f:
                    rendered_bytes = f.read()

                st.subheader("🎬 Final Repurposed Output")
                st.video(rendered_bytes)
                st.download_button(
                    label="📥 Download ViewMax Repurposed MP4",
                    data=rendered_bytes,
                    file_name="viewmax_repurposed.mp4",
                    mime="video/mp4"
                )

                raw_clip.close()
                ai_audio.close()

            except Exception as err:
                progress.progress(0)
                status_box.error(f"Execution Error: {str(err)}")
                with st.expander("🔍 Stack Trace & Debug Info"):
                    st.code(traceback.format_exc())
