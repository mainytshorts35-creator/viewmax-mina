import streamlit as st
import os
import io
import time
import json
import requests
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# Smart MoviePy compatibility layer (Supports MoviePy v1 and v2+)
try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, AudioArrayClip
    LEGACY_MOVIEPY = True
except (ImportError, ModuleNotFoundError):
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, AudioArrayClip
    LEGACY_MOVIEPY = False

# Helper functions for cross-version MoviePy clip manipulation
def clip_duration(clip, dur):
    return clip.set_duration(dur) if LEGACY_MOVIEPY else clip.with_duration(dur)

def clip_audio(clip, audio):
    return clip.set_audio(audio) if LEGACY_MOVIEPY else clip.with_audio(audio)

def clip_vol(audio, vol):
    return audio.volumex(vol) if LEGACY_MOVIEPY else audio.with_volume_scaled(vol)

# Page Configuration
st.set_page_config(
    page_title="ViewMax AI - Video Creation Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Dark Theme CSS
st.markdown("""
    <style>
        .stApp { background-color: #0b0e14; color: #e6edf3; font-family: 'Inter', sans-serif; }
        .main-header {
            font-size: 2.5rem; font-weight: 900;
            background: linear-gradient(90deg, #FF416C, #8A2BE2, #00F2FE);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF416C, #8A2BE2);
            color: white; font-weight: 700; border: none; border-radius: 8px;
            padding: 0.75rem 1.5rem; width: 100%; transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(138, 43, 226, 0.4); }
        .card { background: #161b22; border: 1px solid #30363d; padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "storyboard" not in st.session_state:
    st.session_state.storyboard = []
if "history" not in st.session_state:
    st.session_state.history = []

# --- AI & Media Helper Engines ---

def generate_gpt_script(topic, niche, tone, openai_key):
    """Generates structured scene script via OpenAI API or fallback template generator."""
    if openai_key and openai_key.startswith("sk-"):
        try:
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            prompt = f"Create a viral 3-scene video script about '{topic}' for the niche '{niche}' with a '{tone}' tone. Output ONLY JSON: [{{\"scene\": 1, \"text\": \"...\", \"visual\": \"...\"}}, ...]"
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception:
            pass

    # Intelligent Template Engine
    return [
        {
            "scene": 1,
            "text": f"Did you know this mind-blowing truth about {topic}?",
            "visual": f"Cinematic epic 8k resolution shot of {topic}, {tone} aesthetic, dramatic lighting, detailed"
        },
        {
            "scene": 2,
            "text": f"In the world of {niche.lower()}, experts revealed something that changes everything we knew.",
            "visual": f"Futuristic laboratory quantum energy breakdown, neon glow, hyperrealistic"
        },
        {
            "scene": 3,
            "text": "Follow and subscribe right now for more incredible daily insights!",
            "visual": "Neon glowing subscribe button inside modern high tech studio background"
        }
    ]

def fetch_ai_visual(prompt, width, height):
    """Generates unique HD AI artwork using Pollinations AI engine."""
    clean_prompt = requests.utils.quote(prompt)
    seed = int(time.time() * 1000) % 100000
    url = f"https://pollinations.ai/p/{clean_prompt}?width={width}&height={height}&nologo=true&seed={seed}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            return Image.open(io.BytesIO(res.content)).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (width, height), color=(15, 20, 30))

def render_subtitle_frame(image, text, style_preset):
    """Renders styled, auto-wrapped subtitles onto the video frame."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Word wrapping logic
    max_chars = 18 if w < h else 36
    words = text.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > max_chars:
            lines.append(" ".join(current[:-1]))
            current = [word]
    if current:
        lines.append(" ".join(current))

    # Font setup
    try:
        font_size = int(h * 0.048)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    line_height = int(h * 0.058)
    start_y = int(h * 0.68)

    # Style presets
    text_color = (255, 225, 0) if "Hormozi" in style_preset else (0, 242, 254) if "Cyberpunk" in style_preset else (255, 255, 255)
    
    for i, line in enumerate(lines):
        y = start_y + (i * line_height)
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(line) * 12
        x = (w - text_w) // 2

        # Draw dark box overlay for readability
        if "Box" in style_preset or "Hormozi" in style_preset:
            pad = 10
            draw.rectangle([x - pad, y - 4, x + text_w + pad, y + line_height - 6], fill=(0, 0, 0, 180))

        # Text outlines & body
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=text_color)

    return img

def generate_synthetic_bgm(duration_sec, sample_rate=22050):
    """Synthesizes an ambient background chord track dynamically using NumPy."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    # Ambient chord synthesis (A minor pad)
    chord = 0.03 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 329.63 * t))
    fade = np.minimum(t / 2.0, (duration_sec - t) / 2.0)
    fade = np.clip(fade, 0, 1)
    stereo = np.column_stack((chord * fade, chord * fade))
    return AudioArrayClip(stereo, fps=sample_rate)

# --- Sidebar Configuration ---
st.sidebar.title("🎛️ Studio Configuration")

aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["9:16 (Shorts / Reels / TikTok)", "16:9 (YouTube Widescreen)"])
voice_accent = st.sidebar.selectbox("Voiceover Accent", ["US English", "UK English", "Australian English", "Indian English"])
caption_preset = st.sidebar.selectbox("Subtitle Preset", ["Alex Hormozi (Yellow + Box)", "Cyberpunk Neon (Cyan)", "Classic White (Clean)"])
bgm_volume = st.sidebar.slider("Background Music Volume", 0.0, 0.5, 0.15)
openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")

st.sidebar.divider()
st.sidebar.info("💡 **Pro-Tip:** Leaving the OpenAI key blank will use ViewMax's built-in intelligent script generator for free.")

# --- Main App Interface ---
st.markdown('<div class="main-header">🎬 ViewMax AI Video Studio</div>', unsafe_allow_html=True)
st.caption("Generate, edit, and export complete short-form AI videos with voiceovers, custom AI artwork, and burned captions.")

# Multi-Tab Workflow Layout
tab1, tab2, tab3, tab4 = st.tabs(["📝 Script & Storyboard", "🎨 AI Visual Customizer", "🎙️ Voice & Audio Studio", "🚀 Render & Export"])

# TAB 1: Script Studio
with tab1:
    st.subheader("1. Video Concept Generator")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        topic_input = st.text_input("What is your video topic?", value="5 Mind-Blowing Facts About AI")
    with c2:
        niche_input = st.selectbox("Niche", ["Tech & AI", "Motivation & Wealth", "Sci-Fi & Astronomy", "Dark History", "Fitness"])
    with c3:
        tone_input = st.selectbox("Tone", ["Dramatic", "Energetic", "Mysterious", "Professional"])

    if st.button("✨ Generate Script & Storyboard"):
        with st.spinner("Writing script and constructing visual storyboard..."):
            st.session_state.storyboard = generate_gpt_script(topic_input, niche_input, tone_input, openai_api_key)
            st.success("Storyboard created successfully!")

    if st.session_state.storyboard:
        st.divider()
        st.subheader("Timeline Editor")
        for idx, scene in enumerate(st.session_state.storyboard):
            with st.expander(f"🎬 Scene {idx+1}", expanded=True):
                sc1, sc2 = st.columns([1, 1])
                with sc1:
                    scene["text"] = st.text_area(f"Voiceover Text (Scene {idx+1})", value=scene["text"])
                with sc2:
                    scene["visual"] = st.text_area(f"AI Visual Prompt (Scene {idx+1})", value=scene["visual"])

# TAB 2: Visual Studio
with tab2:
    st.subheader("2. Visual Assets & Preview Engine")
    if not st.session_state.storyboard:
        st.warning("Generate a script in Tab 1 first!")
    else:
        w_preview, h_preview = (360, 640) if "9:16" in aspect_ratio else (640, 360)
        st.write("Preview generated AI visuals for each scene:")
        prev_cols = st.columns(len(st.session_state.storyboard))
        for idx, scene in enumerate(st.session_state.storyboard):
            with prev_cols[idx]:
                st.caption(f"Scene {idx+1} Frame")
                if st.button(f"🖼️ Test Generate Visual {idx+1}"):
                    with st.spinner("Rendering frame..."):
                        img = fetch_ai_visual(scene["visual"], w_preview, h_preview)
                        sub_img = render_subtitle_frame(img, scene["text"], caption_preset)
                        st.image(sub_img, use_container_width=True)

# TAB 3: Audio Studio
with tab3:
    st.subheader("3. Voiceover & Audio Tuning")
    if not st.session_state.storyboard:
        st.warning("Generate a script in Tab 1 first!")
    else:
        full_script = " ".join([s["text"] for s in st.session_state.storyboard])
        st.write("**Full Video Voiceover Script:**")
        st.info(full_script)

        if st.button("🔊 Preview AI Voiceover Audio"):
            with st.spinner("Synthesizing voiceover..."):
                tld = "co.uk" if "UK" in voice_accent else "co.in" if "Indian" in voice_accent else "com.au" if "Australian" in voice_accent else "com"
                tts = gTTS(text=full_script, lang="en", tld=tld)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                st.audio(fp, format="audio/mp3")

# TAB 4: Render & Export Engine
with tab4:
    st.subheader("4. Compile & Export Final MP4 Video")
    if not st.session_state.storyboard:
        st.warning("Please generate a script in Tab 1 first!")
    else:
        st.write("Ready to build your high-resolution MP4 video file.")
        
        if st.button("🚀 Render Complete Video"):
            status_box = st.empty()
            progress_bar = st.progress(0)
            
            width, height = (1080, 1920) if "9:16" in aspect_ratio else (1920, 1080)
            tld = "co.uk" if "UK" in voice_accent else "co.in" if "Indian" in voice_accent else "com.au" if "Australian" in voice_accent else "com"
            
            scene_clips = []
            temp_dir = tempfile.mkdtemp()
            total_scenes = len(st.session_state.storyboard)

            try:
                for idx, scene in enumerate(st.session_state.storyboard):
                    status_box.info(f"Processing Scene {idx+1}/{total_scenes}: Synthesizing Voiceover...")
                    
                    # 1. Voiceover
                    tts = gTTS(text=scene["text"], lang="en", tld=tld)
                    audio_path = os.path.join(temp_dir, f"voice_{idx}.mp3")
                    tts.save(audio_path)
                    
                    audio_clip = AudioFileClip(audio_path)
                    duration = audio_clip.duration + 0.3
                    
                    status_box.info(f"Processing Scene {idx+1}/{total_scenes}: Generating HD AI Artwork...")
                    # 2. AI Image
                    raw_img = fetch_ai_visual(scene["visual"], width, height)
                    
                    status_box.info(f"Processing Scene {idx+1}/{total_scenes}: Burning Subtitles...")
                    # 3. Subtitle Overlay
                    final_frame = render_subtitle_frame(raw_img, scene["text"], caption_preset)
                    
                    # 4. Construct Clip
                    frame_np = np.array(final_frame)
                    clip = ImageClip(frame_np)
                    clip = clip_duration(clip, duration)
                    clip = clip_audio(clip, audio_clip)
                    
                    scene_clips.append(clip)
                    progress_bar.progress(int(((idx + 1) / total_scenes) * 65))

                status_box.info("Stitching scenes & generating ambient background music...")
                
                # Combine video scenes
                base_video = concatenate_videoclips(scene_clips, method="compose")
                total_duration = base_video.duration
                
                # Mix Ambient BGM if enabled
                if bgm_volume > 0:
                    bgm_clip = generate_synthetic_bgm(total_duration)
                    bgm_clip = clip_vol(bgm_clip, bgm_volume)
                    combined_audio = CompositeAudioClip([base_video.audio, bgm_clip])
                    base_video = clip_audio(base_video, combined_audio)

                output_path = os.path.join(temp_dir, "viewmax_studio_export.mp4")
                status_box.info("Encoding final H.264 MP4 export...")
                
                base_video.write_videofile(
                    output_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None
                )
                
                progress_bar.progress(100)
                status_box.success("🎉 Video Render Complete!")
                
                with open(output_path, "rb") as f:
                    video_data = f.read()
                
                st.video(video_data)
                st.download_button("📥 Download Final MP4 Video", data=video_data, file_name="viewmax_video.mp4", mime="video/mp4")

            except Exception as e:
                status_box.error(f"Render Execution Error: {str(e)}")
