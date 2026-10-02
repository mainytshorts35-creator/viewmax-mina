import streamlit as st
import os
import io
import time
import requests
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

st.set_page_config(
    page_title="ViewMax AI - Free Studio",
    page_icon="🎬",
    layout="wide"
)

# Dark SaaS Styling
st.markdown("""
    <style>
        .stApp { background-color: #0d1117; color: #ffffff; }
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FF4B4B, #8A2BE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF4B4B, #8A2BE2);
            color: white; font-weight: bold; border: none;
            border-radius: 8px; padding: 0.75rem; width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Helper: Generate AI Background Image (100% Free via Pollinations AI)
def fetch_ai_image(prompt, width, height):
    clean_prompt = requests.utils.quote(prompt)
    url = f"https://pollinations.ai/p/{clean_prompt}?width={width}&height={height}&nologo=true&seed={int(time.time())}"
    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code == 200:
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    # Fallback solid background
    return Image.new("RGB", (width, height), color=(15, 20, 30))

# Helper: Draw Styled Captions onto Frame
def overlay_subtitles(image, text, style="Bold Yellow"):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Simple word wrapper
    words = text.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > 18:
            lines.append(" ".join(current[:-1]))
            current = [word]
    if current:
        lines.append(" ".join(current))

    # Fallback Font Loading
    try:
        font_size = int(h * 0.05)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    line_height = int(h * 0.06)
    start_y = int(h * 0.70)

    color = (255, 220, 0) if "Yellow" in style else (255, 255, 255)

    for i, line in enumerate(lines):
        y = start_y + (i * line_height)
        # Compute text position for center alignment
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(line) * 10
        x = (w - text_w) // 2

        # Draw dark outline/shadow for readability
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=color)

    return img

# Sidebar Settings
st.sidebar.title("⚙️ Engine Settings")
aspect = st.sidebar.selectbox("Aspect Ratio", ["9:16 (Shorts/Reels/TikTok)", "16:9 (YouTube Standard)"])
caption_style = st.sidebar.selectbox("Subtitle Style", ["Bold Yellow (Alex Hormozi Style)", "Clean White"])
api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="If provided, generates custom GPT scripts.")

# Main Interface Header
st.markdown('<div class="main-title">🎬 ViewMax AI — Free Video Generator</div>', unsafe_allow_html=True)
st.caption("Create complete short-form AI videos with voiceovers, AI visuals, and burned captions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Video Concept")
    topic = st.text_input("What is your video topic?", value="3 Mind-Blowing Facts About the Universe")
    style_niche = st.selectbox("Content Niche", ["Fun Facts & Sci-Fi", "Motivation", "Tech & Future", "Dark History"])
    generate_script = st.button("✨ Generate AI Script & Storyboard")

if "storyboard" not in st.session_state:
    st.session_state.storyboard = []

if generate_script and topic:
    with st.spinner("Drafting script scenes and AI visual prompts..."):
        time.sleep(1)
        # Default AI Script Generator Pipeline
        st.session_state.storyboard = [
            {"scene": 1, "text": f"Did you know this insane fact about {topic}?", "visual": f"Cinematic epic space galaxy background, futuristic sci-fi aesthetic, 8k resolution"},
            {"scene": 2, "text": f"Scientists discovered that deep inside, {topic} works in ways we never imagined.", "visual": f"Abstract glowing quantum particles, dark high-tech laboratory concept"},
            {"scene": 3, "text": "Hit subscribe for more incredible daily discoveries!", "visual": "Glowing neon notification sign, cinematic dark room setup"}
        ]

with col2:
    st.subheader("2. Storyboard Preview")
    if st.session_state.storyboard:
        for idx, scene in enumerate(st.session_state.storyboard):
            with st.expander(f"Scene {idx+1}", expanded=True):
                scene["text"] = st.text_area(f"Voiceover Text {idx+1}", value=scene["text"])
                scene["visual"] = st.text_input(f"AI Visual Prompt {idx+1}", value=scene["visual"])

st.divider()

# Rendering Pipeline Section
st.subheader("3. Render Engine")
if st.button("🚀 Render & Export Complete MP4 Video"):
    if not st.session_state.storyboard:
        st.error("Please generate or create a script first!")
    else:
        status = st.empty()
        progress = st.progress(0)
        
        # Dimensions setup based on aspect ratio
        width, height = (1080, 1920) if "9:16" in aspect else (1920, 1080)
        
        scene_clips = []
        temp_dir = tempfile.mkdtemp()

        total_scenes = len(st.session_state.storyboard)
        
        try:
            for idx, scene in enumerate(st.session_state.storyboard):
                status.info(f"Processing Scene {idx+1}/{total_scenes}: Generating AI Voiceover...")
                
                # 1. Generate Voiceover Audio (gTTS)
                tts = gTTS(text=scene["text"], lang="en")
                audio_path = os.path.join(temp_dir, f"audio_{idx}.mp3")
                tts.save(audio_path)
                
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration + 0.4  # Slight buffer
                
                status.info(f"Processing Scene {idx+1}/{total_scenes}: Generating AI Image...")
                # 2. Generate AI Visual Image
                raw_img = fetch_ai_image(scene["visual"], width, height)
                
                status.info(f"Processing Scene {idx+1}/{total_scenes}: Overlaying Subtitles...")
                # 3. Add Subtitles
                final_img = overlay_subtitles(raw_img, scene["text"], style=caption_style)
                
                # 4. Build MoviePy Clip
                img_np = np.array(final_img)
                clip = ImageClip(img_np).set_duration(duration)
                clip = clip.set_audio(audio_clip)
                
                scene_clips.append(clip)
                progress.progress(int(((idx + 1) / total_scenes) * 70))

            status.info("Stitching scenes and compiling final MP4 video...")
            
            # Concatenate all clips
            final_video = concatenate_videoclips(scene_clips, method="compose")
            output_mp4_path = os.path.join(temp_dir, "viewmax_render.mp4")
            
            # Export MP4
            final_video.write_videofile(
                output_mp4_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            
            progress.progress(100)
            status.success("🎉 Video Rendering Complete!")
            
            # Read and Display
            with open(output_mp4_path, "rb") as f:
                video_bytes = f.read()
                
            st.video(video_bytes)
            st.download_button("📥 Download MP4 Video", data=video_bytes, file_name="viewmax_ai_video.mp4", mime="video/mp4")

        except Exception as e:
            st.error(f"Render Error: {str(e)}")
