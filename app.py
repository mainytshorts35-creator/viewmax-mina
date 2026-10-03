"""
================================================================================
VIEWMAX STUDIO PRO — MASTER STREAMLIT DASHBOARD & PIPELINE ORCHESTRATOR
FILE: app.py
DESCRIPTION: Interactive Streamlit web interface for downloading YouTube Shorts,
             stripping original audio, synthesizing AI voiceovers, generating
             dynamic kinetic captions, blending background music, and rendering
             high-retention vertical videos.
================================================================================
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
import numpy as np
import streamlit as st

# MUST BE THE VERY FIRST STREAMLIT COMMAND EXECUTED
st.set_page_config(
    page_title="ViewMax Studio Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import (
    global_config,
    CANVAS_PRESETS,
    TYPOGRAPHY_PRESETS,
    AspectRatioEnum,
    CaptionStyleEnum,
    VoiceProviderEnum,
    EnvironmentValidator,
    get_logger
)
from audio_dsp import VocalMasteringProcessor, SidechainMusicDucker
from voice_synthesis import VoiceSynthesisManager, BUILTIN_SPEAKERS
from caption_renderer import (
    WhisperAlignmentEngine,
    PILCaptionGraphicsEngine,
    ASSSubtitleBuilder,
    MoviePyCaptionCompositor
)

logger = get_logger("ViewMaxPro.App")


# ==============================================================================
# 1. HELPER FUNCTIONS & PIPELINE COMPONENTS
# ==============================================================================
def download_youtube_video(url: str, output_dir: str) -> str:
    """Downloads YouTube Shorts video using yt-dlp."""
    output_template = os.path.join(output_dir, "downloaded_raw_input.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        url
    ]
    
    st.info("📥 Downloading source video via yt-dlp...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr}")

    downloaded_file = os.path.join(output_dir, "downloaded_raw_input.mp4")
    if not os.path.exists(downloaded_file):
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mp4")]
        if files:
            downloaded_file = files[0]
        else:
            raise FileNotFoundError("Downloaded MP4 video file could not be located.")

    return downloaded_file


def strip_audio_from_video(video_path: str, output_path: str) -> str:
    """Strips audio track from video file, returning raw muted footage."""
    _, ffmpeg_bin = EnvironmentValidator.check_ffmpeg()
    if not ffmpeg_bin:
        ffmpeg_bin = "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-an",
        "-c:v", "copy",
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        cmd_reencode = [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            output_path
        ]
        subprocess.run(cmd_reencode, check=True)

    return output_path


def render_composite_video(
    muted_video_path: str,
    mastered_audio_path: str,
    ass_subtitle_path: Optional[str],
    output_final_path: str,
    canvas_dimension
) -> str:
    """Muxes mastered audio, applies web-compatible video encodings, and burns dynamic ASS captions."""
    _, ffmpeg_bin = EnvironmentValidator.check_ffmpeg()
    if not ffmpeg_bin:
        ffmpeg_bin = "ffmpeg"

    vf_filters = [
        f"scale={canvas_dimension.width}:{canvas_dimension.height}:force_original_aspect_ratio=decrease",
        f"pad={canvas_dimension.width}:{canvas_dimension.height}:(ow-iw)/2:(oh-ih)/2:black"
    ]

    if ass_subtitle_path and os.path.exists(ass_subtitle_path):
        escaped_ass = ass_subtitle_path.replace("\\", "/").replace(":", "\\:")
        vf_filters.append(f"subtitles='{escaped_ass}'")

    filter_chain = ",".join(vf_filters)

    # Added -pix_fmt yuv420p and -movflags +faststart to solve black video rendering
    cmd = [
        ffmpeg_bin, "-y",
        "-i", muted_video_path,
        "-i", mastered_audio_path,
        "-vf", filter_chain,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_final_path
    ]

    logger.info(f"Executing final FFmpeg render pipeline for '{output_final_path}'...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg render pipeline failed: {result.stderr}")

    return output_final_path


# ==============================================================================
# 2. STREAMLIT USER INTERFACE LAYOUT
# ==============================================================================
def main():
    st.title("🎬 ViewMax Studio Pro")
    st.caption("Automated Short-Form Video Repurposing & Dynamic Kinetic Typography Engine")

    diag_report = EnvironmentValidator.run_full_diagnostics(global_config.paths)

    # --------------------------------------------------------------------------
    # SIDEBAR: CONFIGURATION & CREDENTIALS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ System Credentials & Settings")
        
        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Required for GPT-4o narrative scripting and OpenAI TTS-1-HD."
        )
        
        elevenlabs_key_input = st.text_input(
            "ElevenLabs API Key",
            type="password",
            value=os.getenv("ELEVENLABS_API_KEY", ""),
            help="Optional high-fidelity voice synthesis engine."
        )

        global_config.update_api_keys(
            openai_key=openai_key_input,
            elevenlabs_key=elevenlabs_key_input
        )

        st.divider()
        st.header("🎨 Rendering Options")

        aspect_choice = st.selectbox(
            "Target Aspect Ratio",
            options=[e.value for e in AspectRatioEnum],
            index=0
        )
        selected_canvas = CANVAS_PRESETS[aspect_choice]

        style_choice = st.selectbox(
            "Caption Typography Style",
            options=[e.value for e in CaptionStyleEnum],
            index=0
        )
        selected_style = TYPOGRAPHY_PRESETS[style_choice]

        speaker_choice_key = st.selectbox(
            "VoicePersona Model",
            options=list(BUILTIN_SPEAKERS.keys()),
            format_func=lambda k: BUILTIN_SPEAKERS[k].display_name,
            index=0
        )

        st.divider()
        st.header("🎛️ Audio Processing & DSP")
        enable_dsp = st.checkbox("Enable Broadcast Vocal DSP", value=True)
        enable_bgm_ducking = st.checkbox("Enable Dynamic BGM Ducking", value=False)

    # --------------------------------------------------------------------------
    # MAIN WORKSPACE: CONTENT PROCESSING PIPELINE
    # --------------------------------------------------------------------------
    st.subheader("1. Source Video & Narrative Input")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        youtube_url = st.text_input(
            "YouTube Shorts URL",
            placeholder="https://www.youtube.com/shorts/..."
        )
        uploaded_file = st.file_uploader("Or Upload Video File (MP4/MOV)", type=["mp4", "mov"])

    with col2:
        script_text = st.text_area(
            "Narration Script",
            height=150,
            placeholder="Type or paste the new narration script here. The original video audio will be stripped and replaced with this speech and kinetic captions."
        )

    bgm_file = None
    if enable_bgm_ducking:
        bgm_file = st.file_uploader("Upload Optional Background Music (WAV/MP3)", type=["wav", "mp3"])

    if script_text and global_config.safety.strict_mode:
        lowered_script = script_text.lower()
        forbidden_matches = [kw for kw in global_config.safety.excluded_keywords if kw in lowered_script]
        if forbidden_matches:
            st.error(f"⚠️ Script violates target niche scope. Contains restricted topics: {', '.join(forbidden_matches)}")

    st.divider()

    # --------------------------------------------------------------------------
    # PIPELINE EXECUTION
    # --------------------------------------------------------------------------
    if st.button("🚀 Render Repurposed Video", type="primary", use_container_width=True):
        if not script_text.strip():
            st.warning("Please enter a valid narration script before starting the pipeline.")
            return

        if not youtube_url and not uploaded_file:
            st.warning("Please provide either a YouTube Shorts URL or upload a local video file.")
            return

        temp_dir = tempfile.mkdtemp(dir=global_config.paths.temp_dir)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Step 1/6: Processing source video input...")
            progress_bar.progress(15)

            if youtube_url:
                raw_video_path = download_youtube_video(youtube_url, temp_dir)
            else:
                raw_video_path = os.path.join(temp_dir, "uploaded_input.mp4")
                with open(raw_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            status_text.text("Step 2/6: Stripping original audio track...")
            progress_bar.progress(30)
            muted_video_path = os.path.join(temp_dir, "muted_footage.mp4")
            strip_audio_from_video(raw_video_path, muted_video_path)

            status_text.text("Step 3/6: Synthesizing AI voiceover speech...")
            progress_bar.progress(45)
            voice_mgr = VoiceSynthesisManager(global_config)
            raw_speech_path = os.path.join(temp_dir, "raw_speech.wav")
            synthesis_res = voice_mgr.generate_narration(
                text_script=script_text,
                primary_speaker_key=speaker_choice_key,
                output_wav_path=raw_speech_path
            )

            status_text.text("Step 4/6: Applying vocal mastering and DSP dynamic processing...")
            progress_bar.progress(60)
            mastered_speech_path = os.path.join(temp_dir, "mastered_speech.wav")
            
            if enable_dsp:
                dsp_proc = VocalMasteringProcessor(global_config.audio)
                dsp_proc.process_voice_chain(raw_speech_path, mastered_speech_path)
            else:
                shutil.copy(raw_speech_path, mastered_speech_path)

            final_audio_track = mastered_speech_path

            if enable_bgm_ducking and bgm_file:
                bgm_input_path = os.path.join(temp_dir, "bgm_input.wav")
                with open(bgm_input_path, "wb") as f:
                    f.write(bgm_file.getbuffer())

                mixed_audio_path = os.path.join(temp_dir, "mixed_master.wav")
                ducker = SidechainMusicDucker(global_config.audio)
                ducker.mix_bgm_with_sidechain(mastered_speech_path, bgm_input_path, mixed_audio_path)
                final_audio_track = mixed_audio_path

            status_text.text("Step 5/6: Extracting word timestamps & rendering kinetic typography...")
            progress_bar.progress(75)
            
            aligner = WhisperAlignmentEngine(global_config.whisper)
            word_timings = aligner.transcribe_and_align(mastered_speech_path, reference_text=script_text)
            chunks = aligner.build_caption_chunks(word_timings, chunk_size=selected_style.word_chunk_size)

            ass_path = os.path.join(temp_dir, "kinetic_subtitles.ass")
            ASSSubtitleBuilder.generate_ass_script(
                chunks=chunks,
                style=selected_style,
                canvas=selected_canvas,
                output_ass_path=ass_path
            )

            status_text.text("Step 6/6: Rendering composite video and burning dynamic captions...")
            progress_bar.progress(90)
            
            export_filename = f"viewmax_render_{int(time.time())}.mp4"
            final_export_path = os.path.join(global_config.paths.exports_dir, export_filename)

            render_composite_video(
                muted_video_path=muted_video_path,
                mastered_audio_path=final_audio_track,
                ass_subtitle_path=ass_path,
                output_final_path=final_export_path,
                canvas_dimension=selected_canvas
            )

            progress_bar.progress(100)
            status_text.text("🎉 Processing complete! Render ready.")

            st.success(f"Video rendered successfully! Exported to: `{final_export_path}`")
            
            st.subheader("📺 Video Preview")
            st.video(final_export_path)

            with open(final_export_path, "rb") as video_bytes:
                st.download_button(
                    label="💾 Download Rendered Video MP4",
                    data=video_bytes,
                    file_name=export_filename,
                    mime="video/mp4"
                )

        except Exception as e:
            logger.critical(f"Pipeline execution encountered fatal error: {e}", exc_info=True)
            st.error(f"❌ Pipeline Failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Pipeline execution halted due to errors.")


if __name__ == "__main__":
    main()
