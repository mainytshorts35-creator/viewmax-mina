"""
================================================================================
VIEWMAX STUDIO PRO — STREAMLIT CLOUD ENTRYPOINT
FILE: app.py
DESCRIPTION: Main user interface and orchestration pipeline for automated 
             short-form video generation, voice synthesis, and kinetic captions.
================================================================================
"""

import os
import sys
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & PATH SETUP
# ==============================================================================
st.set_page_config(
    page_title="ViewMax Studio Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure current working directory is in system path for reliable module loading
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ==============================================================================
# 2. SAFE MODULE IMPORTS WITH ERROR BOUNDARIES
# ==============================================================================
try:
    from config import global_config, EnvironmentValidator, AspectRatioEnum, CaptionStyleEnum
    from voice_synthesis import VoiceSynthesisManager, BUILTIN_SPEAKERS
    from audio_dsp import VocalMasteringProcessor, SidechainMusicDucker
except ImportError as e:
    st.error("### 🚨 Critical Module Import Failure")
    st.markdown(f"Failed to load required backend modules. Please verify all project files (`config.py`, `voice_synthesis.py`, `audio_dsp.py`) are present in the repository root (`/mount/src/viewmax-mina/`).")
    st.exception(e)
    st.stop()


# ==============================================================================
# 3. MAIN APPLICATION UI & WORKFLOW ORCHESTRATION
# ==============================================================================
def main():
    st.title("🎬 ViewMax Studio Pro")
    st.markdown("### Enterprise-Grade AI Short-Form Video & Voice Automation Engine")

    # Sidebar Configuration & Credentials
    st.sidebar.header("🔑 API & Environment Settings")
    
    openai_key_input = st.sidebar.text_input("OpenAI API Key", type="password", value=global_config.credentials.openai_api_key)
    elevenlabs_key_input = st.sidebar.text_input("ElevenLabs API Key", type="password", value=global_config.credentials.elevenlabs_api_key)

    if openai_key_input or elevenlabs_key_input:
        global_config.update_api_keys(openai_key_input, elevenlabs_key_input)

    st.sidebar.divider()
    st.sidebar.header("⚙️ Output Settings")
    aspect_ratio = st.sidebar.selectbox("Canvas Preset", [e.value for e in AspectRatioEnum])
    speaker_choice = st.sidebar.selectbox("Voice Persona", list(BUILTIN_SPEAKERS.keys()), format_func=lambda x: BUILTIN_SPEAKERS[x].display_name)

    # Environment Diagnostics Expander
    with st.sidebar.expander("🛠️ System Diagnostics"):
        diag = EnvironmentValidator.run_full_diagnostics(global_config.paths)
        st.write(f"**FFmpeg Installed:** {diag['ffmpeg_installed']}")
        st.write(f"**FFmpeg Path:** {diag['ffmpeg_path']}")
        st.write(f"**Workspace Ready:** {diag['workspace_ready']}")

    # Main Content Area
    st.markdown("#### 📝 Enter Video Narration Script")
    script_text = st.text_area(
        "Script Content",
        placeholder="Type or paste your high-retention short script here...",
        height=150,
        value="Welcome to ViewMax Studio Pro. High retention kinetic video generation is fully initialized."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Generate Voice Narration", type="primary", use_container_width=True):
            if not script_text.strip():
                st.warning("Please enter a valid script before generating audio.")
            else:
                with st.spinner("Synthesizing broadcast-grade voiceover..."):
                    try:
                        synth_manager = VoiceSynthesisManager(global_config)
                        output_wav = os.path.join(global_config.paths.temp_dir, "narration.wav")
                        synth_manager.generate_narration(script_text, primary_speaker_key=speaker_choice, output_wav_path=output_wav)
                        
                        st.success("Voice narration generated successfully!")
                        st.audio(output_wav)
                        st.session_state["last_audio"] = output_wav
                    except Exception as ex:
                        st.error(f"Voice synthesis failed: {ex}")

    with col2:
        if st.button("🎛️ Apply Broadcast Audio Mastering", use_container_width=True):
            target_audio = st.session_state.get("last_audio")
            if not target_audio or not os.path.exists(target_audio):
                st.warning("Please generate or provide a voice narration track first.")
            else:
                with st.spinner("Applying vocal compression, EQ, and limiter..."):
                    try:
                        processor = VocalMasteringProcessor(global_config.audio)
                        mastered_wav = os.path.join(global_config.paths.temp_dir, "mastered_narration.wav")
                        processor.process_voice_chain(target_audio, mastered_wav)
                        
                        st.success("Broadcast mastering applied!")
                        st.audio(mastered_wav)
                    except Exception as ex:
                        st.error(f"Audio DSP mastering failed: {ex}")

    st.divider()
    st.info("System operational. Configure your API keys in the sidebar to begin rendering automated shorts.")


if __name__ == "__main__":
    main()
