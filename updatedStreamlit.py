import io
import json
import re
import time
import tempfile
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError 
# from PIL import Image, ImageDraw, ImageFont # Keep imports, but not used for video

# --- Agent Configuration ---

MODEL_ID = "gemini-robotics-er-1.5-preview" 

# Prompt for Structured Analysis (JSON Output)
DETECTION_PROMPT = """
You are a precision vision-language model for robotics. Analyze the video and provide the location of key objects within a frame that is most relevant to executing a physical task. Point to no more than 10 items. The label returned should be an identifying name for the object detected. The answer must follow the json format: [{"point": <point>, "label": <label>}]. The points are in [y, x] format normalized to 0-1000.
"""

# Prompt for Embodied Reasoning Analysis (Natural Language Output)
DESCRIPTION_PROMPT = """
You are a state-of-the-art embodied reasoning model for a robotic system. Your primary task is to analyze the provided video input and provide a thorough, natural language description of the scene. Focus on key objects, their spatial relationships, affordances (what actions can be performed on them), and the overall context of the environment. Provide a logical plan (in list format) for a robot to achieve the goal implied by the user's prompt. Respond only with the descriptive text and the plan; do not include any JSON or code blocks.
"""

# --- Streamlit Setup ---

st.set_page_config(page_title="Gemini Video Agent", layout="wide")
st.title("📹 Gemini Robotics-ER 1.5 Agent Demo")
st.caption(f"Using model: `{MODEL_ID}` for specialized embodied reasoning.")

# --- Utility Functions ---

@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    """Initializes and returns the Gemini API client."""
    # Ensure the API key is available in Streamlit Secrets or Environment Variables
    if "GEMINI_API_KEY" not in st.secrets:
         st.error("🚨 Gemini API Key not found. Please set it in your Streamlit secrets.")
         st.stop()
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def extract_json_payload(raw_text: str) -> str:
    """Extracts a JSON array string from the raw model text, handling code blocks."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Model returned an empty response.")

    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    if text.startswith("[") and text.endswith("]"):
        return text
        
    list_match = re.search(r'(\[.*?\])', text, re.DOTALL)
    if list_match:
        return list_match.group(1).strip()

    raise ValueError(f"Could not extract JSON payload from model response: {raw_text[:100]}...")


def upload_and_process_video(client: genai.Client, video_file: io.BytesIO) -> Optional[types.File]:
    """Uploads the video file to the Gemini API File Service and waits for processing."""
    
    # Use a temporary file to save the Streamlit BytesIO content
    ext = video_file.name.split('.')[-1] if '.' in video_file.name else 'mp4'
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_file:
            tmp_file.write(video_file.getvalue())
            temp_file_path = tmp_file.name
        
        with st.spinner(f"Uploading and processing **{video_file.name}**... (Large files may take time)"):
            st.info("File upload started...")
            uploaded_file = client.files.upload(file=temp_file_path)
            st.info(f"File uploaded. State: {uploaded_file.state.name}. Waiting for processing...")

            # Polling loop to wait for the file to be processed
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(5) # Wait 5 seconds before polling again
                uploaded_file = client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                st.error(f"Video processing failed on Gemini service: {uploaded_file.name}")
                return None
            
            st.success(f"Video ready for analysis: {uploaded_file.name}")
            return uploaded_file
    
    except APIError as e:
        st.error(f"Gemini API Error during file upload/processing: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- Streamlit Frontend with Model Integration ---

def main():
    client = get_client()

    # --- Video Upload Section ---
    st.header("Upload Your Video 🎥")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"]
    )
    
    if 'gemini_video_file' not in st.session_state:
        st.session_state.gemini_video_file = None

    if uploaded_video is not None:
        
        # Logic to handle re-uploading a new video
        if st.session_state.gemini_video_file is None or st.session_state.gemini_video_file.display_name != uploaded_video.name:
            # Delete previous file if it exists
            if st.session_state.gemini_video_file:
                try:
                    client.files.delete(name=st.session_state.gemini_video_file.name)
                except Exception:
                    # Ignore deletion errors if the file has already expired
                    pass
            
            # Upload the new file and store the reference
            st.session_state.gemini_video_file = upload_and_process_video(client, uploaded_video)
        
        st.video(uploaded_video)
        
        if st.session_state.gemini_video_file is not None:
            gemini_file = st.session_state.gemini_video_file
            st.info(f"File URI: {gemini_file.uri}")

            # --- Prompt Input Section ---
            st.header("Enter Your Prompt and Analysis Mode 🤖")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                user_prompt = st.text_area(
                    "Prompt:",
                    placeholder="E.g., 'What steps are needed to move the blue marker to the yellow tray?'",
                    key="video_prompt_input"
                )
            with col2:
                analysis_mode = st.radio(
                    "Analysis Mode:",
                    ("Embodied Reasoning", "Structured Detection"),
                    key="analysis_mode"
                )
            
            if analysis_mode == "Structured Detection":
                system_instruction = DETECTION_PROMPT
            else:
                system_instruction = DESCRIPTION_PROMPT
            
            # **--- MODEL INTEGRATION HERE ---**
            if st.button("Run Gemini Analysis", use_container_width=True):
                if user_prompt:
                    with st.spinner(f"Running **{analysis_mode}** on **{uploaded_video.name}**..."):
                        
                        try:
                            # 1. Prepare contents: [user text, Gemini File object]
                            contents = [user_prompt, gemini_file]
                            
                            # 2. Prepare configuration with system instruction
                            config = types.GenerateContentConfig(
                                system_instruction=system_instruction
                                # You can optionally add thinking_config here for ER model
                                # thinking_config=types.ThinkingConfig(thinking_budget=10) 
                            )

                            # 3. API Call to the specialized model
                            response = client.models.generate_content(
                                model=MODEL_ID,
                                contents=contents,
                                config=config
                            )
                            
                            st.write("---")
                            st.subheader("✅ Analysis Result")
                            
                            if analysis_mode == "Structured Detection":
                                # Handle JSON output for robotics control
                                try:
                                    raw_json = extract_json_payload(response.text)
                                    st.json(json.loads(raw_json))
                                    st.code(raw_json, language='json')
                                except ValueError as ve:
                                    st.warning(f"Could not parse model output as JSON. Displaying raw text: {ve}")
                                    st.markdown(response.text)
                            else:
                                # Handle natural language output for reasoning/planning
                                st.markdown(response.text)

                        except APIError as e:
                            st.error(f"Gemini API Error during generation: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                            st.code(response.text if 'response' in locals() else "No model response available.")

                else:
                    st.warning("Please enter a prompt before running the analysis.")
        
        else:
            st.error("Cannot run analysis because the video file failed to process on the Gemini service.")

    else:
        st.info("Upload a video file to begin analysis.")
        
if __name__ == "__main__":
    main()
