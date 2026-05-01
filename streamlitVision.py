import io
import json
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont


# --- Agent Configuration ---

MODEL_ID = "gemini-robotics-er-1.5-preview"
MAX_MODEL_WIDTH = 1024 # Standard resizing for model input

# Prompt for Object Detection (Structured JSON Output)
DETECTION_PROMPT = """
Point to no more than 10 items in the image. The label returned should be an identifying name for the object detected. The answer should follow the json format: [{"point": <point>, "label": <label>}]. The points are in [y, x] format normalized to 0-1000.
"""

# Prompt for Descriptive Analysis (Natural Language Output)
DESCRIPTION_PROMPT = """
You are a highly detailed visual analysis assistant for a robotic system. Your primary task is to analyze the provided image/video input and provide a thorough, natural language description of the scene. Focus on key objects, their spatial relationships, physical state, and the overall context of the environment. Respond only with the descriptive text; do not include any JSON or code blocks.
"""

# --- Streamlit Setup ---

st.set_page_config(page_title="Gemini Vision Agent", layout="wide")
st.title("Gemini Robotics Vision Agent Demo")
st.caption(f"Using model: `{MODEL_ID}` for structured detection or descriptive analysis.")


# --- Utility Functions ---

@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client:
    """Initializes and returns the Gemini API client."""
    # The API key is assumed to be handled by the Canvas environment or ENV variables
    return genai.Client()


def load_image(image_source: io.BytesIO | str) -> Image.Image:
    """Loads and resizes the image if it exceeds MAX_MODEL_WIDTH."""
    image = Image.open(image_source).convert("RGB")
    if image.width > MAX_MODEL_WIDTH:
        scale = MAX_MODEL_WIDTH / image.width
        new_size = (MAX_MODEL_WIDTH, int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def extract_json_payload(raw_text: str) -> str:
    """Extracts a JSON array string from the raw model text, handling code blocks."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Model returned an empty response.")

    # Search for code blocks (```json[...]``` or just ```[...]```)
    #code_block_match = re.search(r"