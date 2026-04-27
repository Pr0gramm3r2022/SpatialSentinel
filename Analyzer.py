from google.adk.agents.llm_agent import Agent
from typing import List, Dict, Union

# Define the structured output format for an object detection result
ObjectDetectionResult = List[Dict[str, Union[str, List[int]]]]

# 1. New Tool for Visual Input
def capture_and_process_frame(camera_id: int) -> Dict[str, str]:
    """
    Simulates capturing a frame from the robot's camera.
    In a real system, this would trigger the camera hardware to grab an image
    and make it available to the VLM (e.g., via a path or byte stream).
    For this agent, we assume the initial multimodal input (the image/video)
    is passed directly to the model along with the prompt, and this tool
    is used to get a status update or specific metadata.
    """
    if camera_id == 1:
        return {"status": "SUCCESS", "frame_id": "F_4523", "resolution": "1080p"}
    return {"status": "ERROR", "message": f"Camera {camera_id} not found."}


# 2. Agent Definition with Robotics-ER 1.5 Focus
analyzer_agent = Agent(
    model='gemini-robotics-er-1.5',
    name='Object_Detection_Analyzer',
    description="Analyzes an image to detect, locate, and measure specified objects using embodied reasoning.",
    # The instruction now focuses on spatial reasoning and structured output
    instruction=(
        "You are an expert object detection and spatial reasoning agent for a robot. "
        "Your task is to analyze the provided image/video input and detect all instances "
        "of the target object specified in the user's query. "
        "You must return the results as a **JSON list** of bounding boxes in the format: "
        "`[{\"box_2d\": [y_min, x_min, y_max, x_max], \"label\": \"<object_name>\"}, ...]` "
        "where coordinates are **normalized to 0-1000** for easy robotic interpretation. "
        "Do not use the `capture_and_process_frame` tool unless specifically asked for camera status."
    ),
    tools=[capture_and_process_frame],
)

