from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
import boto3
import requests
from flask import Flask, request, jsonify
import os
import json
import numpy as np
from datetime import datetime
import torch
import cv2
from ultralytics import YOLO
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from pydub import AudioSegment  # For sound file processing
from speech_recognition import Recognizer, AudioFile # For speech-to-text from pydub import AudioSegment # For audio file handling
import requests
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
import boto3



# Flask setup
app = Flask(__name__)

# Directories for uploads and predictions
UPLOAD_FOLDER = 'uploaded_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PREDICTIONS_FOLDER = 'video_predictions'
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Device setup for models
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load models
detector = YOLO("yolov8n.pt")  # YOLO model
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)  # MiDaS model
depth_processor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

# Bedrock setup
os.environ["AWS_PROFILE"] = "hwiiyy"
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your AWS region
)
modelID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
llm = ChatBedrock(
    model_id=modelID,
    client=bedrock_client
)
prompt = PromptTemplate(
    input_variables=["warnings"],
    template=(
        """
        You are a friendly assistant guiding users to avoid obstacles.
        Based on the input warnings, generate a helpful and natural-language message for the user.

        Warnings: {warnings}
        """
    )
)

# Helper functions
import pyttsx3
from concurrent.futures import ThreadPoolExecutor
def text_to_speech(text, output_path="output.wav"):
    """
    Converts text to speech and saves it as a .wav file using pyttsx3 in a separate thread.
    """
    def generate_speech():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)  # Words per minute
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}")

    # Run the TTS in a separate thread
    with ThreadPoolExecutor() as executor:
        future = executor.submit(generate_speech)
        future.result()  # Wait for the TTS generation to complete

    return output_path



def run_inference(image):
    detection_results = detector(image, verbose=False)
    yolo_result = detection_results[0]

    detected_objects = []
    for box in yolo_result.boxes:
        cls_id = int(box.cls[0])
        cls_name = detector.names[cls_id]
        confidence = round(float(box.conf[0]) * 100, 1)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        detected_objects.append({
            "class": cls_name,
            "confidence": confidence,
            "bounding_box": [x1, y1, x2, y2]
        })

    inputs = depth_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        depth_output = depth_model(**inputs).predicted_depth[0].cpu().numpy()

    depth_normalized = (depth_output - depth_output.min()) / (depth_output.max() - depth_output.min())
    depth_colored = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bounding_box"]
        h, w = depth_output.shape
        x1_clamped = max(0, min(w, x1))
        x2_clamped = max(0, min(w, x2))
        y1_clamped = max(0, min(h, y1))
        y2_clamped = max(0, min(h, y2))
        object_depth_region = depth_output[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        if object_depth_region.size > 0:
            obj["distance"] = round(float(np.mean(object_depth_region)), 2)
        else:
            obj["distance"] = None

    return detected_objects, depth_colored

def process_obstacle_data(objects):
    warnings = []
    for obj in objects:
        cls_name = obj.get("class", "unknown")
        distance = obj.get("distance", None)
        if distance is not None:
            if distance < 2.0:
                warnings.append(f"Warning! A {cls_name} is {distance:.1f} meters away.")
            elif distance < 5.0:
                warnings.append(f"Notice: A {cls_name} is {distance:.1f} meters ahead.")
        else:
            warnings.append(f"Notice: Detected a {cls_name} but could not estimate the distance.")
    return warnings

def generate_guidance(warnings):
    try:
        bedrock_chain = prompt | llm
        response = bedrock_chain.invoke({'warnings': warnings})
        return response.content
    except Exception as e:
        return f"Error generating guidance: {str(e)}"

# Video processing logic
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every nth frame (e.g., every 2 seconds assuming 30 FPS)
        if frame_count % (2 * 30) == 0:
            detected_objects, depth_colored = run_inference(frame)

            # Generate warnings and guidance
            warnings = process_obstacle_data(detected_objects)
            guidance = generate_guidance(warnings)

            # Save or log the results
            print(f"Frame {frame_count}: {guidance}")

    cap.release()


import threading
# Flask endpoint
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the video and generate guidance
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        all_warnings = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every nth frame (e.g., every 2 seconds assuming 30 FPS)
            if frame_count % (2 * 30) == 0:
                detected_objects, depth_colored = run_inference(frame)

                # Generate warnings for the current frame
                warnings = process_obstacle_data(detected_objects)
                all_warnings.extend(warnings)

        cap.release()

        # Generate text guidance based on accumulated warnings
        guidance_text = generate_guidance(all_warnings)

        # Convert the guidance text to speech
        wav_file_path = text_to_speech(guidance_text, output_path=os.path.join(PREDICTIONS_FOLDER, "output.wav"))

        # Return the audio file as a response
        with open(wav_file_path, 'rb') as f:
            wav_content = f.read()

        return (wav_content, 200, {
            'Content-Type': 'audio/wav',
            'Content-Disposition': f'attachment; filename="output.wav"'
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process video: {str(e)}'}), 500


def recover_text_with_claude(text_lines):
    """
    Uses a Claude model to recover and make sense of the extracted text.
    """
    try:
        # Prepare the prompt for Claude
        recovery_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                """
                You are an expert in deciphering and recovering fragmented or noisy text.
                Analyze the provided text carefully and try to make sense of it, reconstructing it
                into coherent sentences where possible.

                Text: {text}
                """
            )
        )

        # Combine text lines into a single input for the prompt
        input_text = "\n".join(text_lines)

        # Create the Claude chain
        bedrock_chain = recovery_prompt | llm
        response = bedrock_chain.invoke({'text': input_text})

        # Return the recovered text
        return response.content
    except Exception as e:
        raise RuntimeError(f"Error recovering text with Claude: {str(e)}")



@app.route('/upload_sound', methods=['POST'])
def upload_sound():
    """
    Processes an uploaded sound file:
    1. Performs speech-to-text transcription.
    2. Uses Claude to recover meaningful guidance from the text.
    3. Converts the guidance into a .wav audio response.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = None

    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load the WAV file and extract metadata
        sound = AudioSegment.from_file(file_path, format="wav")
        duration = len(sound) / 1000  # Duration in seconds
        channels = sound.channels
        sample_width = sound.sample_width
        frame_rate = sound.frame_rate

        # Perform speech-to-text transcription
        recognizer = Recognizer()
        with AudioFile(file_path) as audio_file:
            audio_data = recognizer.record(audio_file)  # Load audio data
            try:
                transcribed_text = recognizer.recognize_google(audio_data)  # Transcribe audio
            except Exception as e:
                transcribed_text = f"Speech recognition failed: {e}"
        os.remove(file_path)
        # Step 1: Use Claude to recover and refine guidance text
        try:
            refined_guidance = recover_text_with_claude([transcribed_text])
        except Exception as e:
            refined_guidance = f"Claude processing failed: {e}"

        # Step 2: Convert guidance text to speech using TTS
        audio_output_path = text_to_speech(refined_guidance)

        # Step 3: Return the audio response as a .wav file
        with open(audio_output_path, 'rb') as f:
            wav_content = f.read()
        return (wav_content, 200, {
            'Content-Type':'audio/wav',
            'Content-Disposition':f'attachment; filename="output.wav"'
        })

    except Exception as e:
        # Cleanup files in case of an error
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({'error': f'Error processing sound file: {str(e)}'}), 500


@app.route('/extract_text', methods=['POST'])
def extract_text_and_audio():
    """
    API endpoint to:
    1. Extract text from the latest image in the uploaded directory.
    2. Use Claude to recover and make sense of the text.
    3. Convert the recovered text into a .wav audio file.
    """
    # Listen for the 'process' boolean value
    data = request.get_json()
    if not data or 'process' not in data:
        return jsonify({'error': 'Missing "process" key in the request body'}), 400

    process = data['process']

    if process:
        try:
            # Directory to look for the latest image
            directory = UPLOAD_FOLDER

            # Find the latest image in the directory
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if not files:
                return jsonify({'error': 'No images found in the directory'}), 404

            latest_image_path = max(files, key=os.path.getmtime)

            # Step 1: Perform OCR on the latest image
            preprocessed_image = preprocess_image(latest_image_path)
            results = ocr.ocr(preprocessed_image, det=True, rec=True, cls=True)
            raw_text_lines = [line[1][0] for line in results[0]]

            # Step 2: Recover text using Claude
            recovered_text = recover_text_with_claude(raw_text_lines)

            # Step 3: Convert recovered text to speech
            audio_file_path = text_to_speech(recovered_text)

            # Step 4: Return the audio file
            return send_file(audio_file_path, as_attachment=True, download_name="output.wav", mimetype="audio/wav")

        except Exception as e:
            return jsonify({'error': f'Failed to process request: {str(e)}'}), 500

    else:
        return jsonify({'message': 'Boolean value is False, no action performed'}), 200

@app.route('/upload_coordination', methods=['POST'])
def upload_coordination():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        # Check if latitude and longitude are provided
        if latitude is None or longitude is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400

        # Define destination (replace with your logic to fetch the destination)
        # Example: Predefined destination
        destination = "Eiffel Tower, Paris, France"

        # Google Maps API endpoint and key
        google_maps_endpoint = "https://maps.googleapis.com/maps/api/directions/json"
        google_api_key = "your_google_api_key"

        # Parameters for the Google Maps Directions API
        params = {
            "origin": f"{latitude},{longitude}",
            "destination": destination,
            "key": google_api_key
        }

        # Make the request to Google Maps API
        response = requests.get(google_maps_endpoint, params=params)

        # If the request to Google API fails
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch directions from Google Maps API'}), 500

        directions = response.json()

        # Parse and return the directions
        if directions.get("status") == "OK":
            route = directions["routes"][0]["legs"][0]
            steps = [
                {
                    "distance": step["distance"]["text"],
                    "duration": step["duration"]["text"],
                    "instruction": step["html_instructions"]
                }
                for step in route["steps"]
            ]
            return jsonify({
                "start_address": route["start_address"],
                "end_address": route["end_address"],
                "distance": route["distance"]["text"],
                "duration": route["duration"]["text"],
                "steps": steps
            }), 200
        else:
            return jsonify({'error': directions.get("error_message", "Unknown error")}), 500

    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"}), 500


# --------------------
# MAIN
# --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
