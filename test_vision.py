import os
import json
import numpy as np
from ultralytics import YOLO
import torch
import cv2
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from datetime import datetime

# Check if GPU is available and being used
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load YOLOv8 for Object Detection
detector = YOLO("yolov8n.pt")  # Replace with yolov8s.pt for better accuracy

# Load MiDaS for Depth Estimation
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
depth_processor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

# Create the 'predictions' folder if it doesn't exist
os.makedirs("predictions", exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Timer variables for 2-second intervals
last_processed_time = datetime.now()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Calculate elapsed time since last processing
    elapsed_time = (datetime.now() - last_processed_time).total_seconds()

    # Only process every 2 seconds
    if elapsed_time >= 2:
        last_processed_time = datetime.now()

        # --------------------
        # YOLOv8 - Object Detection
        # --------------------
        detection_results = detector(resized_frame)
        detected_objects = []
        for r in detection_results[0].boxes:
            cls_id = int(r.cls[0])
            cls_name = detector.names[cls_id]
            confidence = round(float(r.conf[0]) * 100, 1)
            box = r.xyxy[0].cpu().numpy().astype(int).tolist()  # Convert bounding box to list
            detected_objects.append({
                "class": cls_name,
                "confidence": confidence,
                "bounding_box": box
            })

        # --------------------
        # MiDaS - Depth Estimation
        # --------------------
        inputs = depth_processor(images=resized_frame, return_tensors="pt").to(device)
        with torch.no_grad():
            depth_output = depth_model(**inputs).predicted_depth[0].cpu().numpy()

        # Normalize depth for visualization (optional)
        depth_normalized = (depth_output - depth_output.min()) / (depth_output.max() - depth_output.min())
        depth_colored = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)

        # --------------------
        # Combine YOLO and Depth
        # --------------------
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bounding_box"]
            # Extract depth values within the bounding box
            object_depth_region = depth_output[y1:y2, x1:x2]
            if object_depth_region.size > 0:
                # Calculate the average or minimum depth
                obj["distance"] = round(float(np.mean(object_depth_region)), 2)  # Average depth
            else:
                obj["distance"] = None  # No depth available (e.g., invalid box)

        # --------------------
        # Save Outputs
        # --------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestep_folder = f"predictions/timestep_{timestamp}"
        os.makedirs(timestep_folder, exist_ok=True)

        # Save JSON file
        with open(os.path.join(timestep_folder, "object_detection_with_depth.json"), "w") as f:
            json.dump({"objects": detected_objects}, f, indent=4)

        print(f"Saved outputs to {timestep_folder}")

        # --------------------
        # Display Outputs
        # --------------------
        # YOLO detection results
        cv2.imshow("Object Detection (YOLOv8)", detection_results[0].plot())
        # Depth map
        cv2.imshow("Depth Map (MiDaS)", depth_colored)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
