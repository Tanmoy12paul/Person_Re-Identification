# real_time_reid_yolo.py

import cv2  # OpenCV for webcam and image operations
import os  # File operations
import torch  # PyTorch for deep learning
import numpy as np  # Numerical operations
from pathlib import Path  # For file path handling
from data_loader import load_and_preprocess  # Custom image preprocessing function
from feature_extractor import get_model, extract_features  # Load ReID model & extract embeddings
from similarity import compute_cosine_similarity  # Compare embeddings using cosine similarity

# === YOLOv5 Setup ===
# Load YOLOv5s model locally
yolo_model = torch.hub.load('yolov5', 'yolov5s', source='local')
yolo_model.conf = 0.5  # Set confidence threshold for detection
yolo_model.classes = [0]  # Only detect class 0: person (from COCO dataset)

# === ReID Model Setup ===
device = 'cpu'  # Change to 'cuda' if using a GPU
reid_model = get_model(device)  # Load modified ResNet50 model

# === Load Gallery Embeddings ===
gallery_path = "gallery"  # Folder with known person images
gallery_feats = []  # List to store gallery feature vectors
labels = []  # Labels for gallery identities

print("Loading gallery images...")
for img_name in os.listdir(gallery_path):
    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(gallery_path, img_name)  # Full path to image
        label = img_name.split('_')[0]  # Extract person name from filename
        image = load_and_preprocess(path)  # Preprocess the image
        feature = extract_features(reid_model, image, device=device)  # Get feature embedding
        gallery_feats.append(feature[0])  # Add to gallery features
        labels.append(label)  # Save corresponding label
print(f"✅ Loaded {len(labels)} gallery images.")

# === Start Real-Time Detection ===
cap = cv2.VideoCapture(0)  # Open webcam stream (0 = default camera)
print("🚀 Running YOLO + ReID... Press 'q' to quit.")

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Stop if no frame is received

    results = yolo_model(frame)  # Run YOLOv5 detection on the frame
    detections = results.xyxy[0]  # Get detection results (bounding boxes)

    for *box, conf, cls in detections:  # Loop over each detected person
        x1, y1, x2, y2 = map(int, box)  # Extract bounding box coordinates
        person_crop = frame[y1:y2, x1:x2]  # Crop the detected person from the frame

        # Skip tiny detections that are likely noise
        if person_crop.shape[0] < 50 or person_crop.shape[1] < 30:
            continue

        cv2.imwrite("temp_person.jpg", person_crop)  # Save crop temporarily for processing
        try:
            img = load_and_preprocess("temp_person.jpg")  # Preprocess cropped image
            query_feat = extract_features(reid_model, img, device=device)  # Get feature vector
            sims = compute_cosine_similarity([query_feat[0]], gallery_feats)[0]  # Compare to gallery
            top_idx = np.argmax(sims)  # Find the most similar gallery entry
            top_score = sims[top_idx]  # Get similarity score

            # Decide if match is valid based on threshold
            if top_score > 0.80:
                label = labels[top_idx]
                print(f"✅[MATCH] Detected: {label} with confidence {top_score:.2f}")
                color = (0, 255, 0)  # Green box for known identity
            else:
                label = "Unknown"
                print(f"❌[NO MATCH] Person not identified. Highest similarity: {top_score:.2f}")
                color = (0, 0, 255)  # Red box for unknown

        except Exception as e:
            label = "Error"
            print(f"[ERROR] Failed to process cropped image: {e}")
            color = (255, 0, 0)  # Blue box for error

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the updated frame in a window
    cv2.imshow("YOLO + ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quit loop if 'q' is pressed

cap.release()  # Release webcam resource
cv2.destroyAllWindows()  # Close OpenCV window
