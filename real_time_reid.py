# real_time_reid.py

import cv2
import os
import numpy as np
from data_loader import load_and_preprocess
from feature_extractor import get_model, extract_features
from similarity import compute_cosine_similarity

device = 'cpu'  # Change to 'cuda' if using GPU

# Step 1: Load gallery images and extract features
gallery_path = "gallery"
gallery_feats = []
labels = []

print("Loading gallery images...")
model = get_model(device)

for img_name in os.listdir(gallery_path):
    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(gallery_path, img_name)
        label = img_name.split('_')[0]  # e.g., 'tanmoy_1.jpg' → 'tanmoy'
        image = load_and_preprocess(path)
        feature = extract_features(model, image, device=device)
        gallery_feats.append(feature[0])
        labels.append(label)

print(f"Loaded {len(labels)} gallery images.")

# Step 2: Start webcam stream
cap = cv2.VideoCapture(0)
print("✅Starting real-time ReID. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite("temp.jpg", frame)

    try:
        image = load_and_preprocess("temp.jpg")
        query_feat = extract_features(model, image, device=device)

        sims = compute_cosine_similarity([query_feat[0]], gallery_feats)[0]
        top_idx = np.argmax(sims)
        top_score = sims[top_idx]

        # Threshold to determine known vs unknown
        if top_score > 0.8:
            label = f"{labels[top_idx]} ({top_score:.2f})"
            color = (0, 255, 0)  # Green
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    except Exception as e:
        cv2.putText(frame, "Processing error", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-Time ReID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
