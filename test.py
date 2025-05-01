import cv2
import math
import numpy as np
import time
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Constants
IMAGE_SIZE = 224       # Size expected by the model
PADDING = 20           # Padding around detected hand for cropping

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Initialize hand detector (max 2 hands)
hand_detector = HandDetector(maxHands=2)

# Load trained model (do not compile)
model = load_model("Model/keras_Model.h5", compile=False)

# Load class labels from labels.txt
with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] if ' ' in line else line.strip() for line in f.readlines()]

def preprocess_hand(cropped_hand, w, h):
    """
    Resize and pad the cropped hand image to IMAGE_SIZE x IMAGE_SIZE.
    Maintains the aspect ratio by padding with white background.
    """
    white_bg = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
    aspect_ratio = h / w

    if aspect_ratio > 1:
        scale = IMAGE_SIZE / h
        new_w = math.ceil(w * scale)
        resized_hand = cv2.resize(cropped_hand, (new_w, IMAGE_SIZE))
        gap = math.ceil((IMAGE_SIZE - new_w) / 2)
        white_bg[:, gap:gap + new_w] = resized_hand
    else:
        scale = IMAGE_SIZE / w
        new_h = math.ceil(h * scale)
        resized_hand = cv2.resize(cropped_hand, (IMAGE_SIZE, new_h))
        gap = math.ceil((IMAGE_SIZE - new_h) / 2)
        white_bg[gap:gap + new_h, :] = resized_hand

    return white_bg

def predict(image):
    """
    Preprocess image for model prediction, normalize and expand dimensions,
    return predicted label and confidence.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = (image_rgb.astype(np.float32) / 127.5) - 1.0
    input_data = np.expand_dims(normalized, axis=0)
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return labels[class_index], confidence

# Initialize time for FPS calculation
prev_time = time.time()

# Start capturing frames from webcam
while True:
    success, frame = video_capture.read()
    if not success:
        print("Failed to grab frame")
        continue

    img_output = frame.copy()

    # Detect hands in the frame
    hands, _ = hand_detector.findHands(frame)

    if hands:
        height, width, _ = frame.shape

        # Loop through all detected hands
        for idx, hand in enumerate(hands):
            x, y, w, h = hand['bbox']  # Bounding box for hand

            # Calculate coordinates with padding, keeping them within frame bounds
            x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
            x2, y2 = min(width, x + w + PADDING), min(height, y + h + PADDING)

            # Crop the hand region from the frame
            cropped = frame[y1:y2, x1:x2]

            # Skip if crop is empty
            if cropped.size == 0:
                print(f"Empty crop for hand {idx + 1}. Skipping.")
                continue

            # Preprocess the hand image
            processed_img = preprocess_hand(cropped, w, h)

            # Predict the hand gesture/class
            label, confidence = predict(processed_img)

            # Display label and confidence on the original frame
            cv2.putText(img_output, f"{label} ({confidence*100:.1f}%)", (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            cv2.rectangle(img_output, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Show the cropped and processed hand images
            cv2.imshow(f"Hand {idx + 1} Crop", cropped)
            cv2.imshow(f"Hand {idx + 1} Padded", processed_img)

    # Calculate and display FPS on the main frame
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img_output, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the final output with all overlays
    cv2.imshow("Webcam", img_output)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
