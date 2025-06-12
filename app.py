
# import cv2
# from cvzone.HandTrackingModule import HandDetector

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=2)  # Detect up to 2 hands

# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)

#     # Optional: Access info for both hands
#     if hands:
#         for i, hand in enumerate(hands):
#             lmList = hand["lmList"]      # List of 21 landmarks
#             bbox = hand["bbox"]          # Bounding box [x, y, w, h]
#             center = hand["center"]      # Center of hand
#             handType = hand["type"]      # 'Left' or 'Right'

#             print(f"Hand {i+1}: {handType}, Center: {center}")

#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Set up webcam and hand detector
video_capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=2)  # Detect up to 2 hands
padding = 20  # Padding around the hand bounding box
image_size = 300  # Size to resize the image
save_folder = "Data/H"  # Folder to save the captured images
image_counter = 0  # Counter to track saved images

while True:
    # Capture frame from webcam
    success, frame = video_capture.read()

    if not success:
        print("Failed to grab frame")
        continue

    # Detect hands in the current frame
    hands, frame = hand_detector.findHands(frame)

    if hands:
        height, width, _ = frame.shape

        for hand_index, hand in enumerate(hands):
            x, y, w, h = hand['bbox']  # Bounding box coordinates for the hand

            # Create a white background to center the resized hand image
            white_background = np.ones((image_size, image_size, 3), np.uint8) * 255

            # Crop the hand from the frame with the specified padding
            cropped_hand = frame[y - padding:y + h + padding, x - padding:x + w + padding]

            if cropped_hand.size == 0:  # Check if the cropped image is empty
                print(f"Invalid crop for Hand {hand_index + 1}. Skipping.")
                continue

            # Calculate the aspect ratio of the hand
            aspect_ratio = h / w

            # If the hand is taller than wide, adjust the width accordingly
            if aspect_ratio > 1:
                resize_scale = image_size / h
                resized_width = math.ceil(resize_scale * w)
                resized_hand = cv2.resize(cropped_hand, (resized_width, image_size))
                width_gap = math.ceil((image_size - resized_width) / 2)  # Horizontal gap for centering
                white_background[:, width_gap:resized_width + width_gap] = resized_hand  # Center the resized image
            else:
                resize_scale = image_size / w
                resized_height = math.ceil(resize_scale * h)
                resized_hand = cv2.resize(cropped_hand, (image_size, resized_height))
                height_gap = math.ceil((image_size - resized_height) / 2)  # Vertical gap for centering
                white_background[height_gap:resized_height + height_gap, :] = resized_hand  # Center the resized image

            # Display the cropped hand and the padded image
            cv2.imshow(f"Hand {hand_index + 1} Crop", cropped_hand)
            cv2.imshow(f"Hand {hand_index + 1} Padded", white_background)

    # Display the original webcam frame
    cv2.imshow("Webcam", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # Save the image when 's' key is pressed
    if key == ord("s"):
        image_counter += 1
        # Save the image in the specified folder
        cv2.imwrite(f'{save_folder}/Image_{time.time()}.jpg', white_background)
        print(f"Image saved! Total saved images: {image_counter}")

    # Exit the loop when 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()


