import cv2
import math
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Constants
IMAGE_SIZE = 224
PADDING = 20

# Flask app
app = Flask(__name__)

# Global Variables
video_capture = None
hand_detector = HandDetector(maxHands=2)
model = load_model("Model/keras_Model.h5", compile=False)

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] if ' ' in line else line.strip() for line in f.readlines()]

def preprocess_hand(cropped_hand, w, h):
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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = (image_rgb.astype(np.float32) / 127.5) - 1.0
    input_data = np.expand_dims(normalized, axis=0)
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return labels[class_index], confidence

def generate_frames():
    global video_capture
    while True:
        if video_capture is None:
            break
        success, frame = video_capture.read()
        if not success:
            break

        img_output = frame.copy()
        hands, _ = hand_detector.findHands(frame)

        if hands:
            height, width, _ = frame.shape
            for hand in hands:
                x, y, w, h = hand['bbox']
                x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
                x2, y2 = min(width, x + w + PADDING), min(height, y + h + PADDING)
                cropped = frame[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue

                processed_img = preprocess_hand(cropped, w, h)
                label, confidence = predict(processed_img)

                cv2.putText(img_output, f"{label} ({confidence*100:.1f}%)", (x, y - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                cv2.rectangle(img_output, (x1, y1), (x2, y2), (255, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', img_output)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            video_capture = None
            return jsonify(status="Failed to access camera")
        return jsonify(status="Camera started")
    return jsonify(status="Camera is already running")

@app.route('/stop_camera')
def stop_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        cv2.destroyAllWindows()
        video_capture = None
        return jsonify(status="Camera stopped")
    return jsonify(status="Camera is not running")

@app.route('/documentation')
def documentation():
    return render_template('about_sign_lang.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
