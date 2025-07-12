import time
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response
import paho.mqtt.client as mqtt
import threading

# MQTT Configuration
MQTT_BROKER = "7031e8491d334b88b475025af999ffeb.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "esp32cam"
MQTT_PASSWORD = "Bracketservo2"
MQTT_TOPIC_PAN_CMD = "esp32/commands/pan"
MQTT_TOPIC_TILT_CMD = "esp32/commands/tilt"

# Setup Flask
app = Flask(__name__)

# Setup MQTT Client
client = mqtt.Client()
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Connect to the MQTT Broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

client.on_connect = on_connect
client.tls_set()  # Use TLS encryption
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Start the MQTT client loop in a separate thread
client.loop_start()

# Initialize MediaPipe Face Mesh for face tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to detect head position and landmarks using MediaPipe
def detect_head_position(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None  # No face detected

    # Assuming there's one face, use the first detected face
    face_landmarks = results.multi_face_landmarks[0]
    
    # Get the position of landmarks for the nose and eyes
    nose_tip = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[133]

    # Get the center between the eyes for pan calculation
    pan_center_x = (left_eye.x + right_eye.x) / 2
    pan_center_y = (left_eye.y + right_eye.y) / 2

    # Map the pan_center position to the actual screen coordinates
    frame_height, frame_width, _ = frame.shape
    pan_center_x = int(pan_center_x * frame_width)
    pan_center_y = int(pan_center_y * frame_height)

    return pan_center_x, pan_center_y

# Smooth movement function: gradually move the servo
def smooth_move(prev_value, target_value, smoothing_factor=0.5):
    """
    Apply smooth interpolation between current position and target position.
    Smoothing factor determines how slowly the servo will reach the target value.
    A higher value makes the movement smoother but slower.
    """
    return prev_value + (target_value - prev_value) * smoothing_factor

# Function to map pan/tilt values and add a threshold for smoother movement
def map_to_pan_tilt(pan_center_x, pan_center_y, frame_width, frame_height, prev_pan, prev_tilt):
    # Convert to servo angles
    pan = np.interp(pan_center_x, [0, frame_width], [0, 180])
    tilt = np.interp(pan_center_y, [0, frame_height], [0, 180])

    # Apply smooth movement to pan and tilt
    pan = smooth_move(prev_pan, pan)
    tilt = smooth_move(prev_tilt, tilt)

    # Apply threshold: only send command if change is larger than threshold
    threshold = 5  # Only move servo if change is greater than 5 degrees
    if abs(pan - prev_pan) < threshold:
        pan = prev_pan
    if abs(tilt - prev_tilt) < threshold:
        tilt = prev_tilt

    # Optionally add a dynamic deadzone based on the distance from center
    pan, tilt = dynamic_deadzone(pan, tilt)

    return int(pan), int(tilt)

def dynamic_deadzone(pan, tilt, deadzone_range=20):
    """
    Adjust the deadzone dynamically based on how close the servo is to center position.
    The closer to 90°, the smaller the deadzone.
    """
    if abs(pan - 90) < deadzone_range:
        pan = 90
    if abs(tilt - 90) < deadzone_range:
        tilt = 90
    return pan, tilt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    pan = request.form.get('pan')
    tilt = request.form.get('tilt')

    if pan:
        client.publish(MQTT_TOPIC_PAN_CMD, pan)
    if tilt:
        client.publish(MQTT_TOPIC_TILT_CMD, tilt)

    return jsonify({"status": "success", "pan": pan, "tilt": tilt})

# Function to stream video (MJPEG)
def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce the resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Background task to detect face and move servo
def face_tracking():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce the resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_pan, prev_tilt = 90, 90  # Initialize previous pan and tilt
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame to speed up the system
            continue

        # Detect head position using MediaPipe
        results = detect_head_position(frame)
        if results:
            pan_center_x, pan_center_y = results
            pan, tilt = map_to_pan_tilt(pan_center_x, pan_center_y, frame.shape[1], frame.shape[0], prev_pan, prev_tilt)

            # Publish pan and tilt values to MQTT
            client.publish(MQTT_TOPIC_PAN_CMD, pan)
            client.publish(MQTT_TOPIC_TILT_CMD, tilt)

            # Update previous pan and tilt values
            prev_pan, prev_tilt = pan, tilt

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_tracking_thread = threading.Thread(target=face_tracking)
    face_tracking_thread.daemon = True
    face_tracking_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
