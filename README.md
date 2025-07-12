                                                                import time
                                                                import cv2
                                                                import numpy as np
                                                                import mediapipe as mp
                                                                from flask import Flask, render_template, request, jsonify, Response
                                                                import paho.mqtt.client as mqtt
                                                                import threading
                                                                import requests
                                                                
                                                                # === KONFIGURASI ===
                                                                ESP32_CAM_URL = "http://192.168.1.200/stream"  # GANTI DENGAN IP ESP32-CAM
                                                                
                                                                # MQTT
                                                                MQTT_BROKER = "7031e8491d334b88b475025af999ffeb.s1.eu.hivemq.cloud"
                                                                MQTT_PORT = 8883
                                                                MQTT_USERNAME = "esp32cam"
                                                                MQTT_PASSWORD = "Bracketservo2"
                                                                MQTT_TOPIC_PAN_CMD = "esp32/commands/pan"
                                                                MQTT_TOPIC_TILT_CMD = "esp32/commands/tilt"
                                                                
                                                                app = Flask(__name__)
                                                                
                                                                # === MQTT SETUP ===
                                                                client = mqtt.Client()
                                                                client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
                                                                client.tls_set()
                                                                
                                                                def on_connect(client, userdata, flags, rc):
                                                                    print("Connected to MQTT Broker" if rc == 0 else f"Failed to connect: {rc}")
                                                                
                                                                client.on_connect = on_connect
                                                                client.connect(MQTT_BROKER, MQTT_PORT, 60)
                                                                client.loop_start()
                                                                
                                                                # === MediaPipe Face Mesh ===
                                                                mp_face_mesh = mp.solutions.face_mesh
                                                                face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
                                                                mp_drawing = mp.solutions.drawing_utils
                                                                
                                                                # === Get Frames from ESP32-CAM ===
                                                                def get_esp32_cam_frames():
                                                                    try:
                                                                        response = requests.get(ESP32_CAM_URL, stream=True, timeout=10)
                                                                        content_type = response.headers.get('Content-Type')
                                                                        if not content_type or 'boundary=' not in content_type:
                                                                            print("Boundary not found")
                                                                            return
                                                                        boundary = "--" + content_type.split('boundary=')[1]
                                                                        buffer = b''
                                                                
                                                                        for chunk in response.iter_content(chunk_size=1024):
                                                                            buffer += chunk
                                                                            while True:
                                                                                start = buffer.find(boundary.encode())
                                                                                if start == -1:
                                                                                    break
                                                                                end = buffer.find(boundary.encode(), start + len(boundary))
                                                                                if end == -1:
                                                                                    break
                                                                                frame_data = buffer[start:end]
                                                                                buffer = buffer[end:]
                                                                                jpeg_start = frame_data.find(b'\r\n\r\n')
                                                                                if jpeg_start != -1:
                                                                                    yield frame_data[jpeg_start + 4:]
                                                                    except Exception as e:
                                                                        print(f"ESP32-CAM Error: {e}")
                                                                
                                                                # === Face Tracking + Servo Control ===
                                                                def detect_head_position(frame):
                                                                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                                                    results = face_mesh.process(rgb)
                                                                    if not results.multi_face_landmarks:
                                                                        return None
                                                                    lm = results.multi_face_landmarks[0].landmark
                                                                    left_eye = lm[33]
                                                                    right_eye = lm[133]
                                                                    cx = int(((left_eye.x + right_eye.x) / 2) * frame.shape[1])
                                                                    cy = int(((left_eye.y + right_eye.y) / 2) * frame.shape[0])
                                                                    return cx, cy
                                                                
                                                                def smooth_move(prev, target, factor=0.2):
                                                                    return prev + (target - prev) * factor
                                                                
                                                                def map_to_angle(x, y, w, h, prev_pan, prev_tilt):
                                                                    pan = smooth_move(prev_pan, np.interp(x, [0, w], [180, 0]))
                                                                    tilt = smooth_move(prev_tilt, np.interp(y, [0, h], [0, 180]))
                                                                    pan, tilt = int(np.clip(pan, 0, 180)), int(np.clip(tilt, 0, 180))
                                                                    if abs(pan - prev_pan) < 2: pan = prev_pan
                                                                    if abs(tilt - prev_tilt) < 2: tilt = prev_tilt
                                                                    return pan, tilt
                                                                
                                                                def face_tracking_task():
                                                                    prev_pan, prev_tilt = 90, 90
                                                                    for jpeg in get_esp32_cam_frames():
                                                                        arr = np.frombuffer(jpeg, np.uint8)
                                                                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                                                        if frame is None:
                                                                            continue
                                                                        pos = detect_head_position(frame)
                                                                        if pos:
                                                                            pan, tilt = map_to_angle(pos[0], pos[1], frame.shape[1], frame.shape[0], prev_pan, prev_tilt)
                                                                            if pan != prev_pan:
                                                                                client.publish(MQTT_TOPIC_PAN_CMD, str(pan))
                                                                            if tilt != prev_tilt:
                                                                                client.publish(MQTT_TOPIC_TILT_CMD, str(tilt))
                                                                            prev_pan, prev_tilt = pan, tilt
                                                                
                                                                # === Flask Routes ===
                                                                @app.route('/')
                                                                def index():
                                                                    return render_template('index.html')
                                                                
                                                                @app.route('/move', methods=['POST'])
                                                                def move():
                                                                    pan = request.form.get('pan')
                                                                    tilt = request.form.get('tilt')
                                                                    try:
                                                                        if pan:
                                                                            val = int(pan)
                                                                            if 0 <= val <= 180:
                                                                                client.publish(MQTT_TOPIC_PAN_CMD, str(val))
                                                                        if tilt:
                                                                            val = int(tilt)
                                                                            if 0 <= val <= 180:
                                                                                client.publish(MQTT_TOPIC_TILT_CMD, str(val))
                                                                        return jsonify(status='success', pan=pan, tilt=tilt)
                                                                    except:
                                                                        return jsonify(status='error'), 400
                                                                
                                                                @app.route('/video_feed')
                                                                def video_feed():
                                                                    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
                                                                
                                                                def gen_frames():
                                                                    for frame in get_esp32_cam_frames():
                                                                        yield (b'--frame\r\n'
                                                                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                                                                
                                                                if __name__ == '__main__':
                                                                    t = threading.Thread(target=face_tracking_task, daemon=True)
                                                                    t.start()
                                                                    app.run(host='0.0.0.0', port=5000, debug=True)
