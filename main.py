import eventlet
eventlet.monkey_patch()

# Setelah monkey patch, baru impor modul lain
import cv2
import numpy as np
import tensorflow as tf
from scipy import spatial
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from scipy.optimize import linear_sum_assignment

# Modul Deteksi dan Pelacakan
class VehicleDetector:
    def __init__(self):
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] in ['car', 'truck', 'bus', 'motorbike']:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return [boxes[i] for i in indexes], [self.classes[class_ids[i]] for i in indexes]

class VehicleTracker:
    def __init__(self):
        self.tracks = []
        self.track_id = 0
        self.max_age = 10 

    def update(self, detections, labels):
        centroids = np.array([[x + w/2, y + h/2] for (x, y, w, h) in detections])
        
        if len(self.tracks) == 0:
            for centroid, label, box in zip(centroids, labels, detections):
                self.tracks.append(Track(self.track_id, centroid, label, box))
                self.track_id += 1
        else:
            prev_centroids = np.array([track.predict() for track in self.tracks])
            distances = spatial.distance_matrix(prev_centroids, centroids)
            row_ind, col_ind = linear_sum_assignment(distances)

            assigned_tracks = set()
            for row, col in zip(row_ind, col_ind):
                if distances[row, col] <= 50: 
                    self.tracks[row].update(centroids[col], labels[col], detections[col])
                    assigned_tracks.add(row)

            unassigned_detections = set(range(len(centroids))) - set(col_ind)
            for i in unassigned_detections:
                self.tracks.append(Track(self.track_id, centroids[i], labels[i], detections[i]))
                self.track_id += 1

           
            for track in self.tracks:
                if track.id in assigned_tracks:
                    track.age = 0
                else:
                    track.age += 1

            self.tracks = [track for track in self.tracks if track.age < self.max_age]

        return self.tracks

class Track:
    def __init__(self, track_id, initial_pos, label, box):
        self.id = track_id
        self.positions = [initial_pos]
        self.label = label
        self.box = box
        self.age = 0
        self.speed = 0  

    def predict(self):
        if len(self.positions) > 1:
            velocity = self.positions[-1] - self.positions[-2]
            return self.positions[-1] + velocity
        return self.positions[-1]

    def update(self, new_position, new_label, new_box):
        if len(self.positions) > 1:
            self.speed = np.linalg.norm(new_position - self.positions[-1])
        self.positions.append(new_position)
        self.label = new_label
        self.box = new_box

    def get_speed(self):
        return self.speed

# Modul Analisis Lalu Lintas
class TrafficAnalyzer:
    def __init__(self):
        self.vehicle_count = defaultdict(int)
        self.speed_data = defaultdict(list)
        self.anomaly_threshold = 3 

    def update(self, tracks):
        self.vehicle_count = defaultdict(int)
        for track in tracks:
            self.vehicle_count[track.label] += 1
            self.speed_data[track.label].append(track.get_speed())

    def get_average_speed(self):
        return {label: np.mean(speeds) if speeds else 0 for label, speeds in self.speed_data.items()}

    def detect_anomalies(self):
        anomalies = []
        for label, speeds in self.speed_data.items():
            if speeds:
                mean_speed = np.mean(speeds)
                std_speed = np.std(speeds)
                threshold = mean_speed + self.anomaly_threshold * std_speed
                if speeds[-1] > threshold:
                    anomalies.append(f"Anomaly detected: {label} speed")
        return anomalies


class TrafficPredictor:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, historical_data):
        X, y = self.prepare_data(historical_data)
        self.model.fit(X, y, epochs=100, verbose=0)

    def prepare_data(self, data):
        X, y = [], []
        for i in range(len(data) - 4):
            X.append(data[i:i+4])
            y.append(data[i+4])
        return np.array(X), np.array(y)

    def predict(self, current_data):
        return self.model.predict(np.array([current_data[-4:]]))

# Modul Visualisasi dan UI
class DataVisualizer:
    @staticmethod
    def draw_tracks(frame, tracks):
        for track in tracks:
            x, y = int(track.positions[-1][0]), int(track.positions[-1][1])
            cv2.putText(frame, f"ID: {track.id}, {track.label}", (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    @staticmethod
    def draw_stats(frame, vehicle_count, avg_speed):
        cv2.putText(frame, f"Total Vehicles: {sum(vehicle_count.values())}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y = 70
        for label, count in vehicle_count.items():
            cv2.putText(frame, f"{label}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 30
        y += 10
        for label, speed in avg_speed.items():
            cv2.putText(frame, f"Avg {label} speed: {speed:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y += 30

# Sistem Peringatan
class AlertSystem:
    def __init__(self, socket):
        self.socket = socket

    def send_alert(self, message):
        print(f"ALERT: {message}")
        self.socket.emit('alert', {'message': message})

# Konfigurasi Flask dan SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(vehicle_count)

# Modifikasi fungsi process_video untuk menggunakan eventlet
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = VehicleDetector()
    tracker = VehicleTracker()
    analyzer = TrafficAnalyzer()
    predictor = TrafficPredictor()
    visualizer = DataVisualizer()
    alert_system = AlertSystem(socketio)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1000 / fps  

    historical_data = []

    while True:
        start_time = cv2.getTickCount()

        ret, frame = cap.read()
        if not ret:
            break

        detections, labels = detector.detect(frame)
        tracks = tracker.update(detections, labels)
        
        analyzer.update(tracks)
        avg_speed = analyzer.get_average_speed()
        anomalies = analyzer.detect_anomalies()

        for anomaly in anomalies:
            alert_system.send_alert(anomaly)

        visualizer.draw_tracks(frame, tracks)
        visualizer.draw_stats(frame, analyzer.vehicle_count, avg_speed)

        cv2.imshow("Traffic Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Akumulasi data untuk prediksi
        total_vehicles = sum(analyzer.vehicle_count.values())
        historical_data.append(total_vehicles)
        if len(historical_data) > 100:
            predictor.train(historical_data)
            prediction = predictor.predict(historical_data)
            print(f"Predicted vehicles in next timeframe: {prediction[0][0]:.2f}")

        # Kirim update ke klien melalui SocketIO
        socketio.emit('update', {
            'vehicle_count': dict(analyzer.vehicle_count),
            'avg_speed': {k: float(v) for k, v in avg_speed.items()}
        })

        # Calculate processing time and sleep if necessary
        end_time = cv2.getTickCount()
        processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        sleep_time = max(1, int(frame_time - processing_time))
        eventlet.sleep(sleep_time / 1000)  # Convert to seconds

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    eventlet.spawn(process_video, 'traffic_video.mp4')
    socketio.run(app, debug=True, use_reloader=False)