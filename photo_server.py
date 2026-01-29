# For flask and related operations
from flask import Flask, send_from_directory, jsonify, Response, request
from flask_cors import CORS
import socket

# Supress warnings for entire notebook
import warnings
warnings.filterwarnings('ignore')

# Import the video processing and deep learning modules
import cv2
import numpy as np
import os
import threading
from ultralytics import YOLO
from datetime import datetime
import time
import math
import json

# Get firebase backend storage functions
import requests
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, storage, firestore, messaging

# FLASK SETUP !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow access from devices on your network

# FOR FIREBASE NOTIFICATIONS ~~~~~~~~~~~~~~~~~~~~~~
# Path to your Firebase service account key JSON file
SERVICE_ACCOUNT_FILE = "firebase_key.json"
fcm_tokens = {}

# FCM API URL for HTTP v1
FCM_URL = "https://fcm.googleapis.com/v1/projects/intruder-detection-syste-3e8f1/messages:send"

# Load credentials
# credentials = service_account.Credentials.from_service_account_file(
#     SERVICE_ACCOUNT_FILE,
#     scopes=["https://www.googleapis.com/auth/cloud-platform"]
# )

# DETECTION MODULE SETUP ~~~~~~~~~~~~~~~~~~~~~~~~~~
INTRUDER_FOLDER = r'C:\\Users\\manoh\\intruder-app\\intruders'  # Folder where detection saves images

# Start analyzing the frames until quit
prev_person_count = curr_person_count = 0
prev_coords = curr_coords =[[0,0,0,0]]

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # Ensure you have the model downloaded


# BMS FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Area Equation
def area_eqn(x1, y1, x2, y2):
    return (x2-x1)*(y2-y1)

# Mid Point Equation
def center_eqn(x1, y1, x2, y2):
    return [(x1+x2)/2, (y1+y2)/2]

# Euclidiean Distance
def euc_eqn(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Logic for Box Movement Score by calculating Area Change and Centre Change
def box_movement_score(prev_person_count, prev_coords, curr_person_count, curr_coords):
    if prev_person_count != curr_person_count:
        return True
    else:
        threshold = 0.8
        for i in range(curr_person_count):
            # AREA CHANGE
            prewid = (prev_coords[i][2] - prev_coords[i][0])
            preht = (prev_coords[i][3] - prev_coords[i][1])
            prevar = prewid * preht
            
            currwid = (curr_coords[i][2] - curr_coords[i][0])
            currht = (curr_coords[i][3] - curr_coords[i][1])
            currar = currwid * currht
            
            area_ch = abs(prevar - currar) 
            area_sc = 0 if prevar == 0 else area_ch / prevar
            
            # CENTRE CHANGE
            prediag = euc_eqn(prev_coords[i][0], prev_coords[i][1], prev_coords[i][2], prev_coords[i][3])
            
            prevmid = center_eqn(prev_coords[i][0], prev_coords[i][1], prev_coords[i][2], prev_coords[i][3])
            currmid = center_eqn(curr_coords[i][0], curr_coords[i][1], curr_coords[i][2], curr_coords[i][3])
            
            x1, y1, x2, y2 = prevmid + currmid
            cent_ch = euc_eqn(x1, y1, x2, y2) 
            cent_sc = 0 if prediag == 0 else cent_ch / prediag
            
            if area_sc > 5*threshold or cent_sc > 3*threshold:
                return True
    return False


# FIREBASE FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Upload image to Firebase Storage
def upload_image(image_path):
    bucket = storage.bucket()
    blob = bucket.blob(f"intruders/{os.path.basename(image_path)}")
    blob.upload_from_filename(image_path)
    blob.make_public()  # Makes the image URL publicly accessible (optional)

    return blob.public_url

# # Save Metadata to Firestore
def save_metadata(image_url):
    doc_ref = db.collection('intruder_logs').add({
        'imageUrl': image_url,
        'timestamp': datetime.now().isoformat()
    })

# # Full Function to Call from Detection Code
def handle_intruder_detection(image_path):
    image_url = upload_image(image_path)
    save_metadata(image_url)
    print(f"Intruder saved: {image_url}")

# To send push notifications for the app
def send_push_notification(fcm_token, message_title, message_body):
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=message_title,
                body=message_body
            ),
            token=fcm_token,
        )
        # Send the message to the device using its FCM token
        response = messaging.send(message)
        print('Successfully sent message:', response)
    except Exception as e:
        print(f"Error sending message: {str(e)}")

# YOLO FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to detect people using YOLO
def detect_persons(frame):
    results = yolo_model(frame)  # Run YOLO
    persons = []

    for result in results:
        for box, class_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):  # Extract confidence score
            if int(class_id) == 0:  # Class 0 in COCO is "person"
                x1, y1, x2, y2 = map(int, box[:4])
                persons.append({
                    "bbox": (x1, y1, x2, y2),  # Bounding box
                    "confidence": conf.item()   # Confidence score
                })

    return persons

# Helper function to save a frame
def save_intruder_frame(frame, prefix=""):
    # Save files to local disk
    filename = f"{INTRUDER_FOLDER}/Intruder_{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")
    
    # Optional: Call FIREBASE cloud upload function here
    handle_intruder_detection(filename)


# NECESSARY INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize Firebase App
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'intruder-detection-syste-3e8f1.firebasestorage.app'
})
db = firestore.client()

# Initialize the camera (0 usually refers to the default camera)
cap = cv2.VideoCapture(0)

# FLASK ROUTES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Endpoint to list all photos with name, timestamp, and URL
@app.route('/photos')
def list_photos():
    try:
        files = os.listdir(INTRUDER_FOLDER)
        photos = []
        for file in files:
            file_path = os.path.join(INTRUDER_FOLDER, file)
            timestamp = os.path.getmtime(file_path)  # Last modified time
            photos.append({
                "name": file,
                "timestamp": timestamp,
                "url": f"http://{get_local_ip()}:8000/images/{file}"
            })
        # Sort by latest first (optional)
        photos.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(photos)

    except Exception as e:
        return jsonify({'error': str(e)})

# Endpoint to serve the actual images
@app.route('/images/intruders/<filename>')
def get_image(filename):
    return send_from_directory(INTRUDER_FOLDER, filename)

# Endpoint to receive fcm token
@app.route('/api/store-token', methods=['POST'])
def store_token():
    data = request.json
    fcm_token = data.get('fcm_token')
    user_id = data.get('user_id')

    if fcm_token:
        # Save the FCM token (this is just an example; you can save it in a database)
        fcm_tokens[user_id] = fcm_token  # You can associate the token with a specific user ID
        return jsonify({"message": "FCM Token stored successfully."}), 200
    else:
        return jsonify({"error": "FCM Token is missing."}), 400
    
# Endpoint to push notifications to the app
@app.route('/api/send-notification', methods=['POST'])
def send_notification():
    data = request.json
    user_id = data.get('user_id')
    message_title = data.get('title')
    message_body = data.get('body')

    # Retrieve the stored token for the user
    fcm_token = fcm_tokens.get(user_id)

    if fcm_token:
        # Send the push notification
        message = messaging.Message(
            notification=messaging.Notification(
                title=message_title,
                body=message_body
            ),
            token=fcm_token
        )

        try:
            response = messaging.send(message)
            return jsonify({"message": "Notification sent successfully.", "response": response}), 200
        except Exception as e:
            return jsonify({"error": f"Error sending notification: {str(e)}"}), 500
    else:
        return jsonify({"error": "FCM Token not found for this user."}), 404

# Helper to get your laptop’s IP (needed to access from phone)
def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

# Process frame for intruder detection and overlay annotations.
def process_frame(frame):
    global prev_person_count, curr_person_count, prev_coords, curr_coords
    persons = detect_persons(frame)

    if persons:  # Intruder Detected
        # Draw boxes and labels
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            confidence = person["confidence"]
            
            curr_coords.append([x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"Intruder Detected: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        curr_person_count = len(persons)
    
    # Check BMS Score
    if box_movement_score(prev_person_count, prev_coords, curr_person_count, curr_coords):
        save_intruder_frame(frame)
        
    # Update prev values
    prev_person_count = curr_person_count
    prev_coords = curr_coords
    
    return frame

# Video streaming function
def gen():
    while True:
        ret, frame = cap.read()  # Capture frame from the camera
        if not ret:
            break

        frame = process_frame(frame)

        _, jpeg = cv2.imencode('.jpg', frame)  # Encode as JPEG
        frame = jpeg.tobytes()  # Convert frame to bytes
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Endpoint to stream live video
@app.route('/live')
def live():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)