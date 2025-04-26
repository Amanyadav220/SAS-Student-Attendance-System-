# utils/helpers.py

import os
import cv2
import numpy as np
import pandas as pd
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import joblib


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Path settings
attendance_dir = 'Attendance'
faces_dir = 'static/faces'
model_path = 'static/face_recognition_model.pkl'

# Ensure necessary directories exist
os.makedirs(attendance_dir, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)

# Ensure today's attendance file exists
attendance_file = f'{attendance_dir}/Attendance-{datetoday}.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    """Return number of registered users (folders in faces directory)."""
    return len(os.listdir(faces_dir))

def extract_faces(img):
    """Detect faces in an image."""
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def identify_face(facearray):
    """Predict the identity using the trained model."""
    model = joblib.load(model_path)
    return model.predict(facearray)

def train_model():
    """Train a KNN model on the registered faces."""
    faces = []
    labels = []
    userlist = os.listdir(faces_dir)
    for user in userlist:
        for imgname in os.listdir(f'{faces_dir}/{user}'):
            img = cv2.imread(f'{faces_dir}/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, model_path)

def extract_attendance():
    """Read the current attendance CSV and return details."""
    df = pd.read_csv(attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    return names, rolls, times, len(df)

def add_attendance(name):
    """Append a new attendance entry if not already marked."""
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)
    if str(userid) not in list(df['Roll']):
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    else:
        print(f"{name} already marked for the day.")

def get_registered_students():
    folder_path = 'static/faces'
    students = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            name_id = os.path.splitext(filename)[0]  # remove .jpg
            parts = name_id.split('_')
            if len(parts) >= 2:
                name = parts[0]
                student_id = parts[1]
                students.append({'name': name, 'id': student_id})
    return students
