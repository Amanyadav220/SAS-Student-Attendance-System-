import os
import cv2
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, session, url_for
from sklearn.neighbors import KNeighborsClassifier
import joblib
import time
from PIL import Image
import pymysql

# Variables
MESSAGE = "WELCOME. Instruction: To register your attendance, kindly click on 'a' on the keyboard."

# Flask App setup
app = Flask(__name__)
app.secret_key = "amizone"


# Saving Date today in 2 formats
def get_datetoday():
    return date.today().strftime("%m_%d_%y")


def get_datetoday2():
    return date.today().strftime("%d-%B-%Y")


# VideoCapture object to access webcam
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Create required directories if they don't exist
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static"):
    os.makedirs("static")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")
if f"Attendance-{get_datetoday()}.csv" not in os.listdir("Attendance"):
    with open(f"Attendance/Attendance-{get_datetoday()}.csv", "w") as f:
        f.write("Name,Roll,Time")


def get_registered_students():
    connection = pymysql.connect(
        host="localhost", user="root", password="", database="Studentattendancesystem"
    )
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, course, specialization, section FROM students")
    students = cursor.fetchall()
    connection.close()
    return students


print("Registered students:", get_registered_students())


# Get total number of registered users
def totalreg():
    return len(os.listdir("static/faces"))


# Extract faces from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        return faces
    return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(facearray)


# Train the model with all the faces in the 'faces' folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir("static/faces")
    for user in userlist:
        user_folder = f"static/faces/{user}"
        for imgname in os.listdir(user_folder):
            img = cv2.imread(f"{user_folder}/{imgname}")
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")


# Extract today's attendance from CSV
def extract_attendance():
    df = pd.read_csv(f"Attendance/Attendance-{get_datetoday()}.csv")
    names = df["Name"]
    rolls = df["Roll"]
    times = df["Time"]
    return names, rolls, times, len(df)


# Add attendance for a specific user
def add_attendance(name):
    username, userid = name.split("_")
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f"Attendance/Attendance-{get_datetoday()}.csv")

    if str(userid) not in list(df["Roll"]):
        with open(f"Attendance/Attendance-{get_datetoday()}.csv", "a") as f:
            f.write(f"\n{username},{userid},{current_time}")
    else:
        print(f"{username} has already marked attendance for the day.")


################## ROUTING FUNCTIONS ##############################


@app.route("/")
def login():
    return render_template("adminlogin.html")


@app.route("/login", methods=["POST"])
def do_login():
    username = request.form["username"]
    password = request.form["password"]
    if username == "admin" and password == "admin123":
        session["user"] = username
        return redirect(url_for("home"))
    return render_template("login.html", error="Invalid Credentials")


@app.route("/signup")
def signup():
    return render_template("sign.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form["username"]
    password = request.form["password"]

    if request.method == "POST":
        name = request.form["newuserName"]
        password = request.form["password"]
        # Insert into MySQL
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="",
            database="Studentattendancesystem",
        )
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO Admin (Username, Password) VALUES (%s, %s)", (name, password)
        )
        connection.commit()
        connection.close()

    # TODO: Save to your database (MySQL, SQLite, etc.)
    print(f"Registered user: {username}")

    return redirect(url_for("login"))  # Redirect to login page after sign-up


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()  # Clear all session data
    return redirect("/")  # Redirect to login page


@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    names, rolls, times, l = extract_attendance()
    students = get_registered_students()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=get_datetoday2(),
        mess=MESSAGE,
        student=students,
    )


@app.route("/start", methods=["GET"])
def start():
    if "face_recognition_model.pkl" not in os.listdir("static"):
        names, rolls, times, l = extract_attendance()
        MESSAGE = "No trained model found. Please register a user first."
        return render_template(
            "home.html",
            names=names,
            rolls=rolls,
            times=times,
            l=l,
            totalreg=totalreg(),
            datetoday2=get_datetoday2(),
            mess=MESSAGE,
        )

    cap = cv2.VideoCapture(0)
    recognized_users = set()

    start_time = time.time()
    duration = 60  # seconds to scan for faces

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            face = cv2.resize(frame[y : y + h, x : x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            if identified_person not in recognized_users:
                add_attendance(identified_person)
                recognized_users.add(identified_person)
                print(f"Attendance marked for {identified_person}")

            # Drawing boxes and names
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                identified_person,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
            )

        cv2.imshow("Taking Attendance... Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    MESSAGE = "Attendance taken successfully"
    return redirect(url_for("home"))


@app.route("/add", methods=["GET", "POST"])
def add():

    if request.method == "POST":
        name = request.form["newusername"]
        roll = request.form["newuserid"]
        course = request.form["courseid"]
        specialization = request.form["specializationid"]
        section = request.form["sectionid"]
        # Insert into MySQL
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="",
            database="Studentattendancesystem",
        )
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO students (id, name, course, specialization, Section) VALUES (%s, %s, %s, %s, %s)",
            (roll, name, course, specialization, section),
        )
        connection.commit()
        connection.close()

    newusername = request.form["newusername"]
    newuserid = request.form["newuserid"]
    userimagefolder = f"static/faces/{newusername}_{newuserid}"

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(
                frame,
                f"Images Captured: {i}/50",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 20),
                2,
            )
            if j % 10 == 0:
                name = f"{newusername}_{i}.jpg"
                cv2.imwrite(f"{userimagefolder}/{name}", frame[y : y + h, x : x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow("Adding new User", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Training Model")
    train_model()
    names, rolls, times, l = extract_attendance()
    MESSAGE = "User added successfully."
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, port=1000)
