from flask import Blueprint, render_template
import cv2
from datetime import datetime
from utils.helpers import *
from utils.helpers import face_detector

bp = Blueprint('attendance', __name__)

@bp.route('/start')
def start():
    ATTENDENCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us , kindly register yourself first'
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                ATTENDENCE_MARKED = True
                break
        if ATTENDENCE_MARKED or cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('Attendance Check', frame)

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2, mess='Attendance taken successfully')
