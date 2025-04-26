# from flask import Blueprint, render_template
# from utils.face_utils import extract_attendance, total_registered_faces
# from utils.database import get_registered_students
# from datetime import date

# main_routes = Blueprint('main_routes', __name__)

# @main_routes.route('/')
# def home():
#     datetoday2 = date.today().strftime("%d-%B-%Y")
#     names, rolls, times, l = extract_attendance()
#     students = get_registered_students()
#     return render_template('home.html',
#                            datetoday2=datetoday2,
#                            names=names, rolls=rolls, times=times, l=l,
#                            totalreg=len(students), students=students,
#                            mess="Welcome! Press 'a' to take attendance.")
