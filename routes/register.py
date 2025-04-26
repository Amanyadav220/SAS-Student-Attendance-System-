import os
import cv2
import numpy as np
from flask import Blueprint, request, redirect
from utils.helpers import encode_face, datetoday
from utils.database import insert_student

register = Blueprint('register', __name__)

IMAGES_DIR = 'static/faces/'

@register.route('/register', methods=['POST'])
def register_student():
    name = request.form['name']
    student_id = request.form['student_id']
    image_file = request.files['image']

    student_folder = os.path.join(IMAGES_DIR, name + '_' + student_id)
    os.makedirs(student_folder, exist_ok=True)

    filepath = os.path.join(student_folder, image_file.filename)
    image_file.save(filepath)

    insert_student(name, student_id)  # Save in MySQL
    return redirect('/')
