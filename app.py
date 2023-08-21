"""
author @ kumar dahal
"""
from flask import Flask, render_template, redirect, url_for,request,jsonify
import sqlite3
import mediapipe as mp
import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json
from  keytotext import pipeline
from ASLD_step2.constants import *

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy import *

import pymysql
from flask import Blueprint, render_template, request, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_toastr import Toastr
from flask_login import (
    UserMixin,
    login_user,
    LoginManager,
    login_required,
    current_user,
    logout_user,
)

app = Flask(__name__)



pymysql.install_as_MySQLdb()
rdsConnection = create_engine(
    "mysql+mysqldb://admin:passwordRDS@visionrds.ch0vw6wkyqa3.ca-central-1.rds.amazonaws.com/"
)

# creates the database instance (SQLAlchemy)
db = SQLAlchemy()
DB_NAME = "users"
app = Flask(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
toastr = Toastr(app)

RDS_ENDPOINT = 3306
USERNAME = "admin"
PASSWORD = "passwordRDS"
DB_NAME = "users"

# app.config["SECRET_KEY"] = "test"
# app.config[
#     "SQLALCHEMY_DATABASE_URI"
# ] = f"mysql://admin:passwordRDS@visionrds.ch0vw6wkyqa3.ca-central-1.rds.amazonaws.com/vision"

# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# db = SQLAlchemy(app)

# login_manager = LoginManager(app)


# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key=True)
#     email = db.Column(db.String(150), unique=True)
#     password = db.Column(db.String(150))
#     name = db.Column(db.String(150))


# with app.app_context():
#     db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

def sentence_generation(keywords,model_list):
    # models=["k2t","k2t-base","mrm8488/t5-base-finetuned-common_gen"]
    models_dict = {'k2t_model_tuned':"mrm8488/t5-base-finetuned-common_gepon"}
    model = pipeline(models_dict['k2t_model_tuned'])
    model_list.append(model(keywords))
    return model_list



# Function to detect the landmarks in each frame or image
def landmark_detection(frame, model):
    # Color conversion because mediapipe's landmark detection model expects RGB frames as input.
    image_width = 1280
    image_height = 720
    resized_frame = cv2.resize(frame, (image_width, image_height))
    frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # color conversion BGR to RGB.
    frame.flags.writeable = False  # frame is not writeable.
    results = model.process(frame)  # landmarks detection.
    frame.flags.writeable = True  # frame is writeable.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # color conversion RGB to BGR.
    return frame, results

# Function to extract the coordinates of the detected landmarks
def landmark_extraction(results):
    lh_visible = 0
    rh_visible = 0

    if results.face_landmarks:
        face = np.array([[coordinate.x, coordinate.y] for coordinate in results.face_landmarks.landmark])
    else:
        face = np.array([[0, 0] for idx in range(468)])

    if results.left_hand_landmarks:
        left_hand = np.array([[coordinate.x, coordinate.y] for coordinate in results.left_hand_landmarks.landmark])
        lh_visible = 1
    else:
        left_hand = np.array([[0, 0] for idx in range(21)])
        lh_visible = 0
        
    if results.pose_landmarks:
        pose = np.array([[coordinate.x, coordinate.y] for coordinate in results.pose_landmarks.landmark])
    else:
        pose = np.array([[0, 0] for idx in range(33)])
    
    if results.right_hand_landmarks:
        right_hand = np.array([[coordinate.x, coordinate.y] for coordinate in results.right_hand_landmarks.landmark])
        rh_visible = 1
    else:
        right_hand = np.array([[0, 0] for idx in range(21)])
        rh_visible = 0
            
    return np.concatenate([face, left_hand, pose, right_hand]), lh_visible, rh_visible

def preprocess_landmarks(data):

    landmarks = np.empty((1, NUM_FRAMES, DESIRED_NUM_ROWS), dtype=float)
 
    # Reshaping the data
    num_frames = int(len(data) / TOTAL_ROWS)
    data = data.reshape(num_frames, TOTAL_ROWS, 2)
    data.astype(np.float32)

    # Dropping undesired points
    data = data[:, LANDMARKS_TO_KEEP]

    # Adjusting the number of frames
    if data.shape[0] > NUM_FRAMES:  # time-based sampling
        indices = np.arange(0, data.shape[0], data.shape[0] // NUM_FRAMES)[:NUM_FRAMES]
        data = data[indices]
    elif data.shape[0] < NUM_FRAMES:  # padding the videos
        rows = NUM_FRAMES - data.shape[0]
        data = np.append(np.zeros((rows, len(LANDMARKS_TO_KEEP), 2)), data, axis=0)

    # Reshaping the data
    landmarks = data.reshape(NUM_FRAMES, len(LANDMARKS_TO_KEEP) * 2, order='F')
    del data

    return landmarks
   


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact-us')
def contact():
    return render_template('contact.html')

@app.route('/team')
def team():
    return render_template('Team.html')


@app.route('/handle_button')
def prediction():

    with open('sign_to_prediction_index_map.json', 'r') as j:
     sign_dict = json.loads(j.read())

    arm_model = Path('final_model.h5')
    model = tf.keras.models.load_model(arm_model)

   # Capturing the video frames from the webcam
    cap = cv2.VideoCapture(0)

    # List that will receive the landmark's coordinates for each video
    landmarks_list = []

    sign_status = 0  # 0 = not performing a sign / 1 = performing a sign

    mp_holistic = mp.solutions.holistic  

    words_for_nlp = np.empty((0))
    list_corpus = []
    corpus = []
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    
        # Looping through all the frames
        while cap.isOpened():  # making sure it is reading frames

            # Reading the frames
            ret, frame = cap.read()

            # Making detections
            image, results = landmark_detection(frame, holistic)


                    
            # Extracting landmarks
            landmarks_list_np, lh_visible, rh_visible = landmark_extraction(results)
            landmarks_list.append(landmarks_list_np)

            # Storing the landmarks in an array when the sign is being performed (at least one hand is visible).
            if lh_visible == 1 or rh_visible == 1:
                landmarks_array = np.concatenate(landmarks_list, axis=0)
                sign_status = 1
            
            # Making predictions when the sign is done (both hands are not visible).
            if lh_visible == 0 and rh_visible == 0 and sign_status == 1:
                # Shaping the array
                x_test = np.expand_dims(preprocess_landmarks(landmarks_array), axis=0)

                # Making predictions
                predicted_label = np.argmax(model.predict(x_test))
                predicted_word = np.array([list(sign_dict.keys())[list(sign_dict.values()).index(predicted_label)]])
                print(predicted_label, predicted_word)

                # Storing the words in an array, to send to text-to-text model
                words_for_nlp = np.append(words_for_nlp, predicted_word)
                list_corpus.append(predicted_word)
                # Reseting the variables
                landmarks_list = []
                sign_status = 0
                del landmarks_list_np, landmarks_array

            # Getting the sentence from the predicted words
            if len(words_for_nlp) == 3:
                model_list = []
                model_list = sentence_generation(words_for_nlp, model_list)
                corpus.append(model_list)
                words_for_nlp = np.empty((0))

            frame_with_predictions = overlay_text(image.copy(), list_corpus,corpus) 
            
            
            cv2.imshow("Video", frame_with_predictions)   

            # Break gracefully (it will close capturing video from webcam when user press the button Q)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('thankyou'))
     
def overlay_text(frame, text_list,corpus, position=(10, 30), font_scale=0.8,  thickness=2):
    window_size = 3
    y_offset = position[1]

    if len(text_list) > window_size:
        last_three = text_list[-window_size:]  # Get the last three elements from text_list
        del text_list[-window_size:]
    else:
        last_three = text_list

    for chunk in last_three:
        chunk_text = ' '.join(map(str, chunk))
        cv2.putText(frame, str(chunk_text), (position[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        y_offset += 30
     
    if corpus:
        # Draw the current sentence on the frame
        cv2.putText(frame, str(corpus[-1]), (position[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

            
    return frame
    
@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')


# def get_db_connection():
#     return sqlite3.connect('login.db')


# @app.route("/register", methods=["POST", "GET"])
# def register():
#     print("test")
#     if request.method == "POST":
#         email = request.form.get("email")
#         name = request.form.get("name")
#         password = request.form.get("password")

#         user = User.query.filter_by(email=email).first()
#         if user:
#             flash("Email already exists.", category="error")
#         else:
#             print("test1")
#             if len(email) < 4:
#                 flash("Email must be greater than 4 characters.", category="error")
#             elif len(name) < 2:
#                 flash("Name must be greater than 2 characters.", category="error")
#             elif len(password) < 6:
#                 flash("Password must be greater than 6 characters.", category="error")
#             else:
#                 new_user = User(
#                     email=email,
#                     name=name,
#                     password=generate_password_hash(password, method="sha256"),
#                 )
#                 print(new_user)
#                 db.session.add(new_user)
#                 db.session.commit()
#                 flash("Account created successfully.", category="success")
#                 return redirect(url_for("register"))
#         return render_template("register.html", user=current_user)
#     else:
#         return render_template("register.html", user=current_user)


# @app.route("/login", methods=["POST", "GET"])
# def login():
#     if request.method == "POST":
#         email = request.form.get("email")  # getting the user email from the login form
#         password = request.form.get(
#             "password"
#         )  # getting the user password from the login form

#         user = User.query.filter_by(
#             email=email
#         ).first()  # checking if the user email is in the database
#         if user:
#             if check_password_hash(
#                 user.password, password
#             ):  # checking in the database if the user password is correct
#                 flash(
#                     "Logged in successfully!", category="success"
#                 )  # to show the success messagesss
#                 login_user(user, remember=True)
#                 return redirect(
#                     url_for("index", user=current_user)
#                 )  # redirecting to user page if the login is successful
#             else:
#                 flash(
#                     "Incorrect password, try again.", category="error"
#                 )  # user password is incorrect
#         else:
#             flash(
#                 "Email does not exist.", category="error"
#             )  # if the email does not exist in the database
#     return render_template("login.html", user=current_user)


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))


# login_manager.init_app(app)

# # @app.route('/user', methods=['GET'])
# # def user():
# #     return render_template("user.html")


# @app.route("/logout")
# @login_required
# def logout():
#     logout_user()
#     flash('Logged out successfully!', 'alert-danger')
#     # toastr.info("This is a success message", title="Success")
#     return redirect(url_for("index"))



if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8002)    