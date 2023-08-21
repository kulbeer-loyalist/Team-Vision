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

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


   
@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact-us')
def contact():
    return render_template('contact.html')

@app.route('/team')
def team():
    return render_template('Team.html')



def get_db_connection():
    return sqlite3.connect('login.db')

def sentence_generation(keywords,model_list):
    # models=["k2t","k2t-base","mrm8488/t5-base-finetuned-common_gen"]
    models_dict = {'k2t_model_tuned':"mrm8488/t5-base-finetuned-common_gepon"}
    model = pipeline(models_dict['k2t_model_tuned'])
    model_list.append(model(keywords))
    return model_list




# Route to handle sign-in page
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the username or email and password exist in the database
        cur.execute('SELECT * FROM login WHERE (username = ? OR email = ?) AND password = ?', (username_or_email, username_or_email, password))
        user = cur.fetchone()

        conn.close()

        if user:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid username/email or password'})
    else:
        return render_template('register.html')
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




@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Connect to the SQLite database
    conn = sqlite3.connect('login.db')
    cursor = conn.cursor()

    # Check if the username or email already exists
    cursor.execute('SELECT * FROM signup WHERE username=? OR email=?', (username, email))
    existing_user = cursor.fetchone()
    if existing_user:
        conn.close()
        return jsonify({'error': 'Username or email already exists'})

    # Insert the new user into the signup table
    cursor.execute('INSERT INTO signup (username, email, password) VALUES (?, ?, ?)', (username, email, password))
    conn.commit()
    conn.close()

    # Return a success message
    return jsonify({'success': True, 'message': 'User registered successfully'})

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8001)    