"""
ARM - Webcam Predictions
This script captures the image from the webam, preprocess it, and then predicts the sign using the latest model.

Created by:
- Marcus Vinicius da Silva Fernandes.
2023-08-12.

ARM = Action Recognition Modelling.
References:
- https://www.youtube.com/watch?v=doDUihpj6ro
"""

"""
Importing the libraries
"""
import cv2
import mediapipe as mp
import numpy as np
import json

"""
Set up of the Holistic model by Mediapipe

It will run the following models:
- pose_landmarks
- face_landmarks
- left_hand_landmarks
- right_hand_landmarks
"""
mp_holistic = mp.solutions.holistic  # for landmarks detection.

"""
Landmarks detection function
"""
# Function to detect the landmarks in each frame or image
def landmark_detection(frame, model):
    # Color conversion because mediapipe's landmark detection model expects RGB frames as input.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # color conversion BGR to RGB.
    frame.flags.writeable = False  # frame is not writeable.
    results = model.process(frame)  # landmarks detection.
    frame.flags.writeable = True  # frame is writeable.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # color conversion RGB to BGR.
    return frame, results

"""
Landmarks coordinates extraction function

It will :
- Extract the coordinates from the parameter 'results'.
- Only x and y coordinates are saved
- Store them into a numpy array.
- It will store zeros if the parameter 'results' has no value for the model (e.g. it can happen when the hand was not visible and therefore was not identified).
"""
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

"""
Loading and shaping the landmarks to the desired number of frames
- Creation of a dictionary to associate the words to a unique number.
"""
# Loading the json file adn creation of dictionary to associate the words to a unique number
with open('sign_to_prediction_index_map.json', 'r') as j:
     sign_dict = json.loads(j.read())

del j

"""
Desired number of frames
- Each video will be reshaped to have the number of rows (or frames) equal to the desired number of frames defined below.
"""
NUM_FRAMES = 30

"""
Landmark points to keep
- The objective is to reduce the number of features.
- All the landmarks from the hands will be kept.
"""
face_keep_points = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
face_keep_points.sort()
left_hand_keep_points = [i for i in range(21)]
pose_keep_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
right_hand_keep_points = [i for i in range(21)]

face_keep_idx = [face_keep_points[i] for i in range(len(face_keep_points))]
left_hand_keep_idx = [i + 468 for i in left_hand_keep_points]
pose_keep_idx = [i + 468 + 21 for i in pose_keep_points]
right_hand_keep_idx = [i + 468 + 21 + 33 for i in right_hand_keep_points]

landmarks_to_keep = face_keep_idx + left_hand_keep_idx + pose_keep_idx + right_hand_keep_idx

del face_keep_points, left_hand_keep_points, pose_keep_points, right_hand_keep_points
del face_keep_idx, left_hand_keep_idx, pose_keep_idx, right_hand_keep_idx

TOTAL_ROWS = 543
desired_num_rows = len(landmarks_to_keep) * 2

"""
Preprocessing the landmarks
"""
def preprocess_landmarks(data):

    landmarks = np.empty((1, NUM_FRAMES, desired_num_rows), dtype=float)
 
    # Reshaping the data
    num_frames = int(len(data) / TOTAL_ROWS)
    data = data.reshape(num_frames, TOTAL_ROWS, 2)
    data.astype(np.float32)

    # Dropping undesired points
    data = data[:, landmarks_to_keep]

    # Adjusting the number of frames
    if data.shape[0] > NUM_FRAMES:  # time-based sampling
        indices = np.arange(0, data.shape[0], data.shape[0] // NUM_FRAMES)[:NUM_FRAMES]
        data = data[indices]
    elif data.shape[0] < NUM_FRAMES:  # padding the videos
        rows = NUM_FRAMES - data.shape[0]
        data = np.append(np.zeros((rows, len(landmarks_to_keep), 2)), data, axis=0)

    # Reshaping the data
    landmarks = data.reshape(NUM_FRAMES, len(landmarks_to_keep) * 2, order='F')
    del data

    return landmarks

"""
Model build
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Activation, Dropout, LSTM, Masking

input_shape = (None, 158)
output_classes = 250

model = Sequential()

model.add(Masking(mask_value=0, input_shape=input_shape))

model.add(Dense(512))
model.add(LayerNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(LayerNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(256))
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))

model.add(Dense(output_classes, activation='softmax'))

"""
Loading the weights
"""
model.load_weights('08-12_ARM_GD_Final-Architecture.h5')

"""
Main code for detection and extraction
- Capturing the image from the webcam and converting it into frames by OpenCV.
- For each frame, the function landmark_detection will be called to make the detections.
- If at least one hand is visible, the landmarks will be stored in landmarks_array. Otherwise, they will be discarded.
- When the performer hides the hands from the camera after performing the sign, the array containing the landmarks will be preprocessed and then a prediction will happen.
"""
if __name__ == "__main__":

    # Capturing the video frames from the webcam
    cap = cv2.VideoCapture(0)

    # List that will receive the landmark's coordinates for each video
    landmarks_list = []
    sign_status = 0  # 0 = not performing a sign / 1 = performing a sign

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    
        # Looping through all the frames
        while cap.isOpened():  # making sure it is reading frames

            # Reading the frames
            ret, frame = cap.read()

            # # Resizing every frame to a commom value
            # frame = cv2.resize(frame, (256, 256))

            # Making detections
            image, results = landmark_detection(frame, holistic)

            # Draw landmarks

            # Show to screen
            cv2.imshow("Video", image)
                    
            # Extracting landmarks
            landmarks_list_np, lh_visible, rh_visible = landmark_extraction(results)
            landmarks_list.append(landmarks_list_np)

            if lh_visible == 1 or rh_visible == 1:
                landmarks_array = np.concatenate(landmarks_list, axis=0)
                sign_status = 1
                
            if lh_visible == 0 and rh_visible == 0 and sign_status == 1:
                # Predictions
                x_test = np.expand_dims(preprocess_landmarks(landmarks_array), axis=0)

                # Making predictions
                predicted_label = np.argmax(model.predict(x_test))
                predicted_word = np.array([list(sign_dict.keys())[list(sign_dict.values()).index(predicted_label)]])
                print(predicted_label, predicted_word)
                
                landmarks_list = []
                sign_status = 0
                del landmarks_list_np, landmarks_array

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
