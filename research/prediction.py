import asyncio
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic



def create_landmark_frame_df(results,frame,xyz):
    """
    xyz takes the results from mediapipe and creates a dataframe of the landmark
    
    inputs:
        results: mediapipe results object
        frame: frame number
        xyz: dataframe wof the xyz example data
    
    
    """
    
    #we want the values and rows for every type landmark index so we need skeleton
    xyz_skel = xyz[['type','landmark_index']].drop_duplicates() \
    .reset_index(drop=True).copy()
    
    pose = pd.DataFrame()
    face = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()
    
    if results.face_landmarks is not None:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x','y','z']] = [point.x, point.y, point.z]

    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x','y','z']] = [point.x, point.y, point.z]


    if results.left_hand_landmarks is not None:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x','y','z']] = [point.x, point.y, point.z]


    if results.right_hand_landmarks is not None:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x','y','z']] = [point.x, point.y, point.z]

    face = face.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='face')

    pose = pose.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='pose')

    left_hand = left_hand.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='left_hand')

    right_hand = right_hand.reset_index() \
    .rename(columns={'index': 'landmark_index'}) \
    .assign(type='right_hand')

    landmark = pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True)
    #merge with lanndmark
    landmark = xyz_skel.merge(landmark, on=['type','landmark_index'], how='left')
    #assign frames to make it unique
    landmark = landmark.assign(frame = frame)
    return landmark

async def do_capture_web_async():
    all_landmarks = []

    cap = cv2.VideoCapture(0)
    xyz = pd.read_parquet('./2044/635217.parquet')
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        frame = 0  
        while cap.isOpened():
                #take frame and increment it
            frame+=1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            #create landmark dataframe
            landmark = create_landmark_frame_df(results, frame, xyz)
            all_landmarks.append(landmark)
            

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style())

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_hand_landmarks_style())

                # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        

    cap.release()
    xyz_np = np.array(all_landmarks).astype(np.float32)

        # Perform prediction
    await predict_and_print(xyz_np)

    

    
# Function to perform prediction and print the result
def predict_and_print(landmark):
    # Load the model and get the prediction_fn
    interpreter = tf.lite.Interpreter(model_path="./model.tflite")
    prediction_fn = interpreter.get_signature_runner("serving_default")

    # Perform prediction
    predictions = prediction_fn(inputs=landmark)

    # Get the predicted sign
    sign = predictions['outputs'].argmax()

    # Print the predicted sign (modify this part according to your requirements)
    print(f"Predicted sign: {sign}")

if __name__ == "__main__":
    try:
        # Create a new asyncio event loop and run the capture task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(do_capture_web_async())
    except KeyboardInterrupt:
        # If Esc is pressed, stop the capture task and exit gracefully
        print("Process interrupted by Esc key.")
    finally:
        # Release any resources (e.g., close the model, etc.) before exiting
        pass