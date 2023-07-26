import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
mp_holistic = mp.solutions.holistic.Holistic()

# setting up the path for videos and extracted landmarks
videos_path = "C:\\Users\\saran\\OneDrive\\Desktop\\video_dummy_data\\"
output_path = "C:\\Users\\saran\\OneDrive\\Desktop\\new_extracted_landmarks\\"

class LandmarkDetector:
    def __init__(self):
        # Load the landmark detection model
        self.mp_pose = mp.solutions.pose.Pose()

    def detect_landmarks(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")
        # Color conversion 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # landmarks detection
        results = self.mp_pose.process(frame_rgb)  
        return frame_rgb, results  
# Create an instance of the LandmarkDetector class
landmark_detector = LandmarkDetector()

# Open the video file
video_path = "C:\\Users\\saran\\OneDrive\\Desktop\\video_dummy_data\\00583.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Failed to open the video file. Please check the file path or format.")

while cap.isOpened():
    # Capture a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Perform landmark detection using the class method
    processed_frame, results = landmark_detector.detect_landmarks(frame)

    # Display the processed frame
    cv2.imshow('Landmark Detection', processed_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#Extracting the coordinates of landmarks detection
class LandmarkExtractor:
    def __init__(self):
        # Load the landmark detection model
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_face = mp.solutions.face_mesh.FaceMesh()

    def extract_landmarks(self, frame):
        if frame is None:
            raise ValueError("Input frame is empty or None.")

        # Color conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform landmark detection on the frame
        pose_results = self.mp_pose.process(frame_rgb)
        hand_results = self.mp_hands.process(frame_rgb)
        face_results = self.mp_face.process(frame_rgb)

        # Extract pose landmarks 
        pose_landmarks = []
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                pose_landmarks.extend([landmark.x, landmark.y, landmark.z])
            # Pad with zeros if fewer than 33 landmarks are detected
            pose_landmarks.extend([0.0, 0.0, 0.0] * (33 - len(pose_landmarks) // 3))

        # Extract left hand landmarks 
        left_hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    left_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
            # Pad with zeros if fewer than 21 landmarks are detected
            left_hand_landmarks.extend([0.0, 0.0, 0.0] * (21 - len(left_hand_landmarks) // 3))

        # Extract right hand landmarks 
        right_hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    right_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
            # Pad with zeros if fewer than 21 landmarks are detected
            right_hand_landmarks.extend([0.0, 0.0, 0.0] * (21 - len(right_hand_landmarks) // 3))

        # Extract face landmarks 
        face_landmarks = []
        if face_results.multi_face_landmarks:
            for face_landmarks_list in face_results.multi_face_landmarks:
                for landmark in face_landmarks_list.landmark:
                    face_landmarks.extend([landmark.x, landmark.y, landmark.z])
            # Pad with zeros if fewer than 468 landmarks are detected
            face_landmarks.extend([0.0, 0.0, 0.0] * (468 - len(face_landmarks) // 3))

        # Concatenate the landmarks into a single numpy array
        all_landmarks = np.array(pose_landmarks + left_hand_landmarks + right_hand_landmarks + face_landmarks, dtype=np.float32)

        return all_landmarks

# Create an instance of the LandmarkExtractor class
landmark_extractor = LandmarkExtractor()

# Open the video file
video_path = "C:\\Users\\saran\\OneDrive\\Desktop\\video_dummy_data\\00426.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Failed to open the video file. Please check the file path or format.")

while cap.isOpened():
    # Capture a frame from the video
    ret, frame = cap.read()

    if not ret:
       
        break

    # Extract landmarks using the class method
    landmarks = landmark_extractor.extract_landmarks(frame)

   
    print("Landmarks Shape:", landmarks.shape)
    cv2.imshow('Landmark Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

#storing array in parquet file
class ParquetWriter:
    def __init__(self, landmarks_path):
        self.landmarks_path = landmarks_path

    def write(self, np_array, video_id):
        np_array_flat = np_array.flatten()
        pa_array = pa.array(np_array_flat)  
        table = pa.Table.from_arrays([pa_array], names=[video_id])  
        writer = pq.ParquetWriter(self.landmarks_path + video_id + '.parquet', table.schema) 
        writer.write_table(table)  
        writer.close()  


np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
video_id = "my_video"
dest_folder = output_path = "C:\\Users\\saran\\OneDrive\\Desktop\\new_extracted_landmarks\\"

parquet_writer = ParquetWriter(dest_folder)
parquet_writer.write(np_array, video_id)

#Capturing the video frames from the file 
class VideoProcessor:
    def __init__(self, videos_path, landmarks_path):
        self.videos_path = videos_path
        self.landmarks_path = landmarks_path

    def process_videos(self):
        for item in os.listdir(self.videos_path):
            if item.endswith('.mp4'):  
                cap = cv2.VideoCapture(os.path.join(self.videos_path, item))

                # List for landmark's coordinates for each video
                landmarks_list = []

                # Set mediapipe model
                with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        
                    # Looping through all the frames
                    while cap.isOpened():  

                        # Reading the frames
                        ret, frame = cap.read()
                        if not ret:  
                            break
                        # Making detections
                        image, results = self.landmark_detection(frame, holistic)
                        
                        # Extracting landmarks
                       
                        landmarks_list.append(self.landmark_extraction(results))

                        cv2.waitKey(10)
                    cap.release()
                    cv2.destroyAllWindows()

                # Saving the numPy array
                np.save(os.path.join(self.landmarks_path, item.split(".mp4")[0]), np.array(landmarks_list))
                
                # Converting and storing the array into parquet file
                self.parquet_writer(np.array(landmarks_list), item.split('.mp4')[0])

    def landmark_detection(self, frame, holistic):

        return frame, None

    def landmark_extraction(self, results):
        
        return np.array([])

    def parquet_writer(self, np_array, video_id):
        np_array_flat = np_array.flatten()
        pa_array = pa.array(np_array_flat) 
        table = pa.Table.from_arrays([pa_array], names=[video_id])  
        writer = pq.ParquetWriter(os.path.join(self.landmarks_path, video_id + '.parquet'), table.schema)  
        writer.write_table(table) 
        writer.close()  


videos_path = "C:\\Users\\saran\\OneDrive\\Desktop\\video_dummy_data\\"
landmarks_path = "C:\\Users\\saran\\OneDrive\\Desktop\\new_extracted_landmarks\\"

processor = VideoProcessor(videos_path, landmarks_path)
processor.process_videos()
