"""
author:
time:
this class is created to retrive the data ,extract the data ,and split the data into train and test 
since we have assigned properties to entity we can use those entity here to perform all those task
"""

from ASLD_step2.entity import DataIngestionConfig
from ASLD_step2.logger import logging
from ASLD_step2.Exception import ASLDException
import sys
import os
import urllib.request as request
from pathlib import Path
from ASLD_step2.utils import get_size
from zipfile import ZipFile
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
    #this function will download the file and store in local_data_file
        try:
            logging.info('Trying to download file')
            #urlretrieve needs url and path to store file
            logging.info("Trying to download file...")

            if not os.path.exists(self.config.local_data_file):

                logging.info("Download started...")
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logging.info(f"{filename} download! with following info: \n{headers}")
            else:
                 logging.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

        except Exception as e:
            raise ASLDException(e,sys)

      


    def check_and_extract_videos(self):
        """
        Check if the artifact/videos folder is empty. If empty, unzip video data from WLASL_videos.zip
        and move only video files into artifact/videos. Then delete the unzip folder. If not empty, return.
        """
        try:
    
            if os.path.exists(self.config.unzip_dir):
                # Unzip video data from WLASL_videos.zip
                with ZipFile(self.config.local_data_file, 'r') as zip_ref:
                    zip_ref.extractall(self.config.unzip_dir)
                logging.info("extraction completed")
            else:
                logging.info("error in Extraction...")
        except Exception as e:
            raise ASLDException(e,sys)     




    def split_videos(self):
        # Load the CSV file
        csv_file = os.path.join(self.config.unzip_dir,'videos_id.csv')
        df = pd.read_csv(csv_file)
        captions = df['caption'].tolist()  # Get the captions column
        ids = df['ID'].tolist()  # Get the IDs column

        # Create a dictionary mapping IDs to captions
        id_to_caption = dict(zip(ids, captions))

        # List all video files in the video folder
        video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

        # Split the videos
        videos_train, videos_test = train_test_split(video_files, train_size=train_ratio, random_state=42)

        # Create train and test folders if they don't exist
        train_folder = "train"
        test_folder = "test"
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Move videos to train folder and label them
        for video in videos_train:
            video_id = os.path.splitext(video)[0]
            caption = id_to_caption.get(video_id, "Unknown")  # Get the caption for the video ID
            labeled_video = f"{caption}_{video}"
            shutil.move(os.path.join(video_folder, video), os.path.join(train_folder, labeled_video))

        # Move videos to test folder and label them
        for video in videos_test:
            video_id = os.path.splitext(video)[0]
            caption = id_to_caption.get(video_id, "Unknown")  # Get the caption for the video ID
            labeled_video = f"{caption}_{video}"
            shutil.move(os.path.join(video_folder, video), os.path.join(test_folder, labeled_video))

        print("Videos split into train and test sets.")


    # Example usage
    csv_file = "captions.csv"
    video_folder = "videos"
    train_ratio = 0.8

    split_videos(csv_file, video_folder, train_ratio)


