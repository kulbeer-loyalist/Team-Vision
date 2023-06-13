from ASLD_step2.config import configurations
import os
from ASLD_step2.Exception import ASLDException
import sys
from ASLD_step2.logger import logging
class DataValidation:
    def __init__(self,):
        pass

    def filter_folder(folder_path):
        # Get a list of all files in the folder
        try:
            files = os.listdir(folder_path)

            for file in files:
                file_path = os.path.join(folder_path, file)++++++++++++++

                # Check if the file is of type .mp4 or .csv
                if file.endswith(".mp4") or file.endswith(".csv"):
                    logging.info('file is safe {file}')
                else:
                    # Delete the file if it's not of the desired types
                    os.remove(file_path)
                    logging.info('Unwanted video deleted successfully')
        except Exception as e:
            raise ASLDException(e,sys) from e            