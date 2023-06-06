"""
author @ kumar dahal
date: jun 5 2023
this is the logger of the project
"""

import os
import sys
import logging
import pandas as pd
from ASLD_step2.constants import get_current_time_stamp


LOG_DIR="logs"

"""
this function is written to create the log file ,
where we fetch the constant current time stamp function and create the file so that at each run we can create unique file
"""
def get_log_file_name():
    return f"log_{(get_current_time_stamp)}.log"

LOG_FILE_NAME = get_log_file_name()
#check if exist or not if exist don't create it
os.makedirs(LOG_DIR,exist_ok=True)
#logs/1685974218.log
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)



logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
format='[%(asctime)s]^;%(levelname)s^;%(lineno)d^;%(filename)s^;%(funcName)s()^;%(message)s',
level=logging.INFO
)

"""
this function is written to get the error in particular file line number function name and with  message
"""

def get_log_dataframe(file_path):
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))

    log_df = pd.DataFrame(data)
    columns=["Time stamp","Log Level","line number","file name","function name","message"]
    log_df.columns=columns
    
    log_df["log_message"] = log_df['Time stamp'].astype(str) +":$"+ log_df["message"]

    return log_df[["log_message"]]


