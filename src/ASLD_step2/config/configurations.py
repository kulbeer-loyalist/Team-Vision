"""
author: @kumar Dahal
time: Jun 11 2023
this class is created to assamble entity and properties(config/config.yaml) 
and will return path to entity
"""
from ASLD_step2.constants import *
from ASLD_step2.utils import read_yaml,create_directories
from ASLD_step2.entity import DataIngestionConfig,DataValidationConfig
from ASLD_step2.Exception import ASLDException

import sys

class Configuration:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH
                ):
        #reading the config.yaml file by importing from constants
        self.config = read_yaml(config_filepath)
        #creating parent directories entity
        create_directories([self.config.artifacts_root])


    """
    this function is written to create the skeletion of DataIngestion entity 
    and will return path for DataIngestionConfig
    """  
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        #read config file to get data_ingestion attributes of DataIngestion
        try:
            config = self.config.data_ingestion
            #create directories for attributes of data_ingestion
            create_directories([config.root_dir])
            create_directories([config.train_dir])
            create_directories([config.test_dir])
            
            #assign the entity with its properities and return it
            data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir,
                source_URL = config.source_URL,
                local_data_file = config.local_data_file,
                unzip_dir = config.unzip_dir,
                train_dir = config.train_dir,
                test_dir = config.test_dir
            )

            return data_ingestion_config  
        except Exception as e:
            raise ASLDException(e,sys)
        

    def get_data_validation_config(self) -> DataValidationConfig:
        pass    




