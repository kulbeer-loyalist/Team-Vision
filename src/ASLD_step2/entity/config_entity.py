from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    train_dir: Path
    test_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    message: str

@dataclass(frozen=True)
class DataTransformationConfig:
    transformed_train_dir:Path
    transformed_test_dir:Path
    preprocessed_object_file_path: Path

