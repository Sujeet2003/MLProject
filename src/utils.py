from pathlib import Path
import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill


def save_object(file_path: Path, object):
    try:
        path_dir = os.path.dirname(file_path)

        os.makedirs(path_dir, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)
    