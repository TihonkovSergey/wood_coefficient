from typing import Union
from pathlib import Path

import cv2
import numpy as np

from src.data.data_preprocessing import prepare_image


def get_image_by_path(path: Union[str, Path], prepared: bool = True, scale=0.5) -> np.ndarray:
    img = cv2.imread(str(path))
    if prepared:
        img = prepare_image(img, scale=scale)
    return img
