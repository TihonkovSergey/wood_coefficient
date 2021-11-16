import os
import numpy as np
from pathlib import Path


ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR.joinpath("data")

SEED = 0

DISTORTION_COEFFICIENTS = np.array([-2.05e-5, 0, 0, 0])
