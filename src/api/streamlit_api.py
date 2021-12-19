import os
import json
from pathlib import Path
from typing import Tuple, Union, List, Iterable

import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = ROOT_DIR.joinpath("data")
CHECKPOINTS_DIR = DATA_DIR.joinpath("checkpoints")

SEED = 0

DISTORTION_COEFFICIENTS = np.array([-2.05e-5, 0, 0, 0])


def undistort_image(img: np.ndarray) -> np.ndarray:
    w = img.shape[1]
    h = img.shape[0]
    focus = 10
    calibration_matrix = np.array([
        [focus, 0, 0.5 * w],
        [0, focus, 0.5 * h],
        [0, 0, 1]
    ])
    return cv2.undistort(img, calibration_matrix, DISTORTION_COEFFICIENTS)


def down_scale_image(img: np.ndarray, scale: float = 1.) -> np.ndarray:
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def prepare_image(img: np.ndarray,
                  undistort: bool = True,
                  scale: float = 1.) -> np.ndarray:
    undistorted_img = undistort_image(img) if undistort else img
    return down_scale_image(undistorted_img, scale)


def get_image_by_path(path: Union[str, Path], prepared: bool = True, scale=0.5) -> np.ndarray:
    img = cv2.imread(str(path))
    if prepared:
        img = prepare_image(img, scale=scale)
    return img


def get_front_images_paths(front_dir: Union[str, Path]) -> List[Path]:
    path = Path(front_dir)
    names = path.glob("front*")
    return sort_front_images_paths(names)


def sort_front_images_paths(path_list: Iterable[Union[str, Path]]) -> List[Path]:
    pairs = []
    max_i = 0
    for path in path_list:
        p = Path(path)
        i = int(str(p.name).split('.')[0][5:])
        max_i = max(max_i, i)
        pairs.append((i, p))

    pairs = sorted(pairs)
    return [p[1] for p in pairs]


def filter_image(img: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = 10
    kern = cv2.getGaborKernel(
        (size, size),
        2 / 4,
        np.pi / 2,
        9,
        gamma=1,
        psi=np.pi
    )
    kern[kern > 0] /= kern[kern > 0].sum()
    kern[kern < 0] /= np.abs(kern[kern < 0].sum())

    gradient = cv2.filter2D(img_gray, -1, kern)
    binary_img = (gradient > 8).astype('uint8')

    open_kernel = np.ones((2, 15))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, open_kernel)

    close_kernel = np.ones((2, 30))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, close_kernel)
    return binary_img


if __name__ == '__main__':
    with open(DATA_DIR.joinpath('valid_paths.txt')) as f:
        valid_paths = f.readlines()
    valid_paths = [p.strip() for p in valid_paths]

    path_folder = st.selectbox('Путь:', valid_paths)
    track_dir = DATA_DIR.joinpath(f"part_1/{path_folder}")
    img_paths = get_front_images_paths(track_dir.joinpath("FrontJPG/"))
    images = [get_image_by_path(p) for p in img_paths]

    front_image_number = st.slider('Номер фотографии:', 0, len(img_paths) - 1)
    img = images[front_image_number]

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.pyplot(fig)

    filtered_image = filter_image(img)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(cv2.cvtColor(filtered_image * 255, cv2.COLOR_BGR2RGB))
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 2))

    with open(track_dir.joinpath("info.json")) as file:  # info
        info = json.load(file)
    labels = info["labels"]
    labels = [0 if el == -1 else el for el in labels]

    with open(track_dir.joinpath("lstm_info.json")) as file:  # info
        info = json.load(file)
    lstm_labels = info["labels"]
    lstm_labels = [2 if el == 1 else el for el in lstm_labels]

    sns.lineplot(x=range(len(labels)), y=labels, color="g", linewidth=2.5, label="opencv", ax=ax)
    sns.lineplot(x=range(len(labels)), y=lstm_labels, color="r", linewidth=2.5, label="lstm", ax=ax)
    st.pyplot(fig)



