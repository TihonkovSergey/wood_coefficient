import json

import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import cv2

from src.data.data_load import get_image_by_path
from src.utils.utils import get_front_images_paths
from src.data.data_preprocessing import filter_image
from definitions import DATA_DIR

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
