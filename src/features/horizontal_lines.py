import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
import cv2
import imageio
import imutils

from src.data.data_load import get_image_by_path
from src.data.data_preprocessing import get_middle_strip, filter_image
from definitions import DATA_DIR


def get_lines_from_image(img):
    img_middle_colors = get_middle_strip(img)
    filtered_img = filter_image(img_middle_colors)

    lsd = cv2.ximgproc.createFastLineDetector()
    lines = lsd.detect(filtered_img * 255)

    img_middle_with_lines = lsd.drawSegments(img_middle_colors, lines)
    return lines, img_middle_with_lines


def get_lines_from_image_connectivity(img):
    def fill_component(x, y, color):
        min_x, max_x = w, 0
        min_y, max_y = h, 0

        dxdy = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack.pop()
            used[y][x] = color
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            for dx, dy in dxdy:
                curr_x = x + dx
                curr_y = y + dy
                if 0 <= curr_x < w and 0 <= curr_y < h and img[curr_y][curr_x] and not used[curr_y][curr_x]:
                    stack.append((curr_x, curr_y))
        return {
            "x": {
                "min": min_x,
                "max": max_x,
            },
            "y": {
                "min": min_y,
                "max": max_y,
            }
        }

    img = (img > 0).astype('uint8')
    used = np.zeros_like(img)
    h, w = img.shape[:2]
    color = 0
    line_info_list = []
    y_coords, x_coords = np.where(img)
    for (y, x) in zip(y_coords, x_coords):
        if img[y][x] and not used[y][x]:
            color += 1
            line_info = fill_component(x, y, color)
            line_info_list.append(line_info)
    return line_info_list, used


if __name__ == '__main__':
    pass
