from pathlib import Path
from typing import Union

import numpy as np
import cv2

from definitions import DISTORTION_COEFFICIENTS


def undistort_image(img: np.ndarray):
    w = img.shape[1]
    h = img.shape[0]
    focus = 10
    calibration_matrix = np.array([
        [focus, 0, 0.5 * w],
        [0, focus, 0.5 * h],
        [0, 0, 1]
    ])
    return cv2.undistort(img, calibration_matrix, DISTORTION_COEFFICIENTS)


def down_scale_image(img: np.ndarray, scale: float = 1.):
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def prepare_image(img, undistort=True, scale: float = 1.):
    undistorted_img = undistort_image(img) if undistort else img
    return down_scale_image(undistorted_img, scale)


def get_middle_strip(img, n_strips=5):
    w = img.shape[1]
    left_strip_bound = (w // 2) - int(w / (2 * n_strips))
    right_strip_bound = (w // 2) + int(w / (2 * n_strips))
    return img[:, left_strip_bound: right_strip_bound]


def get_filtered_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_middle = get_middle_strip(gray)
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

    gradient = cv2.filter2D(img_middle, -1, kern)
    binary_img = (gradient > 8).astype('uint8')

    open_kernel = np.ones((2, 15))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, open_kernel)

    close_kernel = np.ones((2, 30))
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, close_kernel)


if __name__ == '__main__':
    pass
