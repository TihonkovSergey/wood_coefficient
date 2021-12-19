from typing import Union, List
from pathlib import Path
import json

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

from src.utils.utils import get_pack_image
from src.features.horizontal_lines import get_lines_from_image
from definitions import DATA_DIR


def get_features(img, track_dir, img_num) -> dict:
    eps = 1e-5
    ####### Lines from middle strip using FastLineDetector #######
    strip_coefs = []
    strip_lens = []
    lines, _ = get_lines_from_image(img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k = (y2 - y1) / (x2 - x1 + eps)
        strip_coefs.append(k)
        length = (x1 - x2) ** 2 + (y1 - y2) ** 2
        strip_lens.append(length)

    ####### Lines from image using connectivity #######
    with open(Path(track_dir).joinpath(f"all_lines.json")) as f:
        all_lines = json.load(f)

    lines = all_lines["lines"][img_num]
    connectivity_lens = []
    for line in lines:
        x_min = line["x"]["min"]
        x_max = line["x"]["max"]
        length = x_max - x_min
        connectivity_lens.append(length)

    ####### Lines from image using LinReg & pick detector #######
    window_size = 125
    x_size = 10
    y_size = 3
    min_length = 300
    coefficients, filtered_lines, edges, x, y = process_image(img,
                                                              window_size=window_size,
                                                              x_size=x_size,
                                                              y_size=y_size,
                                                              min_length=min_length)
    # get angles
    tans = [k for (k, b) in coefficients]

    # get diameters and lengths
    ys_left, ys_right = [], []
    lengths = []

    for line in filtered_lines:
        y_, x_ = zip(*line)
        x_, y_ = np.array(x_), np.array(y_)

        y_window_left = y_[np.where((x_ > 280) & (x_ < 380))]
        y_window_right = y_[np.where((x_ > 580) & (x_ < 680))]

        if len(y_window_left) > 0:
            ys_left.append(np.mean(y_window_left))

        if len(y_window_right) > 0:
            ys_right.append(np.mean(y_window_right))

        argmin = np.argmin(x_)
        argmax = np.argmax(x_)
        length = np.sqrt((y_[argmax] - y_[argmin]) ** 2 + (x_[argmax] - x_[argmin]) ** 2)
        lengths.append(length)

    ys_left = sorted(ys_left)
    ys_right = sorted(ys_right)
    diameters_left = np.abs(np.diff(ys_left))
    diameters_right = np.abs(np.diff(ys_right))

    ####### Build features #######
    tans_hist, tans_stats = extract_photo_features_geom(tans, 'tans', 10, (-0.5, 0.5))
    lengths_hist, lengths_stats = extract_photo_features_geom(lengths, 'lenghts', 10, (min_length, 960))

    diameters_left_hist, diameters_left_stats = extract_photo_features_geom(diameters_left, 'diameters_left', 10,
                                                                            (0, 200))
    diameters_right_hist, diameters_right_stats = extract_photo_features_geom(diameters_right, 'diameters_right', 10,
                                                                              (0, 200))

    rgb_hist, rgb_stats = extract_photo_features_color(img, n_bins=10, arange=(0, 255))
    strip_lens_hist, strip_lens_stats = extract_photo_features_geom(strip_lens, 'strip_lens', 10, (0, 50000))
    connect_lens_hist, connect_lens_stats = extract_photo_features_geom(connectivity_lens, 'connect_lens', 10, (0, 480))
    strip_coefs_hist, strip_coefs_stats = extract_photo_features_geom(strip_coefs, 'strip_coefs', 10, (-0.5, 0.5))

    pack_features = {
        **rgb_hist, **rgb_stats,
        **strip_lens_hist, **strip_lens_stats,
        **connect_lens_hist, **connect_lens_stats,
        **strip_coefs_hist, **strip_coefs_stats,
        **tans_hist, **tans_stats,
        **lengths_hist, **lengths_stats,
        **diameters_left_hist, **diameters_left_stats,
        **diameters_right_hist, **diameters_right_stats,
    }
    return pack_features



def get_stats(array):
    stats = {}

    if array.size == 0:
        median, std = np.nan, np.nan
    else:
        median = np.median(array, axis=0)
        std = np.std(array, axis=0)

    stats.update({
        'median': median,
        'std': std
    })
    return stats


def extract_photo_features_geom(array, name, n_bins, arange):
    array = np.array(array)

    hist, bins = np.histogram(array, bins=n_bins, range=arange, density=False)
    hist = hist.astype('float64')

    hist *= np.diff(bins)
    hist_dict = {f'hist_{name}_{np.round(from_, 3)}_{np.round(to_, 3)}': v
                 for from_, to_, v in zip(bins[:-1], bins[1:], hist)}
    stats = get_stats(array)
    stats_dict = {f'{name}_{k}': v for k, v in stats.items()}

    return hist_dict, stats_dict


def get_pixel_hist(flat_img, n_bins=256, arange=(0, 255)):
    hist = np.zeros((flat_img.shape[1], n_bins))
    bins = None
    for i in range(len(hist)):
        hist[i], bins = np.histogram(flat_img[:, i], bins=n_bins, range=arange, density=False)
        hist[i] *= np.diff(bins)
    return hist, bins


def extract_photo_features_color(array, n_bins, arange):
    hist_dict, stats_dict = {}, {}
    array = array.reshape(-1, 3)

    hist, bins = get_pixel_hist(array, n_bins=n_bins, arange=arange)
    stats = get_stats(array)

    for i, c in enumerate(['r', 'g', 'b']):
        hist_dict.update({
            f'{c}_hist_{np.round(from_, 3)}_{np.round(to_, 3)}': v
            for from_, to_, v in zip(bins[:-1], bins[1:], hist[i])
        })

        stats_dict.update({
            f'{c}_{k}': v[i]
            for k, v in stats.items()
        })

    return hist_dict, stats_dict


matplotlib.rcParams['figure.figsize'] = (16, 9)


def extract_peaks(img_frame, threshold=150, N=10):
    """
    Ищем кандидатов на границы между бревнами на изображении
    :param img_frame:
    :param threshold:
    :param N:
    :return:
    """
    # for i, c in enumerate(['r', 'g', 'b']):

    ch = img_frame[:, :, 0].copy()

    ch[ch > threshold] = 255
    mean_pix = (np.mean(ch, axis=1))
    mean_pix = np.convolve(mean_pix, np.ones((N,)) / N, mode='valid')

    # поиск экстремумов
    x = mean_pix.max() - mean_pix
    peaks, properties = find_peaks(x, prominence=10)

    return peaks, mean_pix


def get_lines(image, window_size, crop_sides=True):
    """
    Проходимся окном по всему изображению, ищем границы при помощи extract_peaks
    :param image:
    :param window_size:
    :param crop_sides:
    :return:
    """
    peaks_s, mean_pix_s = [], []
    for i in range(0, image.shape[1] - window_size):
        img_frame = image[:, i:i + window_size]
        peaks, mean_pix = extract_peaks(img_frame)
        peaks_s.append(peaks)
        mean_pix_s.append(mean_pix)

    if crop_sides and len(peaks_s) >= 3:
        return peaks_s[1:-1], mean_pix_s[1:-1]
    if crop_sides:
        print('unable to crop sides: grid size too small')
    return peaks_s, mean_pix_s


def peaks_to_points(peaks, win_size, max_x, crop_sides=True):
    """
    Переводит список пиков в координаты точек
    :param peaks:
    :param win_size:
    :param max_x:
    :param crop_sides: этот параметр не работает корректно, вроде бы. Поправка на то, что у окна ненулевой размер
    :return:
    """
    points = []
    for i, peak in enumerate(peaks):
        offset = ((i + 1) if crop_sides else i) + int(win_size / 2)
        for p in peak:
            points.append((min(max_x, int((offset))), p))
    y, x = zip(*points)
    return x, y


def connect_line(edges, x, y, x_size=10, y_size=3):
    """
    Проверяет, есть ли хотя бы одна единица в окне (x_size x y_size), при этом точка (x, y) расположена в _середине_
    окна по OX и в начале - по OY
    :param edges: 2d numpy массив, на котором 0 - нет границы, 1 - точка, в которой скользящим окном найдена граница
    :param x: - текущая координата из списка x-координат точек ("единиц") с edges
    :param y: - текущаая координата из списка y-координат точек ("единиц") с edges
    :param x_size: размер окна по OX
    :param y_size: размер окна по OY
    :return: координаты следующей точки либо None, если линия оборвалась (нет ни одной точки в окне x_size * y_size)
    """
    window = edges[max(0, x - x_size):min(edges.shape[0], x + x_size),
             y + 1:min(edges.shape[1], y + y_size)]
    for j in range(window.shape[1]):
        for i in range(window.shape[0]):
            if window[i, j] and j != x_size:
                window[:, 0] = 0
                return x + i - x_size, y + j + 1
    return None


def group_points_into_lines(edges, x_coords, y_coords, x_size=10, y_size=3):
    """
    Группирует отдельные точки в линии при помощи функции connect_line
    :param edges:
    :param x_coords:
    :param y_coords:
    :param x_size:
    :param y_size:
    :return:
    """
    point_dict = {(x_, y_): i for i, (x_, y_) in enumerate(zip(x_coords, y_coords))}
    point = next(iter(point_dict.keys()))

    lines = []
    line = []
    while True:
        next_point = connect_line(edges, point[0], point[1], x_size, y_size)
        try:
            point_dict.pop(point)
        except KeyError:
            pass

        if next_point is not None:
            line.append(point)
            point = next_point
        else:
            lines.append(line)
            line = []
            try:
                point = next(iter(point_dict.keys()))
            except StopIteration:
                break
    return lines


def find_line_coefficients(lines, min_length=300):
    """
    Линейная регрессия для поиска коэффициентов линий.
    :param lines:
    :param min_length:
    :return:
    """
    coefficients = []
    out_lines = []
    for line in lines:
        if len(line) > 0:
            if len(line) > min_length:
                y_, x_ = zip(*line)
                regr = LinearRegression()
                regr.fit(np.expand_dims(np.array(x_), -1), y_)
                w, b = regr.coef_[0], regr.intercept_
                coefficients.append((w, b))
                out_lines.append(line)
    return coefficients, out_lines


def show_lines(img, coefficients, filtered_lines):
    color = (0, 255, 0)
    thickness = 2

    for (w, b), line in zip(coefficients, filtered_lines):
        _, arange = zip(*line)
        x_min, x_max = int(min(arange)), int(max(arange))
        y_min = int(x_min * w + b)
        y_max = int(x_max * w + b)

        img = cv2.line(img, (x_min, y_min), (x_max, y_max), color, thickness)

    return img


def get_edges(tr_img, window_size=125):
    """

    :param tr_img:
    :param window_size:
    :return:
    """
    crop_sides = True
    peaks, mean_pix = get_lines(tr_img, window_size, crop_sides=crop_sides)

    x, y = peaks_to_points(peaks, window_size, tr_img.shape[1] - 1, crop_sides)
    edges = np.zeros(tr_img.shape[:-1], dtype=np.uint8)
    for x_, y_ in zip(x, y):
        edges[x_, y_] = 1

    return edges, x, y


def process_image(img, window_size=125, x_size=10, y_size=3, min_length=300, show_intermediate=False):
    """
    Полный цикл выделения линий
    :param img: изображение на вход, BGR
    :param window_size: размер окна для поиска границ бревен
    :param x_size: размер окна поиска для соединения линий по OX
    :param y_size: размер окна поиска для соединения линий по OY
    :param min_length: минимальная длина линии, для которой будут вычисляться коэффициенты (на самом деле не длина, а
    кол-во точек)
    :param show_intermediate: показывать ли промежуточный результат (точки, который были определены скользящим окном
    как границы)
    :return: список коэффициентов (w, b), список линий, изображение с визуализацией линий
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges, x, y = get_edges(img, window_size)

    if show_intermediate:
        plt.figure(figsize=(16, 9))
        plt.imshow(img)
        plt.scatter(y, x, s=1)
        plt.show()
    lines = group_points_into_lines(edges, x, y, x_size, y_size)
    coefficients, filtered_lines = find_line_coefficients(lines, min_length)

    return coefficients, filtered_lines, edges, x, y


if __name__ == '__main__':
    truck = '02/133216_М443РР10'
    path = DATA_DIR.joinpath("part_1").joinpath(truck)

    fts = get_features(path, 2, "lstm_info")
    print(fts)
