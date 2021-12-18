import json
from copy import copy
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple, Sequence

import numpy as np
import cv2
from joblib import Parallel, delayed

from src.data.data_load import get_image_by_path
from src.data.data_preprocessing import get_middle_strip, filter_image
from definitions import DATA_DIR, N_STRIPS


def get_lines_from_image(img: np.ndarray) -> Tuple[Sequence, np.ndarray]:
    img_middle_colors = get_middle_strip(img)
    filtered_img = filter_image(img_middle_colors)

    lsd = cv2.ximgproc.createFastLineDetector()
    lines = lsd.detect(filtered_img * 255)

    img_middle_with_lines = lsd.drawSegments(img_middle_colors, lines)
    return lines, img_middle_with_lines


def get_lines_from_image_connectivity(img: np.ndarray,
                                      return_used: bool = False) -> Union[List[dict], Tuple[List[dict], np.ndarray]]:
    def fill_component(x: int, y: int, color: int) -> dict:
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
                "min": int(min_x),
                "max": int(max_x),
            },
            "y": {
                "min": int(min_y),
                "max": int(max_y),
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
    if return_used:
        return line_info_list, used
    return line_info_list


def all_lines_for_track(path: Union[str, Path], scale: float = 0.25) -> dict:
    def pipeline(path: Union[str, Path], scale: float) -> List[dict]:
        img = get_image_by_path(path, scale=scale, prepared=True)
        filtered_img = filter_image(img)
        return get_lines_from_image_connectivity(filtered_img)

    path_to_front_dir = Path(path).joinpath("FrontJPG")
    result = []
    try:
        n = len(list(path_to_front_dir.glob("*.jpg")))
        result = Parallel(n_jobs=8)(
            delayed(pipeline)(
                path_to_front_dir.joinpath(f"front{i}.jpg"), scale
            )
            for i in range(n)
        )

        return {
            "path": str(path),
            "lines": result,
        }
    except Exception as e:
        return {
            "path": str(path),
            "lines": result,
            "error": str(e),
        }


def count_line_metrics(all_lines: dict,
                       w: int = 480) -> Tuple[List[int], List[int], List[int], List[int]]:
    left_strip_bound = (w // 2) - int(w / (2 * N_STRIPS))
    right_strip_bound = (w // 2) + int(w / (2 * N_STRIPS))
    w_strip = right_strip_bound - left_strip_bound
    length_threshold = w_strip * 0.3

    full_in_strip = []
    start_in_strip = []
    end_in_strip = []
    for lines in all_lines["lines"]:
        c_full = 0
        c_start = 0
        c_end = 0
        for line in lines:
            start = line["x"]["min"]
            end = line["x"]["max"]
            length = end - start
            if start <= left_strip_bound and end >= right_strip_bound:
                c_full += 1
                continue
            if start > left_strip_bound \
                    and end < right_strip_bound \
                    and length > length_threshold:
                c_start += 1
                c_end += 1
            else:
                if start < left_strip_bound < end \
                        and (end - left_strip_bound) > length_threshold:
                    c_end += 1
                if end >= right_strip_bound > start \
                        and (right_strip_bound - start) > length_threshold:
                    c_start += 1
        full_in_strip.append(c_full)
        start_in_strip.append(c_start)
        end_in_strip.append(c_end)
    middle_in_strip = [x + y for x, y in zip(start_in_strip, end_in_strip)]
    return full_in_strip, middle_in_strip, start_in_strip, end_in_strip


def apply_rules(full: Union[np.ndarray, List[int]],
                middle: Union[np.ndarray, List[int]],
                start: Union[np.ndarray, List[int]],
                end: Union[np.ndarray, List[int]],
                q_low: float = 0.3,
                q_up: float = 0.6) -> np.ndarray:
    n = len(full)

    full_low = np.quantile(full[:-10], q_low)
    full_low = max(full_low, 2)
    full_up = np.quantile(full[:-10], q_up)

    start_threshold = np.quantile(start, 0.9)
    end_threshold = np.quantile(end, 0.9)
    middle_threshold = np.quantile(middle, 0.9)

    labels = np.zeros(n)
    prev_label = -1
    for i in range(n):
        if i < 2 or i > n - 5:
            prev_label = -1
            labels[i] = prev_label
            continue
        if full[i] <= 2:  # background
            if end[i] > end_threshold or start[i] > start_threshold or middle[i] > middle_threshold:
                prev_label = 1
                labels[i] = prev_label
                continue

            prev_label = -1
            labels[i] = prev_label
            continue

        if full[i] > full_up:  # pack
            prev_label = 2
            labels[i] = prev_label
            continue

        if end[i] > end_threshold or start[i] > start_threshold or middle[i] > middle_threshold:
            prev_label = 1
            labels[i] = prev_label
            continue

        if full[i] <= full_low:  # background
            prev_label = -1
            labels[i] = prev_label
            continue

        if i < 4 or i > n - 15:
            labels[i] = -1
            continue

        if i > 3 and start[i - 1] > start_threshold or start[i - 2] > start_threshold:
            prev_label = 2
            labels[i] = prev_label
            continue

        labels[i] = prev_label
    return labels


def postproc_labels(labels: Union[np.ndarray, List[int]]) -> Union[np.ndarray, List[int]]:
    new_labels = copy(labels)
    n = len(labels)
    for i in range(1, n - 1):  # сглаживаем случайный пик ...-1 х -1...
        if labels[i] > 0 > labels[i - 1] and labels[i + 1] < 0:
            new_labels[i] = -1

    for i in range(1, n - 2):  # сглаживаем случайный пик ...-1 1 1 -1...
        if labels[i] == 1 and labels[i + 1] == 1 and labels[i - 1] == -1 and labels[i + 2] == -1:
            new_labels[i] = -1
            new_labels[i + 1] = -1

    def find_packs(labels):
        packs = []
        start = None
        for i in range(1, n):
            if labels[i] == 2 and start is None:
                start = i
            if start is not None and labels[i] != 2:
                packs.append((i - start, start, i - 1))
                start = None
        return packs

    def merge_pack(start, end):
        for shift in range(2, 4):
            left, right = None, None
            if new_labels[start - shift] == 2:
                left = start - shift + 1
                right = start
            elif new_labels[end + shift] == 2:
                left = end
                right = end + shift
            if left is not None:
                for i in range(left, right):
                    new_labels[i] = 2
                return True
        return False

    def delete_pack(start, end):
        for i in range(start, end + 1):
            new_labels[i] = -1

    # слепляем плохие пачки
    packs = find_packs(new_labels)
    repeats = len(packs)
    for _ in range(repeats):
        packs = find_packs(new_labels)
        min_pack = min(packs)

        if min_pack[0] > 3 and len(packs) <= 5:
            break
        elif len(packs) < 3:
            break

        lenght, _, _ = min_pack
        for l, s, e in sorted(packs):
            if l != lenght:
                break
            if merge_pack(s, e):
                break
        packs = find_packs(new_labels)
        min_pack = min(packs)
        if len(packs) > 5:
            if min_pack[0] <= 3:
                delete_pack(min_pack[1], min_pack[2])
            else:
                merge_pack(min_pack[1], min_pack[2])

    return new_labels


def get_image_labels(path: Union[str, Path],
                     q_low: float = 0.3,
                     q_up: float = 0.6) -> Union[np.ndarray, List[int]]:
    with open(Path(path).joinpath("all_lines.json")) as file:
        all_lines = json.load(file)
    full_in_strip, middle_in_strip, start_in_strip, end_in_strip = count_line_metrics(all_lines)
    labels = apply_rules(full_in_strip, middle_in_strip, start_in_strip, end_in_strip, q_low=q_low, q_up=q_up)
    new_labels = postproc_labels(labels)
    return new_labels


def find_all_lines_for_each_track(path_list: List[str]) -> None:
    for p in tqdm(path_list):
        local_path = DATA_DIR.joinpath("part_1").joinpath(p)
        if not local_path.joinpath("all_lines.json").exists():
            all_lines = all_lines_for_track(local_path)
            with open(local_path.joinpath("all_lines.json"), "w") as file:
                json.dump(all_lines, file, indent=4)


if __name__ == '__main__':
    # with open(DATA_DIR.joinpath('valid_paths.txt')) as f:
    #     valid_paths = f.readlines()
    #     valid_paths = [p.strip() for p in valid_paths]
    #
    # find_all_lines_for_each_track(valid_paths)
    pass
