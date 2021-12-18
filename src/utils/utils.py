from pathlib import Path
import os
from typing import Union, List, Tuple

import numpy as np
from tqdm import tqdm

import pandas as pd
import json
from src.data.pack_info import get_packs
from definitions import DATA_DIR


def fix_front_image_names(front_dir):
    path_list = get_front_images_paths(front_dir)
    for new_i, p in enumerate(path_list):
        os.rename(p, p.parent.joinpath(f'front{new_i}.jpg'))


def get_valid_paths(fix_front_names=False):
    table_nek = pd.read_csv(DATA_DIR.joinpath("table_nek_2020_01.csv"))
    path_list = sorted(table_nek.path.unique())

    valid = []
    for raw_path in tqdm(path_list):
        path = "/".join(raw_path.split('/')[-2:])
        local_path = DATA_DIR.joinpath('part_1').joinpath(path).joinpath('FrontJPG')
        if not local_path.exists():
            continue
        if any(local_path.iterdir()):  # is empty
            valid.append(path)
            if fix_front_names:
                fix_front_image_names(local_path)

    return valid


def get_front_images_paths(front_dir):
    path = Path(front_dir)
    names = path.glob("front*")
    return sort_front_images_paths(names)


def sort_front_images_paths(path_list):
    pairs = []
    max_i = 0
    for p in path_list:
        i = int(str(p.name).split('.')[0][5:])
        max_i = max(max_i, i)
        pairs.append((i, p))

    pairs = sorted(pairs)
    return [p[1] for p in pairs]


def get_labels(path):
    with open(path.joinpath("info.json")) as file:
        info = json.load(file)
    return np.array(info['labels']).astype(int)


def get_pack_paths(track_dir: Union[str, Path], pack_count: int, max_number_per_pack: int = 5) -> List[List[Path]]:
    with open(Path(track_dir).joinpath("info.json")) as f:
        info = json.load(f)

    if info['pack_count'] == 0:
        return []

    path_list = []

    for i in range(pack_count):
        idx = i if i < info['pack_count'] else info['pack_count'] - 1

        images = info['packs'][str(idx)]
        if len(images) - 2 >= max_number_per_pack:
            images = images[1:-1]

        numbers = np.random.choice(images, min(len(images), max_number_per_pack), replace=False)
        path_list.append([track_dir.joinpath(f'FrontJPG/front{n}.jpg') for n in numbers])

    return path_list


def get_paths_and_labels_for_sort_cls(track_paths: List[str]) -> Tuple[List[Path], List[int]]:
    packs = get_packs()
    df = pd.read_csv(DATA_DIR.joinpath("prepared_nek.csv"), index_col=0)
    train_img_paths = []
    train_sort = []

    for p in track_paths:
        n = packs[packs['path'] == p]['packs_count'].values[0]
        pack_paths = get_pack_paths(DATA_DIR.joinpath(f'part_1/{p}'), n, max_number_per_pack=5)
        for i in range(len(pack_paths)):
            res = df[(df["path"] == p) & (df['pack_N'] == i + 1)]['sort'].values
            if len(res) > 0:
                sort = res[0]
                for img_path in pack_paths[i]:
                    train_img_paths.append(img_path)
                    train_sort.append(sort)
    return train_img_paths, train_sort


if __name__ == '__main__':
    pass
    # valid_paths = get_valid_paths()
    #
    # for p in tqdm(valid_paths):
    #     path_from = DATA_DIR.joinpath('part_1').joinpath(p)
    #     path_to = DATA_DIR.joinpath('valid').joinpath(p)
    #     shutil.copytree(path_from.joinpath("FrontJPG"), path_to.joinpath("FrontJPG"))
    #     shutil.copy(path_from.joinpath("all_lines.json"), path_to.joinpath("all_lines.json"))
    #     shutil.copy(path_from.joinpath("info.json"), path_to.joinpath("info.json"))

    # with open(DATA_DIR.joinpath('valid_paths.txt'), 'w') as f:
    #     f.writelines([p + "\n" for p in valid_paths])
