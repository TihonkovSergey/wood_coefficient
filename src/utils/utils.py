from pathlib import Path
import os
from typing import Union, List, Tuple, Iterable

import numpy as np
from tqdm import tqdm

import pandas as pd
import json
from src.data.pack_info import get_packs
from src.data.data_load import get_image_by_path
from definitions import DATA_DIR


def fix_front_image_names(front_dir: Union[str, Path]) -> None:
    path_list = get_front_images_paths(front_dir)
    for new_i, p in enumerate(path_list):
        os.rename(p, p.parent.joinpath(f'front{new_i}.jpg'))


def get_valid_paths(fix_front_names: bool = False) -> List[str]:
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


def get_labels(path: Union[str, Path]):
    with open(Path(path).joinpath("info.json")) as file:
        info = json.load(file)
    return np.array(info['labels']).astype(int)


def get_pack_paths(track_dir: Union[str, Path],
                   pack_count: int,
                   max_number_per_pack: int = 5,
                   info_file_name: str = "info",
                   return_numbers: bool = False) -> Union[List[List[Path]], Tuple[List[List[Path]], List[np.ndarray]]]:
    with open(Path(track_dir).joinpath(f"{info_file_name}.json")) as f:
        info = json.load(f)

    if info['pack_count'] == 0:
        return []

    path_list = []
    numbers_list = []

    for i in range(pack_count):
        idx = i if i < info['pack_count'] else info['pack_count'] - 1

        images = info['packs'][str(idx)]
        if len(images) - 2 >= max_number_per_pack:
            images = images[1:-1]

        numbers = np.random.choice(images, min(len(images), max_number_per_pack), replace=False)
        path_list.append([Path(track_dir).joinpath(f'FrontJPG/front{n}.jpg') for n in numbers])
        numbers_list.append(numbers)
    if return_numbers:
        return path_list, numbers_list
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


def get_pack_image(track_dir: Union[str, Path],
                   pack_num: int,
                   info_file_name: str = "info") -> Tuple[np.ndarray, int]:
    with open(Path(track_dir).joinpath(f"{info_file_name}.json")) as f:
        info = json.load(f)

    if pack_num >= info["pack_count"]:
        pack_num = info["pack_count"] - 1
    pack_image_numbers = info["packs"][str(pack_num)]
    n = pack_image_numbers[len(pack_image_numbers) // 2]  # middle image
    img = get_image_by_path(Path(track_dir).joinpath(f"FrontJPG/front{n}.jpg"))
    return img, n


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
