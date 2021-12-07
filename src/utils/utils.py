from pathlib import Path
import os
from tqdm import tqdm

import pandas as pd

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


if __name__ == '__main__':
    # valid_paths = get_valid_paths()
    # with open(DATA_DIR.joinpath('valid_paths.txt'), 'w') as f:
    #     f.writelines([p + "\n" for p in valid_paths])
    pass
