from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from src.data.data_preprocessing import get_middle_strip
from src.data.data_load import get_image_by_path
from src.utils.utils import get_front_images_paths, get_labels, get_paths_and_labels_for_sort_cls
from definitions import DATA_DIR


class FrontSequenceDataset(Dataset):
    def __init__(self, path_list: Sequence[Union[str, Path]], n_strips: int = 5, scale: float = 0.25):
        super().__init__()
        self.path_list = path_list  # лесовозы
        self.n_strips = n_strips
        self.scale = scale
        self.transform = transforms.Normalize(  # ImageNet mean & std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        track_dir = Path(self.path_list[index])
        front_dir = track_dir.joinpath('FrontJPG')
        front_paths = get_front_images_paths(front_dir)
        front_images = torch.stack([self._get_image_tensor(p) for p in front_paths], dim=0)
        labels = torch.tensor(get_labels(track_dir))
        return front_images, labels

    def __len__(self) -> int:
        return len(self.path_list)

    def _get_image_tensor(self, path: Union[str, Path]) -> torch.Tensor:
        img = get_middle_strip(get_image_by_path(path, scale=self.scale))
        front_image = torch.tensor(
            img.transpose(2, 0, 1),
            dtype=torch.float32
        )
        # return img
        return self.transform(front_image)


def collate_fn(elements):
    """
    Prepare batch in dataloader
    :param elements: list of (images, labels)
    :return: 
    """
    images, labels = list(zip(*elements))

    seq_sizes = [image.shape[0] for image in images]

    image_batch = torch.cat(images, dim=0)
    label_batch = torch.full((len(seq_sizes), max(seq_sizes)), -1, dtype=torch.int8)
    for i, label in enumerate(labels):
        label_batch[i, :label.shape[0]] = label

    return image_batch, label_batch, seq_sizes


class FrontDataset(Dataset):
    def __init__(self,
                 path_list: Sequence[Union[str, Path]],
                 labels: Sequence[int],
                 n_strips: int = 5,
                 scale: float = 0.25):
        assert len(path_list) == len(labels)
        super().__init__()
        self.path_list = path_list  # фотографии
        self.labels = labels
        self.n_strips = n_strips
        self.scale = scale
        self.transform = transforms.Normalize(  # ImageNet mean & std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = Path(self.path_list[index])
        img = get_middle_strip(get_image_by_path(img_path, scale=self.scale))
        front_image = torch.tensor(
            img.transpose(2, 0, 1),
            dtype=torch.float32
        )
        return self.transform(front_image), self.labels[index]

    def __len__(self) -> int:
        return len(self.path_list)


if __name__ == '__main__':
    pass
    # with open(DATA_DIR.joinpath('train_paths.txt')) as f:
    #     train_paths = f.readlines()
    # train_paths = [p.strip() for p in train_paths]
    #
    # path_list, labels = get_paths_and_labels_for_sort_cls(train_paths)
    #
    # dataset = FrontDataset(path_list, labels)
    #
    # k = 30
    # elements = []
    # for i in range(k):
    #     elements.append(dataset[i])
