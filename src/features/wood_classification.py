from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from src.utils.utils import get_paths_and_labels_for_sort_cls
from src.models.models import PretrainedModel
from src.data.dataset import FrontDataset
from definitions import DATA_DIR, CHECKPOINTS_DIR


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          save_path: Union[str, Path],
          device="cpu",
          save_freq: int = 1,
          epochs: int = 100,
          verbose: bool = True) -> Tuple[List[float], List[float]]:
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(1, epochs + 1), total=epochs):
        average_train_loss = 0
        model.train()
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predicted_labels = model(images)
            loss = F.cross_entropy(predicted_labels, labels.long())
            loss.backward()
            optimizer.step()
            average_train_loss += loss.item()
            # print(f"train batch loss: {loss.item()}")
        average_train_loss /= len(train_dataloader)

        average_val_loss = 0
        model.eval()
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predicted_labels = model(images)
            loss = F.cross_entropy(predicted_labels, labels.long())
            average_val_loss += loss.item()
        average_val_loss /= len(train_dataloader)

        train_loss_list.append(average_train_loss)
        val_loss_list.append(average_val_loss)
        if verbose:
            print(f"epoch : {epoch}, average train loss {average_train_loss}, average val loss {average_val_loss}")
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_list,
                'val_loss': val_loss_list
            }, Path(save_path).joinpath(f"epoch_{epoch}.pth"))
    return train_loss_list, val_loss_list


if __name__ == '__main__':
    pass
    # k_train = 64
    # k_val = 8
    #
    # scale = 0.1
    # batch_size = 8
    #
    # with open(DATA_DIR.joinpath('train_paths.txt')) as f:
    #     train_paths = f.readlines()
    # train_paths = [p.strip() for p in train_paths]
    # train_path_list, train_labels = get_paths_and_labels_for_sort_cls(train_paths)
    # train_path_list = train_path_list[:k_train]
    # train_labels = train_labels[:k_train]
    #
    # train_dataset = FrontDataset(train_path_list, train_labels, scale=scale)
    #
    # with open(DATA_DIR.joinpath('val_paths.txt')) as f:
    #     val_paths = f.readlines()
    # val_paths = [p.strip() for p in val_paths]
    # val_path_list, val_labels = get_paths_and_labels_for_sort_cls(val_paths)
    # val_path_list = val_path_list[:k_val]
    # val_labels = val_labels[:k_val]
    #
    # val_dataset = FrontDataset(val_path_list, val_labels, scale=scale)
    #
    # print(len(train_dataset), len(val_dataset))
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    #
    # device = "cuda"
    # model_name = "resnet"
    # lr = 0.01
    # model = PretrainedModel(output_size=4, model_name=model_name).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # save_path = CHECKPOINTS_DIR.joinpath(f"wood_classification/{model_name}_example")
    # save_path.mkdir(parents=True, exist_ok=True)
    #
    # train_loss_array, val_loss_array = train(model, optimizer, train_dataloader, val_dataloader, save_path,
    #                                          device=device, epochs=1)
