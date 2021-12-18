from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Tuple, Union, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.models import LSTMModelWithEncoder as LSTM
from src.data.dataset import FrontSequenceDataset, collate_fn
from definitions import DATA_DIR, CHECKPOINTS_DIR


def labels_to_track_info(labels: Sequence[int]) -> dict:
    packs = []
    backgrounds = [0]
    edges = []

    for i in range(1, len(labels)):
        if labels[i] == -1:
            backgrounds.append(i)
        elif labels[i] == 1:
            edges.append(i)
        elif labels[i] == 2 and labels[i] == labels[i - 1]:
            packs[-1].append(i)
        else:
            packs.append([i])
    return {
        "labels": list(labels),
        "pack_count": len(packs),
        "packs": {
            i: packs[i] for i in range(len(packs))
        },
        "back": backgrounds,
        "edges": edges
    }


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          save_path: Union[str, Path],
          device: str = "cpu",
          save_freq: int = 1,
          epochs: int = 100,
          verbose: bool = True) -> Tuple[List[float], List[float]]:
    def calculate_loss(predicted_labels: torch.Tensor,
                       labels: torch.Tensor) -> Any:
        batch_size = labels.shape[0]
        predicted_labels = predicted_labels.permute(0, 2, 1)
        loss = F.cross_entropy(predicted_labels, labels.long(), ignore_index=-1, reduction="none")
        loss = loss.sum() / batch_size
        return loss

    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(1, epochs + 1), total=epochs):
        average_train_loss = 0
        model.train()
        for images, labels, sizes in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predicted_labels = model(images, sizes)
            loss = calculate_loss(predicted_labels, labels)
            loss.backward()
            optimizer.step()
            average_train_loss += loss.item()
            # print(f"batch loss: {loss.item()}")
        average_train_loss /= len(train_dataloader)

        average_val_loss = 0
        model.eval()
        for images, labels, sizes in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predicted_labels = model(images, sizes)
            loss = calculate_loss(predicted_labels, labels)
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
    # train_k = 8
    # val_k = 4
    # with open(DATA_DIR.joinpath('valid_paths.txt')) as f:  # _good_paths
    #     valid_paths = f.readlines()
    # valid_paths = [p.strip() for p in valid_paths]
    # valid_paths = [DATA_DIR.joinpath('part_1').joinpath(p) for p in valid_paths]
    #
    # train_paths = valid_paths[:train_k]
    # val_paths = valid_paths[train_k: train_k + val_k]
    #
    # train_dataloader = DataLoader(
    #     FrontSequenceDataset(train_paths, scale=0.25),
    #     batch_size=1,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=1,
    #     drop_last=False
    # )
    #
    # val_dataloader = DataLoader(
    #     FrontSequenceDataset(train_paths, scale=0.25),
    #     batch_size=1,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     num_workers=1,
    #     drop_last=False
    # )
    #
    # device = "cuda"
    # lr = 0.01
    # lstm = LSTM(
    #     input_size=128,
    #     output_size=3,
    #     hidden_size=128,
    #     num_layers=2,
    #     model_name="resnet",
    #     device=device
    # )
    # lstm.to(device)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=lr)
    #
    # save_path = CHECKPOINTS_DIR.joinpath("matching/lstm_resnet_example")
    # save_path.mkdir(parents=True, exist_ok=True)
    #
    # train_loss_array, val_loss_array = train(lstm, optimizer, train_dataloader, val_dataloader, save_path,
    #                                          device=device, epochs=1)
