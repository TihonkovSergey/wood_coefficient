import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class PretrainedModel(nn.Module):
    def __init__(self,
                 output_size: int,
                 model_name: str = "resnet"
                 ):
        super().__init__()
        if model_name == "resnet":
            self.model = models.resnet18(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, output_size)

        elif model_name == "densenet":
            self.model = models.densenet121(pretrained=True)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, output_size)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.model(x)


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int,
                 device: str = "cpu"):
        super().__init__()
        self.device = device

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.end_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0, c0 = self._init_hidden(batch_size)
        out, _ = self.lstm(x, (h0, c0))

        out = self.end_fc(out)
        return out

    def _init_hidden(self, batch_size: int):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h, c


class LSTMModelWithEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 model_name: str = "resnet",
                 device: str = "cpu"):
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.encoder = PretrainedModel(input_size, model_name)
        self.lstm = LSTMModel(input_size, output_size, hidden_size, num_layers, device)

    def forward(self, x, seq_sizes=None):
        z = self.encoder(x)
        if seq_sizes is None:
            batch = z[None, :].to(self.device)
        else:
            embeddings = torch.split(z, seq_sizes)
            batch = torch.full(
                (len(seq_sizes), max(seq_sizes), self.input_size),
                0,
                dtype=torch.float32
            ).to(self.device)

            for i, embedding in enumerate(embeddings):
                batch[i, :embedding.shape[0]] = embedding

        return self.lstm(batch)
