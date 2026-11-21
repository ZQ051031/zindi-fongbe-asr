import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .audio import FEATURES


class Encoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Conv1d(FEATURES, width, 5, padding=2, stride=2)
        self.context = nn.GRU(width, width, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)

    @staticmethod
    def output_lengths(lengths):
        return (lengths + 1) // 2

    def forward(self, frames, lengths=None):
        frames = self.proj(frames.transpose(1, 2)).transpose(1, 2).relu()
        if lengths is None:
            return self.context(frames)[0]
        packed = pack_padded_sequence(frames, self.output_lengths(lengths),
                                      batch_first=True, enforce_sorted=False)
        return pad_packed_sequence(self.context(packed)[0], batch_first=True)[0]


class AcousticModel(nn.Module):
    def __init__(self, classes, width=96):
        super().__init__()
        self.encoder = Encoder(width)
        self.classifier = nn.Linear(2 * width, classes)

    def forward(self, frames, lengths=None):
        return self.classifier(self.encoder(frames, lengths)).log_softmax(-1)


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = AcousticModel(len(state["units"]), state["width"])
    model.load_state_dict(state["model"])
    return model.eval(), state["units"]
