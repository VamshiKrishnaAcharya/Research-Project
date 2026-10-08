"""Heterogeneous detectors. Both take a flat feature vector (B, F) and return logits (B,)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import config


class _SeqInput:
    """Cut a flat feature vector into SEQ_LEN chunks (zero-padded)."""

    @staticmethod
    def setup(n_features: int):
        seq_len = min(config.SEQ_LEN, n_features)
        chunk = math.ceil(n_features / seq_len)
        pad = seq_len * chunk - n_features
        return seq_len, chunk, pad

    @staticmethod
    def apply(x, seq_len, chunk, pad):
        if pad:
            x = Fn.pad(x, (0, pad))
        return x.reshape(x.shape[0], seq_len, chunk)


class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.seq_len, self.chunk, self.pad = _SeqInput.setup(n_features)
        self.lstm = nn.LSTM(self.chunk, config.HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(config.DROPOUT)
        self.head = nn.Linear(2 * config.HIDDEN, 1)

    def forward(self, x):
        x = _SeqInput.apply(x, self.seq_len, self.chunk, self.pad)
        _, (h, _) = self.lstm(x)
        z = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(self.drop(z)).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.seq_len, self.chunk, self.pad = _SeqInput.setup(n_features)
        self.embed = nn.Linear(self.chunk, config.D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, config.D_MODEL))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, nhead=config.N_HEAD,
            dim_feedforward=2 * config.D_MODEL, dropout=config.DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.N_LAYERS, enable_nested_tensor=False)
        self.head = nn.Linear(config.D_MODEL, 1)

    def forward(self, x):
        x = _SeqInput.apply(x, self.seq_len, self.chunk, self.pad)
        z = self.encoder(self.embed(x) + self.pos).mean(dim=1)
        return self.head(z).squeeze(-1)


def build_model(arch: str, n_features: int) -> nn.Module:
    if arch == "bilstm":
        return BiLSTMClassifier(n_features)
    if arch == "transformer":
        return TransformerClassifier(n_features)
    raise ValueError(f"unknown architecture: {arch}")
