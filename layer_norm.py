import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, cfg, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(cfg["emb_dim"]))   # scale
        self.beta = nn.Parameter(torch.zeros(cfg["emb_dim"]))   # shift

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta