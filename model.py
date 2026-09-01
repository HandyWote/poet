from dataclasses import dataclass
from typing import override
import torch
import torch.nn as nn
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VOCAB_PATH = PROJECT_ROOT / 'data' / 'poetry-set'

def load_vocab_size(name='vocab.json' ,path=VOCAB_PATH):
    with open(path / name, 'r', encoding='utf-8') as f:
        return json.load(f)['vocab_size']

@dataclass
class GPTConfig:
    """模型的所有参数"""
    vocab_size: int = load_vocab_size()
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

class LayerNorm(nn.Module):
    """标准化向量"""
    def __init__(self, n_embd, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    @override
    def forward(self, x):
        return nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    """消化信息 512 -> 2048 -> 512"""
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)

    @override
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

if __name__ == '__main__':
    gpt = GPTConfig()
    print(gpt)
    
    ln_mine = LayerNorm(gpt.n_embd, gpt.bias)
    ln_ref = nn.LayerNorm(gpt.n_embd, gpt.bias)
    ln_ref.weight.data.copy_(ln_mine.weight.data)
    x = torch.randn(2, 128, gpt.n_embd)
    y_mine = ln_mine(x)
    y_ref = ln_ref(x)
    diff = (y_mine-y_ref).abs().max().item()
    print(f'{diff:.2e}')

    mlp = MLP(gpt)
    y = mlp(x)
    print(f'y_shape: {tuple(y.shape)}')
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f'MLP的参数量: {n_params:,}')
