from dataclasses import dataclass
from typing import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
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
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

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

class CausalSelfAttension(nn.Module):
    """看i时，从头回看"""
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = self.n_embd // self.n_head

        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd, config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer('mask', mask)

    @override
    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(C, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

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

    attn = CausalSelfAttension(gpt)
    y = attn(x)
    print(f'attention 形状: {tuple(y.shape)}, (2, 128, 512)')
