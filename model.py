from dataclasses import dataclass
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

   




if __name__ == '__main__':
    gpt = GPTConfig()
    print(gpt)
