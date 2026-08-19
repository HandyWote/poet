import json


from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# 让 matplotlib 能显示中文（按顺序尝试系统里的中文字体）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 项目根目录 = 本文件所在目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'


def get_counter():
    """读取训练集，统计每个字出现的次数"""
    with open(DATA_DIR / 'poetry-set' / 'train_set.json', 'r', encoding='utf-8') as f:
        poems = json.load(f)
        counter = Counter()
        for poem in poems:
            text = poem['title'] + '\n' + poem['content']
            counter.update(text)
    return counter


def plot_distribution(counter, threshold=10):
    """根据字频统计画两张并排图，帮助决定词表大小"""
    ranked = counter.most_common()
    total = sum(counter.values())
    n = len(ranked)

    freqs = [c for _, c in reversed(ranked)]        # 从最稀有到最常见的次数
    coverages = []                                   # 收前 N 个字的累计覆盖率
    cum = 0
    for _, c in ranked:
        cum += c
        coverages.append(cum / total * 100)

    # 左图：长尾分布
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n + 1), freqs, linewidth=0.8)
    plt.yscale('log')                                # 次数跨度大，用对数刻度
    plt.axhline(threshold, color='red', linestyle='--')
    plt.xlabel('从最稀有的字开始排名')
    plt.ylabel('出现次数')
    plt.title('长尾分布')

    # 右图：累计覆盖率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n + 1), coverages, linewidth=1)
    for level in (99.0, 99.5, 99.9):
        plt.axhline(level, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('词表大小（收前 N 个字）')
    plt.ylabel('覆盖率（%）')
    plt.title('词表大小 vs 覆盖率')
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(DATA_DIR / 'poetry-set' / 'distribution.png')
    plt.show()


def build_vocab(counter, threshold=10):
    """按频率门槛统计词表并保存。出现次数 < threshold 的字归入 <unk>。"""
    total = sum(counter.values())

    # 特殊符固定在最前面
    stoi = {'<unk>': 0, '<bos>': 1, '<eos>': 2}
    re_stoi = {0: '<unk>', 1: '<bos>', 2: '<eos>'}
    # 高频字按频率从高到低依次编号，频率低于门槛就停
    for char, count in counter.most_common():
        if count < threshold:
            break
        stoi[char] = len(stoi)
        re_stoi[len(re_stoi)] = char

    vocab = {
        'stoi': stoi,
        'vocab_size': len(stoi),
        'threshold': threshold,
        'total_chars': total,
    }
    re_vocab = {
        're_stoi': re_stoi,
        'vocab_size': len(re_stoi),
        'threshold': threshold,
        'total_chars': total,
    }
    return vocab, re_vocab

def write_vocab(vocab, re_vocab, save_path=DATA_DIR / 'poetry-set' ):
    with open(save_path / 'vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(save_path / 're_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(re_vocab, f, ensure_ascii=False, indent=2)

def load_vocab(name='vocab.json', save_path=DATA_DIR / 'poetry-set'):
    with open(save_path / name, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def encode(tokens, vocab):
    """把 token 列表映射成 id 列表，词表外的 token 归入 <unk>(0)"""
    index_list = []
    for token in tokens:
        index_list.append(vocab['stoi'].get(token, 0))
    return index_list


def decode(index_list, re_vocab):
    """把 id 列表映射回文字，查不到的显示 <unk>"""
    words = []
    for i in index_list:
        words.append(re_vocab['re_stoi'].get(str(i), '<unk>'))
    return words


def encode_set(set_name):
    """把数据集逐首编码成 token id 序列，落盘为二进制 .bin 供训练直接读取"""
    vocab = load_vocab()
    tokenized_set = []
    with open(DATA_DIR / 'poetry-set' / (set_name+'_set.json'), 'r', encoding='utf-8') as f:
        poems = json.load(f)
        for poem in poems:

            tokens = ['<bos>'] + list(poem['title'] + '\n' + poem['content']) + ['<eos>']
            tokenized_set += encode(tokens, vocab)

    arr = np.array(tokenized_set, dtype=np.uint16)
    arr.tofile(DATA_DIR / 'poetry-set' / (set_name+'.bin'))


if __name__ == '__main__':
    counter = get_counter()
    plot_distribution(counter)
    write_vocab(*build_vocab(counter))
    encode_set('train')
    encode_set('test')
