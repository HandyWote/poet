import os
import json
import random
from pathlib import Path



# 项目根目录 = 本文件所在目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'


def extra_poetry(path=DATA_DIR / 'chinese-poetry-raw' / '全唐诗', dynasty='song'):
    """读取数据抽取古诗"""

    poems = ''
    with open(DATA_DIR / 'poetry-set' / 'pretice_set.json', 'w', encoding='utf-8') as f:
        for i in os.listdir(path):
            if dynasty not in i or 'authors' in i:
                continue
            with open(path / i, 'r', encoding='utf-8') as rf:
                poems = json.load(rf)
            for poem in poems:
                json.dump({'title': poem['title'], 'content': '\n'.join(poem['paragraphs'])}, f, ensure_ascii=False)
                f.write('\n')


def gen_set():
    """读取提取的古诗，随机排序之后产生 70% 的训练集 30% 的测试集到 DATA_DIR / 'poetry-set' /"""
    path = DATA_DIR / 'poetry-set' / 'pretice_set.json'
    poems = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                poems.append(json.loads(line))

    random.seed(42)
    random.shuffle(poems)

    split = int(len(poems) * 0.7)
    train = poems[:split]
    test = poems[split:]

    with open(DATA_DIR / 'poetry-set' / 'train_set.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / 'poetry-set' / 'test_set.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f'共 {len(poems)} 首：训练 {len(train)} 首，测试 {len(test)} 首')


if __name__ == '__main__':
    extra_poetry()
    gen_set()
