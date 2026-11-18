import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from process.tokenizer import decode, load_vocab

# 项目根目录 = 本文件所在目录（train.py 在根目录）
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT = DATA_DIR / 'poetry-set' / 'checkpoint.pt'
# 最优权重（纯 state_dict，不含优化器状态），早停后回滚用它
BEST_MODEL = DATA_DIR / 'poetry-set' / 'best.pt'

# 数据集用 memmap 按需读盘，不占内存（make encode 之后才存在）
train_data: np.memmap = np.memmap(DATA_DIR / 'poetry-set' / 'train.bin', dtype=np.uint16, mode='r')
# eval.bin = 训练中监控/早停用的验证集（不参与梯度）
eval_data: np.memmap = np.memmap(DATA_DIR / 'poetry-set' / 'eval.bin', dtype=np.uint16, mode='r')
# test.bin = 最后只评估一次的测试集
test_data: np.memmap = np.memmap(DATA_DIR / 'poetry-set' / 'test.bin', dtype=np.uint16, mode='r')

# 优先用独立 GPU；AMD 卡要装 ROCm 并设 HSA_OVERRIDE_GFX_VERSION=10.3.0
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainConfig:
    """训练的所有参数"""
    batch_size: int = 64        # 每步并行的序列条数
    block_size: int = 128       # 每条序列长度，和 GPTConfig 保持一致
    max_steps: int = 5000       # 总步数
    lr: float = 3e-4            # 峰值学习率
    min_lr: float = 3e-5        # cosine 衰减的终点（峰值的 1/10）
    warmup_steps: int = 100     # 前一百步线性升温，防止开局大梯度把权重打飞
    log_interval: int = 10         # 每多少步向 stderr 打一行训练进度
    eval_interval: int = 200    # 每多少步评估并存一次 checkpoint
    eval_iters: int = 200       # 评估时平均多少个 batch，越大越准越慢
    patience: int = 4           # 验证集连续多少次评估没改善就早停
    min_delta: float = 0.001    # 改善幅度小于这个值不算有改善
    grad_clip: float = 1.0      # 梯度范数裁剪，防个别 batch 爆梯度


def get_batch(cfg, split):
    """随机取一批 block_size 窗口，x 是原文，y 是右移一位的答案"""
    data = {'train': train_data, 'eval': eval_data, 'test': test_data}[split]
    # -1 是给 y 的右移留出界余量
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + cfg.block_size + 1].astype(np.int64)) for i in ix])
    return x, y


def get_lr(cfg, step):
    """先线性 warmup，再 cosine 衰减到 min_lr"""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def estimate_loss(model, cfg, splits=('train', 'eval')):
    """在指定集合上各平均 eval_iters 个 batch 的 loss，eval 模式顺带关掉 dropout"""
    model.eval()
    out = {}
    for split in splits:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(cfg, split)
            _, loss = model(x.to(device), y.to(device))
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(model, cfg, max_new_tokens=200, temperature=0.8, top_k=50):
    """从 <bos> 开始逐字采样一首诗，遇到 <eos> 提前收笔"""
    stoi = load_vocab()['stoi']
    re_vocab = load_vocab('re_vocab.json')
    x = torch.full((1, 1), stoi['<bos>'], dtype=torch.int64, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -cfg.block_size:])      # 位置编码只认 128 以内，超长就掐头
        logits = logits[:, -1, :] / temperature        # 只取最后一个位置，温度越高越放飞
        v, _ = torch.topk(logits, top_k)               # 只在概率前 top_k 的字里采样
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)
        if next_id.item() == stoi['<eos>']:
            break
    ids = x[0].tolist()[1:]                            # 去掉开头的 <bos>
    return ''.join(c for c in decode(ids, re_vocab) if c not in ('<bos>', '<eos>'))


if __name__ == '__main__':
    torch.manual_seed(1337)
    cfg = TrainConfig()
    print(cfg)

    model = GPT(GPTConfig()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'设备: {device}，参数量: {n_params:,}, 22,727,680 (约22.73M)')

    # 冒烟自检：随机权重 loss 应接近理论值 ln(vocab_size)
    x, y = get_batch(cfg, 'train')
    _, loss = model(x.to(device), y.to(device))
    print(f'随机权重 loss: {loss.item():.4f}, 理论值 {math.log(GPTConfig().vocab_size):.4f}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()

    start = time.time()
    best_eval, best_step, bad_count = float('inf'), 0, 0
    for step in range(cfg.max_steps + 1):
        # cosine 调度：手动把本步的学习率写进优化器
        for group in optimizer.param_groups:
            group['lr'] = get_lr(cfg, step)

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, cfg)
            print(f'step {step:5d} | train {losses["train"]:.4f} | eval {losses["eval"]:.4f} | {time.time() - start:6.1f}s')
            # 续训用的完整存档，每次覆盖
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, CHECKPOINT)

            if losses['eval'] < best_eval - cfg.min_delta:
                best_eval, best_step, bad_count = losses['eval'], step, 0
                torch.save(model.state_dict(), BEST_MODEL)
                print(f'  ↑ 新最优 eval {best_eval:.4f} @ step {step}，已存 {BEST_MODEL.name}')
            else:
                bad_count += 1
                print(f'  — eval 未改善 {bad_count}/{cfg.patience}')
                if bad_count >= cfg.patience:
                    print(f'早停 @ step {step}，最优 eval {best_eval:.4f} @ step {best_step}')
                    break

        x, y = get_batch(cfg, 'train')
        logits, loss = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # 细粒度进度：每 log_interval 步一行，走 stderr，不影响 stdout 的评估主线
        if step % cfg.log_interval == 0:
            elapsed = time.time() - start
            speed = (step + 1) / elapsed if elapsed > 0 else 0.0
            eta = (cfg.max_steps - step) / speed if speed > 0 else 0.0
            print(f'\r[step {step:5d}/{cfg.max_steps}] loss {loss.item():.4f} lr {optimizer.param_groups[0]["lr"]:.2e} {speed:.1f} it/s | ETA {eta/60:.1f} min', file=sys.stderr, end='', flush=True)

    # 回滚到验证集上最优的那一版，并用 test 只评估这一次（真实泛化成绩）
    print('--- 加载最优权重 ---')
    model.load_state_dict(torch.load(BEST_MODEL))
    test_loss = estimate_loss(model, cfg, splits=('test',))
    print(f'最优 eval {best_eval:.4f} @ step {best_step} | 最终 test loss {test_loss["test"]:.4f}')

    print('--- 采样一首 ---')
    print(generate(model, cfg))
