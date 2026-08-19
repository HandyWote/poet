# poet — 从零训练一个唐诗宋词 GPT

用 PyTorch 从数据清洗到模型训练，完整走一遍"字符级 GPT 生成古体诗"的流程。
当前进度：**数据准备阶段完成**（提取 → 词表 → 编码），下一步实现 Transformer 模型与训练脚本。

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)、Python 3.12+、git。

```bash
uv sync                 # 安装依赖
make                    # 下载数据 → 切分数据集 → 建词表 → 编码 .bin（数据已存在会自动跳过）
```

`make` 一条命令跑通全部数据准备；也可以分步执行：
`make download`（只下载原始数据）、`make split`（只切分）、`make encode`（只建词表编码）、`make clean`（清空产物，保留原始数据）。

## 目录结构

```
poet/
├── Makefile                 # 数据准备流水线入口
├── process/
│   ├── process.py           # 原始数据 → 训练集/测试集 JSON（7:3 随机切分，seed=42）
│   └── tokenizer.py         # 字频统计 → 词表构建 → 数据集编码 .bin
├── docs/
│   └── decision-vocab-threshold.md  # 决策记录：词表频率门槛 T=10 的选择依据
├── data/poetry-set/         # 数据产物（git 忽略，make 本地生成）
│   ├── vocab.json / re_vocab.json   # 7399 token 词表（含 <unk>/<bos>/<eos>）
│   ├── train.bin / test.bin         # uint16 编码数据，训练时 memmap 读取
│   └── distribution.png             # 字频长尾分布与覆盖率图
└── pyproject.toml
```

## 数据

原始语料来自 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
（宋诗 177,973 首 + 唐诗）。`make download` 用稀疏检出只拉取 `全唐诗/` 目录（约 500MB），不克隆整个 2GB+ 仓库。
`data/` 整体被 git 忽略，不随仓库分发，克隆后运行 `make` 即可重建全部产物。

流水线产出：`train_set.json` / `test_set.json`（177,973 / 76,275 首）、
7399 token 词表（门槛 T=10，依据见决策记录）、
`train.bin` / `test.bin`（13,564,418 / 5,806,696 个 uint16 token id）。

## 编码格式约定

每首诗 = `<bos>` + 标题 + `\n` + 正文 + `<eos>`，词表外字符归 `<unk>`(0)。
训练时用 `np.memmap(path, dtype=np.uint16, mode='r')` 随机偏移切取 `block_size=128` 的窗口作为样本。

## 路线图

- [x] 数据提取与切分
- [x] 词表构建（含频率门槛决策记录）
- [x] 数据集编码（.bin + memmap 就绪）
- [ ] `model.py`：6 层 Transformer（d=512，block_size=128，共享输入输出 embedding，~22.8M 参数）
- [ ] `train.py`：交叉熵 + AdamW + cosine LR，checkpoint 与采样输出
- [ ] `generate.py`：训练后采样生成古诗，test set 困惑度评估

## 参考

- [nanoGPT](https://github.com/karpathy/nanoGPT) —— 本项目数据格式与模型结构的主要参考
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) —— 中文古诗数据集
