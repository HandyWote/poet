# poet — Training a Classical Chinese Poetry GPT from Scratch

A complete "character-level GPT for classical Chinese poetry" pipeline with PyTorch, from data cleaning to model training.
Current progress: **data preparation complete** (extraction → vocabulary → encoding). Next up: the Transformer model and training script.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/), Python 3.12+, and git.

```bash
uv sync                 # install dependencies
make                    # download data → split dataset → build vocabulary → encode .bin (skips steps whose outputs already exist)
```

`make` runs the entire data preparation pipeline in one command; you can also run steps individually:
`make download` (download raw data only), `make split` (split only), `make encode` (build vocab and encode only), `make clean` (remove artifacts, keep raw data).

## Directory Layout

```
poet/
├── Makefile                 # Data preparation pipeline entry point
├── process/
│   ├── process.py           # Raw data → train/test JSON (7:3 random split, seed=42)
│   └── tokenizer.py         # Character frequency stats → vocabulary → dataset encoding (.bin)
├── docs/
│   └── decision-vocab-threshold.md  # Decision record: rationale for vocabulary frequency threshold T=10
├── data/poetry-set/         # Data artifacts (git-ignored, generated locally by make)
│   ├── vocab.json / re_vocab.json   # 7,399-token vocabulary (incl. <unk>/<bos>/<eos>)
│   ├── train.bin / test.bin         # uint16-encoded data, read via memmap during training
│   └── distribution.png             # Character frequency long-tail distribution and coverage plot
└── pyproject.toml
```

## Data

Raw corpus from [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
(177,973 Song poems + Tang poems). `make download` uses a sparse checkout to fetch only the `全唐诗/` directory (~500MB) instead of cloning the 2GB+ repo.
`data/` is entirely git-ignored and not distributed with the repo; after cloning, run `make` to rebuild all artifacts.

Pipeline outputs: `train_set.json` / `test_set.json` (177,973 / 76,275 poems),
a 7,399-token vocabulary (threshold T=10, rationale in the decision record),
`train.bin` / `test.bin` (13,564,418 / 5,806,696 uint16 token ids).

## Encoding Format Conventions

Each poem = `<bos>` + title + `\n` + body + `<eos>`; out-of-vocabulary characters map to `<unk>` (0).
During training, use `np.memmap(path, dtype=np.uint16, mode='r')` and slice random offsets as `block_size=128` windows for samples.

## Roadmap

- [x] Data extraction and split
- [x] Vocabulary construction (incl. frequency threshold decision record)
- [x] Dataset encoding (.bin + memmap ready)
- [ ] `model.py`: 6-layer Transformer (d=512, block_size=128, shared input/output embeddings, ~22.8M params)
- [ ] `train.py`: cross-entropy + AdamW + cosine LR, checkpoints and sample generation
- [ ] `generate.py`: sample poems after training, evaluate perplexity on the test set

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — main reference for this project's data format and model architecture
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — classical Chinese poetry dataset
