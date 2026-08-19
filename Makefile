# 唐诗宋词 GPT 数据准备流水线
# 用法：
#   make           全流程（下载 → 切分 → 编码）
#   make download  只下载原始数据
#   make split     只提取切分数据集
#   make encode    只构建词表并编码
#   make clean     清空数据产物（保留原始数据）
#
# 说明：data/ 不进 git，克隆后先 make 下载数据再跑流水线。

URL = https://github.com/chinese-poetry/chinese-poetry.git
RAW = data/chinese-poetry-raw/全唐诗
SET = data/poetry-set

.PHONY: all download split encode clean

all: encode

# ① 下载原始数据（目录已存在 = 目标已满足，自动跳过）
# 稀疏检出：只拉 全唐诗 目录，不克隆 2GB+ 的整个上游仓库
download: $(RAW)

$(RAW):
	git clone --depth 1 --filter=blob:none --sparse $(URL) data/chinese-poetry-raw
	cd data/chinese-poetry-raw && git sparse-checkout set 全唐诗

# ② 提取并切分训练/测试集（.split 是完成标记，process.py 一次产出多个文件）
split: $(SET)/.split
$(SET)/.split: process/process.py | $(RAW)
	uv run process/process.py
	touch $@

# ③ 词表 + 编码 .bin
encode: $(SET)/.encoded
$(SET)/.encoded: process/tokenizer.py $(SET)/.split
	uv run process/tokenizer.py
	touch $@

clean:
	rm -rf $(SET)/.split $(SET)/.encoded $(SET)/train.bin $(SET)/test.bin $(SET)/*.json $(SET)/distribution.png
