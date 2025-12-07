# cs336-jupyter

面向 TinyStories 数据集的 Jupyter 笔记本：从 BPE 分词器训练，到 Transformer 语言模型训练与文本生成。Markdown 版说明位于 `docs/`（内容多为中文笔记）。

## 目录结构
- `BPE_DataPrepare.ipynb`：下载 TinyStories，将每行追加 `<|endoftext|>`，当前示例路径为绝对路径，使用前改成 `./datasets/TinyStories` 下的相对路径。
- `BPE_Train.ipynb`：训练 BPE，生成 `bpeModel/vocab.json` 与 `bpeModel/merges.txt`。
- `BPE_Tokenizer.ipynb`：加载已训分词器，流式编码文本为 token 数组，写入 `datasets/` 下的 `.npy`/memmap。
- `TrainFunctions.ipynb`：公共组件（交叉熵、AdamW、余弦退火、梯度裁剪、数据加载、checkpoint、`TrainingConfig`）。
- `TrainLoop.ipynb`：搭建模型与优化器，训练并把 checkpoint/log 写到 `TrainLoopFiles/`，曲线保存在 `TrainLoop_files/`。
- `TransformerLM.ipynb`：模型主体（Embedding、RoPE 注意力、SwiGLU 前馈、RMSNorm）。
- `Generation.ipynb`：加载 tokenizer 与 checkpoint（默认 `TrainLoopFiles/checkpoints_localhost/checkpoint_final.pt`），从提示语生成文本。
- `docs/`：以上各笔记本的 Markdown 导出，便于快速浏览。
- `bpeModel/`、`datasets/`、`TrainLoopFiles/`、`TrainLoop_files/`、`.ipynb_checkpoints/`：生成产物或缓存，已在 `.gitignore` 中忽略。

## 环境与依赖
- 建议 Python 3.10+。
- 安装依赖（根据你的 CUDA 版本选择合适的 torch）：
  ```bash
  pip install torch datasets regex numpy tqdm matplotlib import-ipynb
  ```
- 从仓库根目录启动 Jupyter，保证 `from TrainFunctions import ...` 等相对导入可用：
  ```bash
  jupyter lab   # 或 jupyter notebook
  ```

## 推荐流程
1. `BPE_DataPrepare.ipynb`：下载 TinyStories、写入 train/valid 文本并追加 `<|endoftext|>`，把绝对路径改为本地相对路径。
2. `BPE_Train.ipynb`：训练 BPE，得到 `bpeModel/vocab.json` 与 `merges.txt`。
3. `BPE_Tokenizer.ipynb`：用 tokenizer 编码数据集，生成 tokens 文件到 `datasets/`。
4. `TrainFunctions.ipynb`：查看/调整优化器、学习率调度、采样与 checkpoint 工具。
5. `TrainLoop.ipynb`：训练 TransformerLM，checkpoint 与日志输出到 `TrainLoopFiles/`，曲线图在 `TrainLoop_files/`。
6. `Generation.ipynb`：加载 tokenizer 和某个 checkpoint（必要时更改路径），输入 prompt 生成文本。

## 备注
- 建议把所有生成文件放在已存在的 `bpeModel/`、`datasets/`、`TrainLoopFiles/` 目录内，避免污染版本库。
- 若在新机器上运行，优先检查路径与 GPU 可用性；若无 GPU，默认会回退到 CPU。
