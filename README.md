# 使用 GPTQ 量化 OLMoE 模型 (Quantizing OLMoE Models with GPTQ)

这个仓库包含了对 `allenai/OLMoE` 系列模型进行 GPTQ（Generative Pre-trained Transformer Quantization）量化的实现代码。脚本支持对模型进行不同比特数（如 2, 3, 4, 16位）的权重量化，并评估其在标准数据集（WikiText-2, PTB, C4）上的困惑度（Perplexity）。

## 功能

*   支持对 OLMoE 系列模型进行事后量化（Post-Training Quantization）。
*   实现了 GPTQ 算法，支持分组量化（group size）、激活排序（activation order）和对称/非对称量化。
*   支持在多个标准数据集上进行模型校准和评估。
*   可以保存量化后的模型权重以供后续使用。

## 环境要求

在运行脚本之前，请确保你已经安装了必要的 Python 库。

```bash
pip install torch transformers datasets sentencepiece accelerate
```

你还需要确保项目中的辅助文件存在于同一个目录下：
*   `gptq.py`
*   `modelutils.py`
*   `quant.py`
*   `datautils.py`

## 使用方法

脚本通过命令行参数来控制模型的加载、量化和评估流程。

### 参数说明

*   `model`: 必选，要加载的 OLMoE 模型名称或路径 (例如: `allenai/OLMoE-1B` 或 `allenai/OLMoE-7B`)。
*   `dataset`: 必选，用于模型校准的数据集 (`wikitext2`, `ptb`, `c4`)。
*   `--wbits`: 量化的比特数 (可选: 2, 3, 4, 8, 16)。设置为 16 将跳过量化，直接评估原始模型。
*   `--groupsize`: 分组量化的大小。默认为 -1，表示按行（per-row）量化。
*   `--nsamples`: 用于校准的样本数量，默认为 128。
*   `--act-order`: 是否在量化时应用激活排序，这有助于提升量化精度。
*   `--sym`: 是否执行对称量化。
*   `--save`: 保存量化后模型权重的文件路径 (例如: `olmoe-1b-4bit.pt`)。
*   `--true-sequential`: 是否以真正的顺序模式执行量化，有助于节省显存。
*   `--nearest`: 使用最邻近取整（RTN）方法进行量化，作为基线对比。

### 示例

#### 1. 对 `allenai/olmoe-1b-7b-0125-instruct` 模型进行 4-bit 量化

以下命令将会：
*   加载 `allenai/olmoe-1b-7b-0125-instruct` 模型。
*   使用 `wikitext2` 数据集的前 128 个样本进行校准。
*   进行 4-bit、group-size 为 128 的量化，并启用激活排序。
*   在 `wikitext2`, `ptb`, `c4` 数据集上评估量化后模型的困惑度。
*   最后，将量化后的模型权重保存到 `olmoe-1b-4bit.pt` 文件中。

```bash
python olmoe.py \
    allenai/OLMoE-1B \
    wikitext2 \
    --wbits 4 \
    --groupsize 128 \
    --act-order \
    --save olmoe-1b-4bit.pt
```

#### 2. 评估原始（FP16）模型的性能

如果你想跳过量化，直接评估原始 FP16 模型的性能，可以将 `wbits` 设置为 16。

```bash
python olmoe.py \
    allenai/OLMoE-1B \
    wikitext2 \
    --wbits 16
```

#### 3. 使用 RTN 方法进行 3-bit 量化

使用简单的取整（Round-To-Nearest）方法进行量化，可以添加 `--nearest` 标志。

```bash
python olmoe.py \
    allenai/OLMoE-1B \
    wikitext2 \
    --wbits 3 \
    --nearest
```

## 注意事项

*   量化过程需要较高的 GPU 显存。对于大型模型（如 7B），请确保你有足够的显存。可以尝试使用 `--true-sequential` 标志来降低显存峰值。
*   脚本会自动从 Hugging Face Hub 下载模型和数据集。
