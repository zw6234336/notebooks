# DGX Spark 本地模型匹配与改造建议

这份文档记录的是当前这台本地机器上的实际排查结果，以及基于当前仓库 notebook 模板给出的改造优先级。

目标不是把这个仓库当成一个统一训练框架，而是把它当成一组可直接改造的模板：

1. 先确认本地模型目录是否完整。
2. 再找到同架构或最接近的 notebook。
3. 先跑通基础微调，再上 GRPO 或强化学习。

## 当前机器画像

- GPU：NVIDIA GB10
- Compute Capability：12.1
- CPU 架构：ARM64
- 系统内存：约 119 GiB
- 根分区可用空间：约 2.6 TiB

这类机器更适合先从中大型、且与仓库模板直接匹配的本地模型开始，不建议第一次就用 120B 级模型做强化学习改造。

## 已发现的本地模型

本次在本机上确认到以下目录可作为候选：

### 完整且优先考虑

- /data/models/huggingface/models--google--gemma-4-26B-A4B-it/snapshots/47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301
- /data/models/huggingface/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/1365cf7aa2de42546878b8d2e4a425019a0be514
- /data/models/openai/gpt-oss-120b
- /data/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

### 当前不适合作为直接训练基座

- /data/models/huggingface/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307

这份 Qwen3.5 本地快照当前只看到配置、模板和 tokenizer 相关文件，没有看到实际权重分片，因此暂时不应当作为本地训练基座使用。除非后续确认权重文件已完整下载。

### 空目录

- /home/wayne0404/output_models/saved_models_DeepSeek-R1-Distill-Qwen-7B_nvfp4_hf

这个目录目前为空，不能直接用于微调或继续训练。

## 与当前项目最匹配的 notebook

### 第一优先级：Gemma 4 26B A4B

推荐 notebook：

1. [nb/Gemma4_(26B_A4B)-Text.ipynb](nb/Gemma4_(26B_A4B)-Text.ipynb)
2. [nb/Gemma4_(E2B)_GRPO.ipynb](nb/Gemma4_(E2B)_GRPO.ipynb)

推荐原因：

- 本地已经有完整的 Gemma 4 26B A4B 模型目录。
- 仓库里有同家族、直接可参考的 Gemma 4 notebook。
- 26B 规模比 31B 和 120B 更适合作为这台机器上的第一批改造对象。
- 先从 Text notebook 跑通基础微调，再迁移到 GRPO notebook，是更稳的路径。

建议顺序：

1. 先改 [nb/Gemma4_(26B_A4B)-Text.ipynb](nb/Gemma4_(26B_A4B)-Text.ipynb) 为本地模型路径。
2. 确认基础微调可跑通。
3. 再改 [nb/Gemma4_(E2B)_GRPO.ipynb](nb/Gemma4_(E2B)_GRPO.ipynb) 做本地 RL 版本。

### 第二优先级：Gemma 4 31B

推荐 notebook：

1. [nb/Gemma4_(31B)-Text.ipynb](nb/Gemma4_(31B)-Text.ipynb)

推荐原因：

- 本地存在完整的 Gemma 4 31B NVFP4 模型。
- 仓库里有直接对应的 31B notebook。

但它不应作为第一选择，原因是：

- 31B 比 26B 更重，调参容错更低。
- 如果第一轮是本地改造和验证流程，26B 更适合先打通全链路。

### 第三优先级：gpt-oss 120B

参考 notebook：

1. [nb/gpt-oss-(20B)-Fine-tuning.ipynb](nb/gpt-oss-(20B)-Fine-tuning.ipynb)
2. [nb/gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb](nb/gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb)

说明：

- 当前仓库对 gpt-oss 的支持是完善的。
- 但本地现成模型是 120B，而仓库中最直接的模板大多是 20B 或专门标注 A100 的 120B notebook。
- 因此它更适合作为第二阶段参考，而不是第一阶段落地目标。

### 不建议当前优先改造：Nemotron 120B

原因：

- 本地虽然有完整模型目录。
- 但仓库里没有和 NemotronH 架构直接对齐的现成 notebook。
- 改造成本会高于 Gemma 4 和 gpt-oss。

## 推荐的整体优先级

按当前机器与本地模型现状，建议优先级如下：

1. 本地 Gemma 4 26B A4B + [nb/Gemma4_(26B_A4B)-Text.ipynb](nb/Gemma4_(26B_A4B)-Text.ipynb)
2. 本地 Gemma 4 26B A4B + [nb/Gemma4_(E2B)_GRPO.ipynb](nb/Gemma4_(E2B)_GRPO.ipynb)
3. 本地 Gemma 4 31B NVFP4 + [nb/Gemma4_(31B)-Text.ipynb](nb/Gemma4_(31B)-Text.ipynb)
4. 本地 gpt-oss-120b，参考 [nb/gpt-oss-(20B)-Fine-tuning.ipynb](nb/gpt-oss-(20B)-Fine-tuning.ipynb) 或 [nb/gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb](nb/gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb)
5. 本地 Nemotron 120B

## 本地改造时的实际操作建议

### 1. 先确认模型目录完整

至少要确认模型目录里存在以下内容中的大部分：

- config.json
- tokenizer.json 或 tokenizer_config.json
- model.safetensors.index.json 或实际分片权重文件
- model-xxxxx-of-xxxxx.safetensors

如果只有 config 和 tokenizer，没有权重文件，就不能作为训练基座。

### 2. 优先替换 notebook 中的模型路径

最小改法是把 notebook 里的 model_name 改为本地路径，而不是先大规模改训练逻辑。

例如将远程模型名替换为：

- /data/models/huggingface/models--google--gemma-4-26B-A4B-it/snapshots/47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301

### 3. 第一轮参数保守设置

建议第一轮本地验证时采用更保守的参数：

- 较短的 max_seq_length
- 较小的 batch size
- 较低的 LoRA rank
- 较少的训练步数

目标不是第一次就追求最终效果，而是先验证：

1. 模型能加载。
2. 数据能进入训练。
3. 显存和内存不会立刻爆掉。
4. checkpoint 能成功保存。

### 4. 先做基础微调，再做强化学习

对当前这台机器，推荐路径是：

1. 先用 Gemma 4 26B Text notebook 打通基础微调。
2. 再把同一个本地模型迁移到 Gemma 4 GRPO notebook。
3. 最后再考虑 31B 或 120B 级别模型。

## 当前最推荐的动作

如果只做一个最稳妥的动作，优先做这个：

1. 基于本地 Gemma 4 26B A4B，改造 [nb/Gemma4_(26B_A4B)-Text.ipynb](nb/Gemma4_(26B_A4B)-Text.ipynb)

如果要继续往强化学习推进，再做这个：

2. 基于同一个本地 Gemma 4 26B A4B，改造 [nb/Gemma4_(E2B)_GRPO.ipynb](nb/Gemma4_(E2B)_GRPO.ipynb)