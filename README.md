# 实验课题：基于 ViT 特征与自实现 Transformer Decoder 的图像描述生成

## 1. 实验简介

本实验面向图像描述生成任务，使用现代 Transformer 思路：冻结预训练图像编码器，学生动手实现 Transformer Decoder，并训练 Decoder 根据图像特征生成英文描述。

实验重点不是调用现成大模型，而是理解并实现图像描述生成中的关键机制：

- 词表构建、caption token 化和 padding。
- causal mask，防止解码器看到未来词。
- multi-head self-attention。
- image cross-attention。
- Transformer decoder block。
- greedy decoding。
- BLEU 评价和结果可视化。

本项目提供两种 encoder 路线。基础版使用 ImageNet 预训练 ViT，依赖少、最稳定；提高版使用 CLIP ViT-B/32 image encoder，图像特征来自图文对比学习，更贴近视觉语言任务。

## 2. 数据集

默认复用原实验的 Flickr8k 数据：

- 图片目录：`./flickr8k/Images/`
- 标注文件：`./flickr8k/captions.txt`

数据划分：

- `FAST_DEV_RUN=True`：只使用几十张图片，用于调试。
- 正式实验：使用全部约 8,000 张图像，按图像数量 **80/10/10** 划分为训练集、验证集和测试集。
- 同一张图像的全部 **5 条 caption** 都参与训练，caption 随所属图像归入同一子集，避免数据泄露。
- 训练 caption 总量约 32,000 条。

## 3. 模型结构

```text
image
  -> frozen pretrained image encoder
  -> image feature vector
  -> projection to d_model
  -> Transformer Decoder cross-attention
caption prefix tokens
  -> token embedding + positional embedding
  -> masked self-attention
  -> cross-attention to image feature
  -> FFN
  -> vocabulary logits
```

默认轻量配置：

- 基础版图像编码器：`torchvision.models.vit_b_16`，使用 `ViT_B_16_Weights.DEFAULT` 加载 ImageNet-1K 分类预训练权重并冻结。
- 提高版图像编码器：`openai/clip-vit-base-patch32` 的 CLIP image encoder，使用图文对比学习预训练权重并冻结。notebook 使用新版 `CLIPImageProcessor` 做图像预处理。
- Decoder hidden size：`d_model=256`
- Decoder layers：2
- Attention heads：4
- FFN hidden size：512
- 最大 caption 长度：32
- batch size：32，显存不足可降到 16 或 8

## 4. 文件说明

- `transformer_caption_starter.ipynb`：初始 notebook，包含 TODOs。
- `requirements.txt`：依赖列表。
- `report_template.md`：实验报告模板。
- `assets/transformer_caption_pipeline.svg`：ViT/CLIP + Transformer Decoder 模型流程图。
- `assets/vit_clip_transformer_caption_architecture.svg`：插入 notebook 的主体方案图，展示双 encoder 路线、Transformer Decoder、cross-attention 和生成流程。
- `assets/torchvision_vit_encoder.svg`：展示 torchvision ViT-B/16 的 patch embedding、Transformer encoder 和全局 CLS 特征输出。
- `assets/clip_vit_encoder.svg`：展示 CLIP ViT-B/32 的图文对比预训练背景，以及当前项目使用的 CLS + patch-token visual memory。
- `assets/decoder_classic_transformer.svg`：展示学生手工实现的 Transformer Decoder（Pre-LN）详细结构，包括 Decoder Block、Masked Self-Attention、Cross-Attention 和 FFN。
- `assets/imcaption.png`、`assets/tocken.gif`、`assets/samples.png`：任务说明、tokenization 和数据样例可视化素材。

## 4.1 数据集获取

本仓库不包含 Flickr8k 数据集（约 1 GB）。请自行下载并解压到项目根目录，确保目录结构如下：

```text
./flickr8k/Images/
./flickr8k/captions.txt
```

常用下载来源：

- **Kaggle**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **百度网盘**：[[https://pan.baidu.com/s/1LCuDxrsCS6LoJyxWvzeZDQ?pwd=qd1x ]]

下载后解压即可：

```bash
unzip flickr8k.zip -d ./
```

## 5. 推荐运行流程

1. 打开 `transformer_caption_starter.ipynb`。
2. 保持 `FAST_DEV_RUN=True`，先跑通数据处理和一个 batch。
3. 基础实验保持 `ENCODER_BACKBONE='torchvision_vit'`。
4. 完成 TODO：词表、mask、attention、decoder block、训练、解码。
5. 调试通过后改为 `FAST_DEV_RUN=False`。
6. 提高实验可改为 `ENCODER_BACKBONE='clip_vit_b32'`，对比 ImageNet ViT 与 CLIP 特征。
7. 训练 3-5 个 epoch，记录 loss、BLEU 和生成样例。
8. 根据 `report_template.md` 撰写报告。

## 6. 提交要求

学生需要提交：

- 完成后的 notebook。
- 实验报告。
- 至少 8 张测试图像的生成描述可视化。
- 对模型结构、mask、attention 和 cross-attention 的解释。
- 对 ImageNet ViT 与 CLIP image encoder 差异的说明。
- 对训练结果、失败案例和改进方向的讨论。

## 7. 资源控制建议

- GPU 上训练前先使用 `FAST_DEV_RUN=True`，快速验证代码正确性。
- 图像编码器默认冻结，只训练轻量 Decoder。
- 显存不足时优先降低 `BATCH_SIZE`、`TRAIN_SIZE`、`MAX_LEN`。
- 如果训练仍然慢，可以先缓存 image feature。
- 如果出现训练集 loss 持续下降、验证集 loss 不再下降甚至上升，说明 decoder 已开始过拟合。应使用验证集 loss 最低的 epoch 作为最终模型，并启用 early stopping。
- CLIP 提高版使用 `CLIPVisionModel.last_hidden_state` 中的 CLS token 和 patch tokens 作为 cross-attention 的 visual memory，而不是只使用一个 `pooler_output` 全局向量。这样 decoder 可以对多个图像 token 做 cross-attention，通常比单向量图像特征更适合 caption 生成。
- 本项目默认使用新版 torchvision 权重枚举接口：`ViT_B_16_Weights.DEFAULT` 和 `vit_b_16(weights=...)`。
- 如果使用 CLIP 提高版，当前 notebook 使用 `CLIPImageProcessor` 进行图像预处理，CLIP 图像编码器使用 `CLIPVisionModel`。
- 如果在线下载 torchvision ViT 太慢，可以手工下载 ViT-B/16 的 `.pth` 权重，并复制到 notebook 当前工作目录 `./models/vit_b_16-c867db91.pth`，然后设置 `USE_LOCAL_TORCHVISION_VIT=True`。
- 如果在线下载 CLIP 太慢，可以手工下载 `./models/clip-vit-base-patch32`或者‘`./models/hub`’，并把完整模型目录复制到 notebook 当前工作目录 `./models/hub`，然后设置 `USE_LOCAL_CLIP=True`。

## 8. GPU 平台推荐

如果本地没有合适的 NVIDIA GPU，可以使用阿里云免费 GPU 平台完成本实验：

- **阿里云免费试用**：https://free.aliyun.com/
- 新用户通常可以领取免费 GPU 实例（如 P100、V100 等），用于 Notebook 训练与调试。
- 建议在免费实例中安装本项目依赖（`requirements.txt`），上传数据集和 Notebook 后即可运行。

## 9. 手工下载 Encoder 权重

### 8.1 手工下载 torchvision ViT 权重

基础版默认会执行：

```python
weights = ViT_B_16_Weights.DEFAULT
vit_b_16(weights=weights)
```

离线下载地址：

```text
https://download.pytorch.org/models/vit_b_16-c867db91.pth
```

如果平台在线下载很慢，可以提前下载这个 `.pth` 文件，并复制到 notebook 当前工作目录：

```text
.models/vit_b_16-c867db91.pth
```

然后在 notebook 配置区改为：

```python
USE_LOCAL_TORCHVISION_VIT = True
LOCAL_TORCHVISION_VIT_WEIGHTS = Path("./models/vit_b_16-c867db91.pth")
```

代码会自动改用：

```python
self.vit = vit_b_16(weights=None)
checkpoint = torch.load(LOCAL_TORCHVISION_VIT_WEIGHTS, map_location="cpu")
self.vit.load_state_dict(state_dict)
```

这样构建 torchvision ViT encoder 时不会再触发在线下载。

如果不手工下载，torchvision 在线下载后的默认缓存位置通常是：

```text
~/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth
```

如果设置了 `TORCH_HOME`，则通常在：

```text
$TORCH_HOME/hub/checkpoints/vit_b_16-c867db91.pth
```

### 8.2 手工下载 CLIP 权重

CLIP 提高版默认会执行：

```python
CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
```

Hugging Face 模型主页：

```text
https://huggingface.co/openai/clip-vit-base-patch32
```

建议下载完整模型目录，并复制到 notebook 当前工作目录：

```text
./models/hub/models--openai--clip-vit-base-patch32/
```

至少需要包含这些文件：

```text
config.json
preprocessor_config.json
pytorch_model.bin
```

关键文件也可以分别从下面地址下载：

```text
https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json
https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json
https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin
```

如果平台支持 safetensors，也可以使用：

```text
https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/model.safetensors
```

如果你下载的是普通 Hugging Face 模型目录，目录结构可以是：

```text
./clip-vit-base-patch32/
  config.json
  preprocessor_config.json
  pytorch_model.bin
```

如果你复制的是 Hugging Face 缓存目录，目录结构通常是(**推荐方式，当前代码直接支持**)：

```text
./models/hub/
  models--openai--clip-vit-base-patch32/
    blobs/
    refs/
    snapshots/
      <commit-hash>/
        config.json -> ../../blobs/...
        preprocessor_config.json -> ../../blobs/...
        pytorch_model.bin -> ../../blobs/...
```

这种情况下，`from_pretrained()` 不能直接指向 `models/hub/`；它需要具体 snapshot 目录。本 notebook 已提供 `resolve_clip_local_dir()`，可以从 `./models/hub` 自动找到完整 snapshot。

如果使用你当前的缓存目录结构，在 notebook 配置区改为：

```python
USE_LOCAL_CLIP = True
LOCAL_CLIP_MODEL_DIR = Path("./models/hub")
```

也可以直接指向完整 snapshot 目录。例如当前目录中可用的路径是：

```python
USE_LOCAL_CLIP = True
LOCAL_CLIP_MODEL_DIR = Path(
    "./models/hub/models--openai--clip-vit-base-patch32/"
    "snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)
```

如果你整理成普通模型目录，也可以写成：

```python
USE_LOCAL_CLIP = True
LOCAL_CLIP_MODEL_DIR = Path("./clip-vit-base-patch32")
```

代码会自动解析路径并调用：

```python
clip_source = resolve_clip_local_dir(LOCAL_CLIP_MODEL_DIR)
CLIPImageProcessor.from_pretrained(clip_source, local_files_only=True)
CLIPVisionModel.from_pretrained(clip_source, local_files_only=True)
```

这样运行时不会再访问 Hugging Face 网络。

如果不手工下载，Hugging Face 在线下载后的默认缓存位置通常是：

```text
~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/
```

真实文件一般位于：

```text
~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/<commit-hash>/
```

如果设置了环境变量，缓存目录可能改到：

```text
$HF_HOME/hub/
$HF_HUB_CACHE/
$TRANSFORMERS_CACHE/
```

常用检查命令：

```bash
ls ~/.cache/torch/hub/checkpoints/
ls ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/
```

***上述过程如果不能解决，请直接从下面百度网盘下载models.zip, 并解压到项目根目录下即可。***

CLIP-ViT模型文件：[[https://pan.baidu.com/s/1UzoQHg37Y5t9grY22ERIiQ?pwd=pjsm ]]