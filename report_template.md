# ViT + Transformer Decoder 图像描述生成实验报告模板

## 1. 实验目的

说明本实验要解决的图像描述生成问题，以及 Transformer Decoder 相比 CNN + LSTM 的主要变化。

## 2. 数据集与预处理

- 数据集名称：
- 使用图片数量：
- train / val / test 划分：
- caption 清洗方式：
- 词表大小：
- 最大句长：
- 样例可视化：

## 3. 模型结构

请说明以下模块：

- 冻结预训练图像编码器的作用。
- 基础版 ImageNet ViT encoder 与提高版 CLIP image encoder 的预训练目标差异。
- token embedding 和 positional embedding。
- causal mask 的目的。
- multi-head self-attention 的计算流程。
- cross-attention 如何使用图像特征。
- Transformer Decoder Block 的组成。
- greedy decoding 生成 caption 的过程。

## 4. 实验设置

- 运行平台：
- GPU 型号：
- batch size：
- epoch：
- learning rate：
- d_model：
- num layers：
- num heads：
- FFN hidden size：
- encoder backbone：`torchvision_vit` 或 `clip_vit_b32`

## 5. 实验结果

请展示至少 3 个测试样例：

| Image | Ground Truth | Generated Caption |
| ----- | ------------ | ----------------- |
| 示例 1  |              |                   |
| 示例 2  |              |                   |
| 示例 3  |              |                   |
| 示例 4  |              |                   |
| 示例 5  |              |                   |

请填写评价指标：

| Model                               | BLEU-1 | BLEU-2 | BLEU-4 | Avg Length |
| ----------------------------------- | -----: | -----: | -----: | ---------: |
| ImageNet ViT + Transformer Decoder  |        |        |        |            |
| CLIP ViT-B/32 + Transformer Decoder |        |        |        |            |

## 6. 分析与讨论

- 模型是否学会了常见物体和动作？
- 生成结果中有哪些重复、遗漏或错误？
- 小规模训练数据带来了什么限制？
- CLIP 图像特征是否比 ImageNet 分类特征更适合 caption？为什么？

## 7. 总结

总结本实验的收获、问题和可改进方向。
