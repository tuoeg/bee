## 小小蜜蜂
## 总述
- 原始模型的名称：LayoutLMv3(https://github.com/microsoft/unilm/tree/master/layoutlmv3)


## 原始模型
### 模型简介
1.LayoutLMv3，用于预训练具有统一文本图像掩码的文档AI的多模态Transformer。此外，LayoutLMv3还预训练一个词块对齐目标，通过预测一个文本词的对应图像块是否被掩码来学习跨模态对齐。LayoutLMv3不依赖预训练的CNN或Faster R-CNN骨干来提取视觉特征，大大节省了参数并消除了局部标注。简单的统一架构和训练目标使LayoutLMv3成为通用的预训练模型，适用于以文本为中心和以图像为中心的文档AI任务。实验结果表明，LayoutLMv3不仅在以文本为中心的任务中，包括表单理解、小票理解和文档视觉问答等，而且在以图像为中心的任务中，如文档图像分类和文档布局分析等，均取得了最先进的性能。
