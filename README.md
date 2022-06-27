## Torch2TRT for LayoutLMv3
- 原始模型的名称：LayoutLMv3(https://github.com/microsoft/unilm/tree/master/layoutlmv3)


## 原始模型
### 模型简介
1. LayoutLMv3，用于预训练具有统一文本图像掩码的文档AI的多模态Transformer。此外，LayoutLMv3还预训练一个词块对齐目标，通过预测一个文本词的对应图像块是否被掩码来学习跨模态对齐。LayoutLMv3不依赖预训练的CNN或Faster R-CNN骨干来提取视觉特征，大大节省了参数并消除了局部标注。简单的统一架构和训练目标使LayoutLMv3成为通用的预训练模型，适用于以文本为中心和以图像为中心的文档AI任务。
2. 模型框架
![image](https://user-images.githubusercontent.com/49616374/170653429-e9557526-3e14-4c17-b00c-20c7a709fe7b.png)

## 系统环境
TensorRT Version: 8.4.1.4<br>
NVIDIA GPU: NVIDIA A10<br>
NVIDIA Driver Version: 510.73.08<br>
CUDA Version：11.6<br>
CUDNN Version: 8.4<br>
Operating System: Ubuntu 20.04.4 LTS<br>
Python Version (if applicable): 3.8.10<br>
PyTorch Version (if applicable): 1.11.0<br>

## 项目结构


## 模型转换以及优化
### 1.torch to onnx

```
$ python3 torch2onnx.py -h
```

遇到的问题/n
（1）opt_version=9不支持，使用opset_version=11

![image](https://user-images.githubusercontent.com/49616374/174259578-b0606449-3a40-4171-aa32-d2dab8549a93.png)

（2）optset_version 11不支持amax操作，修改torch网络中amax函数为max函数

![image](https://user-images.githubusercontent.com/49616374/174259606-c2d4ea64-4125-42cf-82b8-657e660c54ed.png)
### 2.LayerNorm算子优化
## 遇到的问题
### 1.opt_version版本
（1）opt_version=9不支持，使用opset_version=11

![image](https://user-images.githubusercontent.com/49616374/174259578-b0606449-3a40-4171-aa32-d2dab8549a93.png)

（2）optset_version 11不支持amax操作，修改torch网络中amax函数为max函数

![image](https://user-images.githubusercontent.com/49616374/174259606-c2d4ea64-4125-42cf-82b8-657e660c54ed.png)

### 2.算子转换
（1）onehot算子不支持，根据onehot算子原理将onehot+cast+matmul算子合并成gather算子

![image](https://user-images.githubusercontent.com/49616374/174260371-2d1e6093-3a0f-4808-a76d-9380f6654b7f.png)

（2）添加cast进行数据类型转换

![image](https://user-images.githubusercontent.com/49616374/174260502-3a511afc-2b91-49f4-adc2-92b607f2ec43.png)
### 3.精度误差优化
（1）原始数据集存在int64类型导致精度损失；编写小型torch网络并生成trt网络，输入int64类型数据进行测试，发现onnx推理与trt推理存在误差较大，onnx推理与torch推理几乎无误差；提出issue:https://github.com/NVIDIA/TensorRT/issues/2037， 获悉tensorrt默认将int64类型数据转成int32类型数据<br>

![image](https://user-images.githubusercontent.com/49616374/174260801-f0c100b5-84db-4bc2-916a-dfa2ca21e481.png)<br>
（2）cast算子在计算float32数据时存在误差，通过round()放缩，规避误差<br>

![image](https://user-images.githubusercontent.com/49616374/174262579-f7b5157d-ab4c-4d00-af7a-8d568ee69582.png)

### Hackathon 2022 BUG
issue地址：https://github.com/NVIDIA/TensorRT/issues/2063
