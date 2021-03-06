## Torch2TRT for LayoutLMv3
- 原始模型的名称：LayoutLMv3(https://github.com/microsoft/unilm/tree/master/layoutlmv3)


## 原始模型
### 模型简介
1. LayoutLMv3，用于预训练具有统一文本图像掩码的文档AI的多模态Transformer。此外，LayoutLMv3还预训练一个词块对齐目标，通过预测一个文本词的对应图像块是否被掩码来学习跨模态对齐。LayoutLMv3不依赖预训练的CNN或Faster R-CNN骨干来提取视觉特征，大大节省了参数并消除了局部标注。简单的统一架构和训练目标使LayoutLMv3成为通用的预训练模型，适用于以文本为中心和以图像为中心的文档AI任务。
2. 模型框架
![image](https://user-images.githubusercontent.com/49616374/170653429-e9557526-3e14-4c17-b00c-20c7a709fe7b.png)

## 系统环境
**TensorRT Version:** 8.4.1.4<br>
**NVIDIA GPU:** NVIDIA A10<br>
**NVIDIA Driver Version:** 510.73.08<br>
**CUDA Version**：11.6<br>
**CUDNN Version:** 8.4<br>
**Operating System:** Ubuntu 20.04.4 LTS<br>
**Python Version (if applicable):** 3.8.10<br>
**PyTorch Version (if applicable):** 1.11.0<br>

## 项目结构

## 模型转换前准备
### 1.配置文件
根据官方说明进行配置
```
conda create --name layoutlmv3 python=3.7
conda activate layoutlmv3
git clone https://github.com/microsoft/unilm.git
cd unilm/layoutlmv3
pip install -r requirements.txt
# install pytorch, torchvision refer to https://pytorch.org/get-started/locally/
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# install detectron2 refer to https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -e .
```
### 2.数据准备
将预处理之后的TrainData和EvalData从train_dataset和eval_dataset中提取出来。提取的TrainData用来int8量化，EvalData用来评测。

### 3.源码改动
由于onnx无法像torch一样指定输入，因此需要将LayoutLMv3ForTokenClassification类中的forward函数入参进行顺序修改，将四个输入input_ids、bbox、attention_mask和images的顺序放在最上面。
```
def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        images=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
```
## 模型转换以及优化
### 1.torch to onnx

### 此过程遇到的问题      

（1）opt_version=9不支持。
<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/49616374/174259578-b0606449-3a40-4171-aa32-d2dab8549a93.png" >
</div>

我们将opset_version设为11跳过了这个问题。

（2）optset_version 11不支持amax操作。
<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/49616374/174259606-c2d4ea64-4125-42cf-82b8-657e660c54ed.png" >
</div>

我们修改torch网络中amax函数为max函数，跳过了这个问题。

<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/53067559/175852416-f750cd2c-d357-485a-919c-86d640eb56f0.png" >
</div>

（3）一些算子不支持多种类型进行操作，或者不支持int类型进行操作。

<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/49616374/174260502-3a511afc-2b91-49f4-adc2-92b607f2ec43.png" >
</div>

我们加入cast节点进行类型转换和修改节点内部数据类型。下图为其中一个修改点。

<table width="100%" align="center">
    <tbody>
        <tr align="center">
            <th>BEFORE</th>
            <th>AFTER</th>
        </tr>
        <tr style="text-align: center;">
            <td>
                <img height="300px" alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/53067559/175861628-a1186389-0ac6-416f-98a2-44efa4862df9.png" >
            </td>
            <td>
                <img height="300px" alt="企业微信截图_16563018741884" src="https://user-images.githubusercontent.com/53067559/175861708-19ad1dcf-d09f-43a3-bee6-0d8c7ea8d1fc.png" >
            </td>
        </tr>
    </tbody>
</table>

### simplify方法  

```
# 静态
$ python3 -m onnxsim layoutv3.onnx layoutv3_sim.onnx
```
我们使用Nsight发现TRT自处理后的结构比和使用onnxsim后的结构好，因此未使用onnxsim。

onnx推理测试
```
$ python3 -m onnxsim layoutv3.onnx layoutv3_sim.onnx
```

### 2.ONNX2TensorRT（FP32）
使用trtexec
```
# 动态
$ trtexec --onnx=layout.onnx --minShapes=input_ids:1x512,bbox:1x512x4,images:1x3x224x224 --optShapes=input_ids:6x512,bbox:6x512x4,images:6x3x224x224 --maxShapes=input_ids:6x512,bbox:6x512x4,images:6x3x224x224 --workspace=300000 --saveEngine=layout.plan --verbose --noTF32 --plugins=./LayerNorm.so


# 静态
$ trtexec --onnx=layout.onnx --workspace=300000 --saveEngine=layout.plan --verbose --plugins=./LayerNorm.so --noTF32
```

### 此过程遇到的问题  
（1）onehot算子不支持。

<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/49616374/174260371-2d1e6093-3a0f-4808-a76d-9380f6654b7f.png" >
</div>
  
我们根据源码和算子里面的数据判断onehot算子加上后面的matmul算子就是nn.embedding的结构。因此我们将onehot+cast+matmul算子合并成nn.embedding转成的gather算子。如下图。

<table width="100%" align="center">
    <tr>
        <th align="center">BEFORE</th>
        <th align="center">AFTER</th>
    </tr>
    <tr>
        <td>
            <img height="300px" alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/53067559/175856736-2cbc4e4c-1033-4283-83c8-6e247b22b38b.png">
        </td>
        <td>
            <img height="100px" alt="企业微信截图_16563018741884" src="https://user-images.githubusercontent.com/53067559/175856737-8c0f6787-4472-4e01-b169-be63379ee9f5.png">
        </td>
    </tr>
</table>


（2）出现精度误差。  

<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/49616374/174260801-f0c100b5-84db-4bc2-916a-dfa2ca21e481.png" >
</div>

转换后的TRT模型精度出现较大误差，但是在之前生成的ONNX模型上是符合e-5的误差的。我们先使用了polygrahy工具尝试查找精度出问题的layer。
```
polygraphy run layout.onnx --trt --onnxrt --onnx-outputs mark all --trt-outputs mark all --rtol 1e-5 --atol 1e-5
```
出现了如下图的问题，老师解释貌似是每层加输出会破坏Myelin的优化融合。  

<div align="center">
    <img alt="企业微信截图_1656301766905" src="https://user-images.githubusercontent.com/53067559/175874907-d65fb57c-b868-4c3f-b282-e7d0c7286027.png" >
</div>

因此我们使用二分法排查精度出问题的layer。经过一段时间的努力，使用了onnx_graphsurgeon定位并裁剪出了有问题的小型网络并生成trt网络。  

<div align="center">
    <img height="450px" src="https://user-images.githubusercontent.com/53067559/175875678-bffd11e4-477f-4265-a487-ca0a1dbb256c.png" />
</div>

如上图，我们发现经过cast算子之前的输出roi精度符合要求，但是经过cast算子之后的输出out精度发生较大的误差。我们将cast之前的输出roi打印出来，发现是float转int发生的误差。  

<div align="center">
    <img height="450px" src="https://user-images.githubusercontent.com/49616374/174262579-f7b5157d-ab4c-4d00-af7a-8d568ee69582.png" />
</div>

由于，round算子是不能指定位数的，所以最直接的方法就是需要实现一个可指定位数的round插件，但是这样需要花费我们一些时间。最后老师给了一个建议，提出将cast之前的输出乘1e5，再使用round，最后除以1e5还原。我们添加了Mul,Round,Div三个算子，TRT的结果与Torch结果的误差在e-5。ONNX部分如下图所示。  

<div align="center">
    <img height="450px" src="https://user-images.githubusercontent.com/53067559/175878044-35a5b911-3bec-4919-a7bb-890601d77e4b.png" />
</div>
    
### 此过程的优化  

我们使用Nsight工具，发现网络中有些LayerNorm层并没有被网络融合。如下图。  

<div align="center">
    <img height="450px" src="https://user-images.githubusercontent.com/53067559/176043916-f04650cd-58eb-45ae-b9aa-d4fadbd05795.png" />
</div>
我们使用Nsight将没有融合的layerNorm层选择出来，并使用我们实现的LayerNorm插件进行融合加速。

<table width="100%" align="center">
    <tr>
        <th align="center">BEFORE</th>
        <th align="center">AFTER</th>
    </tr>
    <tr>
        <td>
            <img height="450px" alt="企业微信截图_16563337066001" src="https://user-images.githubusercontent.com/53067559/175980793-b5550b97-08aa-463a-a6e1-4bbccc855f76.png">
        </td>
        <td>
            <img height="90px" alt="企业微信截图_16563337066001" src="https://user-images.githubusercontent.com/53067559/175980898-11dcbacd-91c7-4045-8eb9-ddfed7029169.png">
        </td>
    </tr>
</table>

### 2.ONNX2TensorRT（FP16）
```
# 动态
$ python3 onnx2TRT.py


# 静态
$ python3 onnx2TRT.py
```

使用trtexec（误差极大）
```
# 动态
$ trtexec --onnx=layout.onnx --minShapes=input_ids:1x512,bbox:1x512x4,images:1x3x224x224 --optShapes=input_ids:6x512,bbox:6x512x4,images:6x3x224x224 --maxShapes=input_ids:6x512,bbox:6x512x4,images:6x3x224x224 --workspace=300000 --saveEngine=layout.plan --verbose --fp16 --plugins=./LayerNorm.so


# 静态
$ trtexec --onnx=layout.onnx --workspace=300000 --saveEngine=layout.plan --verbose --plugins=./LayerNorm.so --fp16
```
### 此过程遇到的问题  
（1）我们直接使用trtexec去转FP16时，发现结果的精度误差极大。（TRT的精度BUG）  

<div align="center">
    <img src="https://user-images.githubusercontent.com/53067559/175913159-17aeaa8a-3187-4067-a859-2e1f544ca792.png" />
</div>


我们尝试使用polygrahy去找精度出问题的layer，但依然发生了之前的问题，破坏了Myelin的优化融合，失败。没有办法，只能再次使用二分法去查找。我们将layoutlmv3分为两部分，第一部分是embedding模块，第二部分是transformer模块。embedding模块的输入为input_ids,bbox,images,输出为rel_pos，rel_2d_pos和hidden_states。transformer模块的输入为rel_pos，rel_2d_pos，hidden_states和attention_mask,输出为预测的分类结果。
  
  经过检测，在FP16精度下，发现embedding模块的rel_pos，rel_2d_pos输出部分值为0，hidden_states输出基本符合fp16的精度误差。然后再次细分，发现只输出rel_pos，rel_2d_pos时，结果精度没问题，但是加入hidden_states分支，影响了rel_pos，rel_2d_pos输出精度。再对hidden_states分支进行层层拆分，最后定位成以下问题：

<div align="center">
    <img src="https://user-images.githubusercontent.com/53067559/175931815-2c400d9a-a165-4c1a-b354-1af2376c88fd.png" />
</div>

 阻断融合，我们尝试了回退两个分支的连接点ADD算子的精度为fp32，修改不成功（TRT的设置BUG），失败；尝试了添加输出层，还是失败，无法阻断融合；最后只能写了两个Matmul算子让两个分支的输出乘以对角为1的矩阵，这次终于将ADD拆分出来，但是两个分支依旧融合成一个外部节点，还是失败了。
 
 我们想到FP32下的engine是OK的，于是我们查看了FP32下的节点融合策略和FP16下的节点融合策略。发现融合策略是一致的，那么真相只有一个！
 ```
 Layer   2:{ForeignNode[layoutlmv3.embeddings.word_embeddings.weight...Add_65]}
    Input [ 0]:[(6, 512, 4), (6, 512, 4), (6, 512, 4)],DataType.INT32,(2048, 4, 1),TensorFormat.LINEAR
    Input [ 1]:[(512,), (512,), (512,)],DataType.INT32,(1,),TensorFormat.LINEAR
    Input [ 2]:[(197,), (197,), (197,)],DataType.INT32,(1,),TensorFormat.LINEAR
    Output[ 0]:[(6, 512), (6, 512), (6, 512)],DataType.FLOAT,(512, 1),TensorFormat.LINEAR
    Output[ 1]:[(6, 512), (6, 512), (6, 512)],DataType.FLOAT,(512, 1),TensorFormat.LINEAR
    Output[ 2]:[(6, 709, 709), (6, 709, 709), (6, 709, 709)],DataType.INT32,(502681, 709, 1),TensorFormat.LINEAR
    Output[ 3]:[(6, 709, 709), (6, 709, 709), (6, 709, 709)],DataType.INT32,(502681, 709, 1),TensorFormat.LINEAR
    Output[ 4]:[(6, 709, 709), (6, 709, 709), (6, 709, 709)],DataType.INT32,(502681, 709, 1),TensorFormat.LINEAR
    algorithm:[implementation:2147483683,tactic:0,timing:0.000000ms,workspace:0MB]
 ```
 没错，恭喜你猜对了！确实是kernel选择的问题，我们将这个节点的tactic打印出来，发现有**三**个。

<div align="center">
    <img src="https://user-images.githubusercontent.com/53067559/175952213-37da4d70-58e7-4cf9-abe7-9bd344421cde.png" />
</div>

 我们实验发现有一个kernel结果是错误的，还有一个kernel的误差精度是在e-3左右。应该是最后的transpose为fp16的精度导致的误差。同时，我们察觉到这可能是水平结构的融合，同时观察到有很多gather算子，进一步确认应该是gather算子的水平融合。于是我们编写了一个gather算子的plugin，替换掉rel_pos，rel_2d_pos分支的gather算子，下图是rel_pos分支的gather算子。

<table width="100%" align="center">
    <tr>
        <th align="center">BEFORE</th>
        <th align="center">AFTER</th>
    </tr>
    <tr>
        <td>
            <img height="450px" alt="企业微信截图_16563337066001" src="https://user-images.githubusercontent.com/53067559/175944251-33ccdb0d-8eda-453e-9a5c-20d984f16866.png">
        </td>
        <td>
            <img height="450px" alt="企业微信截图_16563337066001" src="https://user-images.githubusercontent.com/53067559/175944261-0d9016ee-4848-47ab-a6ec-83e09791fd8f.png">
        </td>
    </tr>
</table>


同时选择正确的kernel,将transpose退回FP32精度,得出的结果与fp32一致。这验证了我们的想法，融合节点的tactic选择了最快的一个，但是精度有问题。因此我们在fp16时，选择了相应的tactic并达到较好的指标。

### 3.ONNX2TensorRT（INT8）
```
$ python3 onnx2TRT.py
```
当我们直接将整个模型进行int8量化后，精度误差到e-1，达不到工业部署的要求。我们再次将模型拆分成embedding和transformer两个部分，发现embedding校准之后效果是可以的，但是transformer模块是不行的。因此，可以采取int8和fp16混合精度优化，但由于时间关系，最后未能完成。

## 精度与加速效果  
为了靠近实际场景，我们设置最小batch为1，最优为4，最大为8的情况下进行评测。我们发现在最优的shape上加速最明显。
<table width="100%" align="center">
    <tbody align="center">
        <tr align="center">
            <th></th>
            <th>Batch Size</th>
            <th>Latency(ms)</th>
            <th>Through Put</th>
            <th>Latency Speedup</th>
        </tr>
        <!-- PyTorch -->
        <tr>
            <td rowspan="4">PyTorch</td>
            <td>1</td>
            <td>69</td>
            <td>14.4</td>
            <td>x1</td>
        </tr>
        <tr>
            <td>2</td>
            <td>124</td>
            <td>16.12</td>
            <td>x1</td>
        </tr>
        <tr>
            <td>4</td>
            <td>248</td>
            <td>16.12</td>
            <td>x1</td>
        </tr>
        <tr>
            <td>8</td>
            <td>481</td>
            <td>16.63</td>
            <td>x1</td>
        </tr>
        <!-- TensorRT -->
        <tr>
            <td rowspan="4">TensorRT(FP32)</td>
            <td>1</td>
            <td>57</td>
            <td>17.54</td>
            <td>x1.2</td>
        </tr>
        <tr>
            <td>2</td>
            <td>106</td>
            <td>18.86</td>
            <td>x1.17</td>
        </tr>
        <tr>
            <td>4</td>
            <td>196</td>
            <td>20.04</td>
            <td>x1.26</td>
        </tr>
        <tr>
            <td>8</td>
            <td>407</td>
            <td>19.65</td>
            <td>x1.19</td>
        </tr>
        <!-- TensorRT(FP16) -->
        <tr>
            <td rowspan="4">TensorRT(FP16)</td>
            <td>1</td>
            <td>54</td>
            <td>18.5</td>
            <td>x1.28</td>
        </tr>
        <tr>
            <td>2</td>
            <td>95</td>
            <td>21.05</td>
            <td>x1.3</td>
        </tr>
        <tr>
            <td>4</td>
            <td>137</td>
            <td>29.19</td>
            <td>x1.81</td>
        </tr>
        <tr>
            <td>8</td>
            <td>357</td>
            <td>22.4</td>
            <td>x1.34</td>
        </tr>
    </tbody>
</table>
<img width="785" alt="企业微信截图_f6c7d185-3de7-447b-a20d-b3e5c277eea2" src="https://user-images.githubusercontent.com/20862479/176047700-43a87016-ac47-4023-a821-bb3243aae764.png">


## Hackathon 2022 BUG
本次比赛我们总共发现了三个BUG。  

issue地址（PrecisionBUG）：https://github.com/NVIDIA/TensorRT/issues/2091  

issue地址（MessageBUG）：https://github.com/NVIDIA/TensorRT/issues/2073  

issue地址（SettingBUG）：https://github.com/NVIDIA/TensorRT/issues/2080

## 未来工作
由于没有掌握好时间，一开始我们花费了一部分时间精力准备将transformer块写成一个插件，可惜还在校验中，在之后的工作中，我们将使用transformer插件进行加速。另外我们还想将动态的batchsize指定到一个输入，其他输入的batchsize随着该输入而改变，这样应该可以减小显存和提高动态推理的性能。
