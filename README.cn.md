SqueezeNet
===========
Introduction
-----------
近来深层卷积网络的主要研究方向集中在提高正确率。对于相同的正确率水平，更小的CNN架构可以提供如下的优势： 
* 在分布式训练中，与服务器通信需求更小 
* 参数更少，从云端下载模型的数据量小 
* 更适合在FPGA等内存受限的设备上部署。 
基于这些优点, Squeezenet 提出fire module 模块，它在ImageNet上实现了和AlexNet相同的正确率，但是只使用了1/50的参数。
更进一步，使用模型压缩技术，可以将SqueezeNet压缩到0.5MB，这是AlexNet的1/510。

Architecture
-----------
### Architecture Design Strategies
  * 使用1∗1卷积代替3∗3 卷积：参数减少为原来的1/9 
  * 减少输入通道数量：这一部分使用squeeze layers来实现 
  * 将欠采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
### The Fire Module
<!--![](https://github.com/Panxj/SqueezeNet/raw/master/images/fire_module.jpg)-->
  * squeeze layer: 使用`1∗1`卷积核压缩通道数
  * expand layer: 分别使用 `1∗1` 与 `3∗3` 卷积核对扩展通道数
  * Fire module中使用3个可调的超参数：`s1x1`（squeeze convolution layer中1∗1 filter的个数）、`e1x1`（expand layer中1∗1 filter的个数）、`e3x3`（expand layer中3∗3 filter的个数）
  * 使用Fire module的过程中，令`s1x1 < e1x1 + e3x3`，这样squeeze layer可以限制输入通道数量

### The SqueezeNet Architecture
SqueezeNet以卷积层（conv1）开始，接着使用8个Fire modules (fire2-9)，最后以卷积层（conv10）结束。每个fire module中的filter数量逐渐增加，并且在conv1, fire4, fire8, 和 conv10这几层之后使用步长为2的max-pooling，即将池化层放在相对靠后的位置，如下图左侧子图，中间与右侧子图分别在初始结构上添加
simple bypass 与 complex bypass.

<!--![](https://github.com/Panxj/SqueezeNet/raw/master/images/architecture.jpg)-->

Overview
-----------
    Tabel 1. Directory structure
|file | description|
|:--- |:---|
train.py | Train script
infer.py | Infer script

Data Preparation
-----------
Training
-----------
Infer
-----------
Reference
-----------
