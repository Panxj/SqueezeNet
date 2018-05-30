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
  ### The SqueezeNet Architecture
  
Overview
-----------
Data Preparation
-----------
Training
-----------
Infer
-----------
Reference
-----------
