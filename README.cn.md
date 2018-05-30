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
![](https://github.com/Panxj/SqueezeNet/raw/master/images/fire_module.jpg)
  * squeeze layer: 使用`1∗1`卷积核压缩通道数
  * expand layer: 分别使用 `1∗1` 与 `3∗3` 卷积核对扩展通道数
  * Fire module中使用3个可调的超参数：`s1x1`（squeeze convolution layer中1∗1 filter的个数）、`e1x1`（expand layer中1∗1 filter的个数）、`e3x3`（expand layer中3∗3 filter的个数）
  * 使用Fire module的过程中，令`s1x1 < e1x1 + e3x3`，这样squeeze layer可以限制输入通道数量

### The SqueezeNet Architecture
SqueezeNet以卷积层（conv1）开始，接着使用8个Fire modules (fire2-9)，最后以卷积层（conv10）结束。每个fire module中的filter数量逐渐增加，并且在conv1, fire4, fire8, 和 conv10这几层之后使用步长为2的max-pooling，即将池化层放在相对靠后的位置，如下图左侧子图，中间与右侧子图分别在初始结构上添加
simple bypass 与 complex bypass.

![](https://github.com/Panxj/SqueezeNet/raw/master/images/architecture.jpg)

Overview
-----------
Tabel 1. Directory structure

|file | description|
|:--- |:---|
train.py | Train script
infer.py | Prediction using the trained model
reader.py| Data reader
squeezenet.py| Model definition
data/val.txt|Validation data list
data/train.txt| Train data list

Data Preparation
-----------
首先从官网下载imagenet数据集，使用ILSVRC 2012(ImageNet Large Scale Visual Recognition Challenge)比赛用的子数据集，其中<br>
* 训练集: 1,281,167张图片 + 标签
* 验证集: 50,000张图片 + 标签
* 测试集: 100,000张图片

训练时， 所有图片按照短边resize到256，之后随机从左上，右上，中间，左下，右下 crop 出 `227 * 227` 大小图像输入网络。验证与测试时，同样按短边resize到 256，之后从中间 crop `227 * 227` 图像输入网络。所有图像均减去均值`[104,117,123]`，与imagenet 官网提供的均值文件稍有不同。
`reader.py`中相关函数如下，
```python
def random_crop(orig_im,new_shape):
        id = random.randint(0,5)
        # left-top
        if id == 0:
            image = orig_im[:new_shape[0], :new_shape[1]]
        # right-top
        elif id == 1:
            image = orig_im[:new_shape[0], orig_im.shape[1]-new_shape[1]:]
        # left-bottom
        elif id == 2:
            image = orig_im[orig_im.shape[0]-new_shape[0]:,:new_shape[1]]
        # right-bottom
        elif id == 3:
            image = orig_im[orig_im.shape[0]-new_shape[0]:, orig_im.shape[1]-new_shape[1]:]
        # center
        else:
            off_w = (orig_im.shape[1] - new_shape[1]) / 2
            off_h = (orig_im.shape[0] - new_shape[0]) / 2
            image = orig_im[off_h:off_h + new_shape[0], off_w:off_w + new_shape[1]]
        return image
def center_crop(orig_im,new_shape=(227,227)):
    off_w = (orig_im.shape[1] - new_shape[1]) / 2
    off_h = (orig_im.shape[0] - new_shape[0]) / 2
    image = orig_im[off_h:off_h + new_shape[0], off_w:off_w + new_shape[1]]
    return image

def load_img(img_path, resize = 256, crop=227, flip=False):
    im = cv2.imread(img_path)
    if flip:
        im  = im[:,::-1,:]    #horizental-flip
    h, w = im.shape[:2]
    h_new, w_new = resize, resize
    if h > w:
        h_new = resize * h / w
    else:
        w_new = resize * w / h
    im = cv2.resize(im,(h_new, w_new), interpolation=cv2.INTER_CUBIC)
    #b, g, r = cv2.split(im)
    #im = cv2.merge([b - mean_value[0], g - mean_value[1], r - mean_value[2]])
    im = random_crop(im,(crop,crop))
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # HWC => CHW
    im = im - mean_value
    im = im.flatten()
    #im = im / 255.0
    return im
```

train.txt 中数据如下：
```
n01440764/n01440764_10026.JPEG 0
n01440764/n01440764_10027.JPEG 0
n01440764/n01440764_10029.JPEG 0
n01440764/n01440764_10040.JPEG 0
n01440764/n01440764_10042.JPEG 0
n01440764/n01440764_10043.JPEG 0
```
val.txt 中数据如下：
```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
ILSVRC2012_val_00000006.JPEG 57
```
Training
-----------
为加快模型的训练，首先提取[caffe](http://caffe.berkeleyvision.org/)下的SqueezeNet模型参数，之后赋值到[PaddlePaddle](http://www.paddlepaddle.org/)模型中作为预训练参数，之后`finetune`. 论文作者[github](https://github.com/DeepScale/SqueezeNet)开源了两个版本的SqueezeNet 模型。 其中 SqueezeNet_v1.0 与论文中结构相同，SqueezeNet_v1.1 对原有结构进行了些许改动，使得在保证accuracy 不下降的情况下，计算量降低了 2.4x 倍。 SqueezeNet_v1.1 相比于论文中结构改动如下：

Tabel 2. changes in SqueezeNet_v1.1
 
 | | SqueezeNet_v1.0 | SqueezeNet_v1.1|
 |:---|:---|:---|
 |conv1| 96 filters of resolution 7x7|64 filters of resolution 3x3|
 |pooling layers| pool_{1,4,8} | pool_{1,3,5}|
 |computation| 1.72GFLOPS/image| 0.72 GFOLPS/image:2.4x less computation|
 |ImageNet accuracy| >=80.3% top-5| >=80.3% top-5|
 
此项目中，采用SqueezeNet_v1.1 结构。<br>
#### caffe2paddle 参数转化。
caffemodel中参数按照[PaddlePaddle](https://github.com/PaddlePaddle/models/tree/develop/image_classification/caffe2paddle)介绍的方法进行转化，并在训练开始前进行赋值，如下：
```python
#Load pre-trained params
if args.model is not None:
    for layer_name in parameters.keys():
        layer_param_path = os.path.join(args.model,layer_name)
        if os.path.exists(layer_param_path):
            h,w = parameters.get_shape(layer_name)
            parameters.set(layer_name,load_parameter(layer_param_path,h,w))
```
#### train
 `python train.py --model your/path/to/parametersFromCaffe --trainer num`
--model: 从caffemodel中提取的参数
Infer
-----------
Reference
-----------
