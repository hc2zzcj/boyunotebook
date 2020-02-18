#### CNN

##### Conv

关于卷积的具体计算不再赘述，感受野的计算看了一下，可以参考这篇博客https://blog.csdn.net/program_developer/article/details/80958716

##### stride & padding：

$out_h = in_h + p_h - stride +1$

$out_w = in_w + p_w - stride +1$

拿pytorch里的Conv2D举例

我们用cifar10的数据，尺寸为32*32,3通道，batchsize=8

那么输入的尺寸就是[8 , 3, 32, 32]

```
nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,             bias=True, padding_mode='zeros')

x.shape = [8, 3, 32, 32]
conv = nn.Conv2d(3, 64, 3 padding=1)
out = conv(x)
out.shape=?

[8, 64, 32, 32]
```

处理图像时，Conv 与 Full Connect对比：

+ 图像是二维的，FC把图像展平成一个向量，在输入中相邻的元素可能不再相邻，难以捕捉局部信息。而Conv的设计天然具有提取局部信息的能力
+ Conv核的参数共享，减少参数量。可以处理尺寸更大的图像。

##### Pooling

池化层主要用于缓解卷积层对位置的过度敏感性。这句话没太理解。参考一下其他博客，可能有如下优点：

+ 对输入的特征图进行压缩，一方面使特征图变小，一方面进行特征压缩
+ Translation  Invariance，Input有微笑位移时，Pooling的输出是不变的，增强了模型鲁棒性.
+ 降低敏感性，避免过拟合

不过同时也会丢失一些特征，所以要选取合适的池化方法，一般有Max跟Mean。最大池化跟平均池化。

##### LeNet

Conv+Pooling+Conv+Pooling+Dense

![LeNet](D:\study\课程\LeNet.png)

![LeNet2](D:\study\课程\LeNet2.png)

+ 在大数据集上的表现不尽如人意
+ 网络计算复杂
+ 还没有大量深入研究参数初始化和非凸优化算法

##### AlexNet

+ 将sigmoid激活函数更改成了计算更简单的ReLU
+ Dropout来控制模型复杂度，避免过拟合
+ 引入数据增强，如翻转、剪裁、颜色变换

![AlexNet&LeNet](D:\study\课程\AlexNet&LeNet.png)

##### VGG

+ 通过重复使用简单的基础块来构建深度模型
+ Block：数个相同的填充为1、窗口形状3X3的卷积层，接上一个stride为2、窗口形状2X2的最大池化层

![vgg](D:\study\课程\vgg.png)

##### NiN

LeNet、AlexNet、VGG都是先由卷积层构成的模型充分抽取空间特征，再有全连接层构成的模块来输出分类结果

NiN：串联多个卷积层和全连接层来构建一个深层网络

用了输出通道数等于标签类别数的NiN块，然后使用全局平均池化对每个通道中的所有元素求平均直接用于分类

![NiN](D:\study\课程\NiN.png)

去除了容易造成过拟合的全连接层。替换成了

##### GoogLeNet

Inception系列

+ 由Inception基础块组成
+ Inception相当于有4条线路的子网络，通过不同窗口形状的卷积层和最大池化层来并行抽取信息。并使用1X1卷积减少通道数降低模型复杂度
+ 可以自定义的超参数是每个层的输出通道数

![Inception](D:\study\课程\Inception.png)

![googlenet](D:\study\课程\googlenet.png)

##### 错题记录

通道数为3，宽高均为224的输入，经过一层输出通道数为96，卷积核大小为11，步长为4，无padding的卷积层后，得到的feature map的宽高为：

+ 96
+ 54
+ 53
+ 224

答案为54，错误选择53。

计算公式为(n - k +2p)/s +1









