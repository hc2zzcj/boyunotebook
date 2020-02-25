#### Batch Norm

批量归一化，引入两个超参数：拉伸参数和位移参数

+ 对全连接层做BN，位于全连接层中的仿射变换和激活函数之间
+ 对卷积层做BN，位于卷积计算之后，激活函数之前。对卷积计算输出的每一个通道有一个独立的拉伸和位移参数。
+ 预测时的BN
  + 训练：以batch为单位，对每个batch计算均值和方差。
  + 预测：用移动平均估算整个训练数据集的样本均值和方差。

使用BN的LeNet实现。

```
![ResNetBlock](D:\study\课程\ResNetBlock.png)net = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            d2l.FlattenLayer(),
            nn.Linear(16*4*4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
```

#### ResNet

深度CNN网络达到一定深度后再一味的增加层数并不能带来进一步地分类性能提高，反而会招致网络收敛变得更慢，准确率也变得更差。

##### Residual Block

![ResNetBlock](D:\study\课程\ResNetBlock.png)



可以看到残差块的最后一步是与输入X相加，所以每一个残差块的输入输出尺寸一样

##### DenseNet

dense block：

![densenet](D:\study\课程\densenet.png)

还有过渡层（transition layer）：1X1卷积层，控制通道数，避免过大



