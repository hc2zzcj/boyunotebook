#### 线性回归

关于优化函数：[优化方法总结：SGD，Momentum，AdaGrad，RMSProp，Adam - Joe的博客 - CSDN博客](https://blog.csdn.net/u010089444/article/details/76725843)

课后练习：

课程中的损失函数定义为：

```
def squared_loss(y_hat, y):
	return (y_hat - y.view(y_hat.size())) ** 2 / 2
```

将返回结果替换为先的哪一个会导致模型无法训练：（阅读材料：https://pytorch.org/docs/stable/notes/broadcasting.html）

+ ``` (y_hat.view(-1) - y)```
+ ```(y_hat - y.view(-1))```
+ ```(y_hat - y.view(y_hat.shape))```
+ ```(y_hat - y.view(-1, 1))```

> `y_hat`的形状是`[n, 1]`，而`y`的形状是`[n]`，两者相减得到的结果的形状是`[n, n]`，相当于用`y_hat`的每一个元素分别减去`y`的所有元素，所以无法得到正确的损失值。对于第一个选项，`y_hat.view(-1)`的形状是`[n]`，与`y`一致，可以相减；对于第二个选项，`y.view(-1)`的形状仍是`[n]`，所以没有解决问题；对于第三个选项和第四个选项，`y.view(y_hat.shape)`和`y.view(-1, 1)`的形状都是`[n, 1]`，与`y_hat`一致，可以相减。

关于激活函数：[常用激活函数（激励函数）理解与总结_网络_StevenSun的博客空间-CSDN博客](https://blog.csdn.net/tyhj_sf/article/details/79932893)

