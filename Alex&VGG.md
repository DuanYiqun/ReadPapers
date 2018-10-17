
# Brief review of Alex&VGG
AlexNet 和 VGG 应该是最早的几篇网络结构方面的论文。 这两篇论文都属于没有任何层级结构的straigh forwardnetwork structures。
# AlexNet
先说 [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). 在introduction部分， UT的几位作者直接申明了这篇论文和之前最大的不同是在更大的dataset方面进行了训练并获得了优秀的表现。 本文主要用的dataset为ILSVRC-2012（ImageNet），大概包括1.2Million的训练数据。
在结构方面直接给出结构定义：
```python
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```
很明显的可以看到，在feature extraction 方面使用了卷积-Relu激活-maxpooling的三层为一组的结构。 一共五组这个结构。 值得注意的是因为网络层数很浅，所以以现在的视角来看，每层的kernelsize都比较大。stride 也比较大，这点在目前的论文中已经看不到了。 并且值得注意的是在输出方面这篇论文，使用了三个linear 全连接层。这点在目前看来，会占用比较大的模型空间，并且带来冗余。 Han et al.在2015，2016年两篇论文针对这种冗余提出了神经网络剪枝和deep compression 来减小alexnet的网络占用。

