
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
很明显的可以看到，在feature extraction 方面使用了卷积-Relu激活-maxpooling的三层为一组的结构。 一共五组这个结构。 值得注意的是因为网络层数很浅，所以以现在的视角来看，每层的kernelsize都比较大。stride 也比较大，这点在目前的论文中已经看不到了。 并且值得注意的是在输出方面这篇论文，使用了三个linear 全连接层。这点在目前看来，会占用比较大的模型空间，并且带来冗余。 Han et al.在2015，2016年两篇论文针对这种冗余提出了神经网络剪枝和deep compression 来减小alexnet的参数规模。
值得注意的一点是，限于当时的条件 （GTX580 3GB GPU）为了在ImageNet上面完成这个操作，作者把Alexnet变成了并行的两部分，结构如下：  
![AlexNet](https://cdn-images-1.medium.com/max/800/0*xPOQ3btZ9rQO23LK.png "GitHub,Social Coding")   
并行的两部分直到classifier layer 才有参数交流。
这篇文章在当时极大的提升了CNN在主流识别数据集上的准确率，虽然目前看来（ImageNet Top5 36.7% Top1 15.4%， 自己做了一个小实验100 epoch cifar10 在89% 左右）这样的准确率不是很高，但是在当时是有重大意义的。

## VGG
主要贡献在于研究了加深网络层数对准确率的影响。结构方面也是平直的网络结构堆叠。因为网络层数的加深，卷积核方面使用了和目前比较贴近的3*3 卷积核的堆叠。
直接给出网络结构定义：

```python 
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```



这里是给出了一个网络类，以及配置方法，在配置的时候只需要:
``` model = VGG(make_layers(cfg['A']), **kwargs)```  
cfg里面的参数就是每层的通道数以及是否进行downsampling。 在表现方面比AlexNet有大幅提升。不过VGG没有解决当网络结构变得更深的时候如何有效抑制grediant vanishing 和如何有效传递参数的问题。

