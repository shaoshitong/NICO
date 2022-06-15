import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
class IDNet(nn.Module):
    def __init__(self, depth, num_classes, im=3, maxpool_or_not=True, k=36):
        super(IDNet, self).__init__()
        '''
        Args:
            block: IDBlock
            nblock: a list, the elements is number of bottleneck in each denseblock
            growth_rate: channel size of bottleneck's output
        '''
        block=IDBlock
        self.n = (depth - 4) // 9
        self.im = im
        self.maxpool_or_not = maxpool_or_not
        if maxpool_or_not:  # 赋值语句可修改
            self.k1 = k // 4
        else:
            self.k1 = k // 3

        self.growth_rate = k  # growth_rate = k 在论文中

        num_planes = 16

        self.conv1 =  nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_planes, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)),
                ('norm0', nn.BatchNorm2d(num_planes)),
                ('relu0', nn.SiLU(inplace=True)),
                # ('pool0', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))),
            ]))

        # a DenseBlock and a transition layer
        self.dense1 = self._make_dense_layers(block, num_planes, self.n)
        num_planes += self.n * self.growth_rate
        self.trans1 = Transition(num_planes)  # 不进行压缩

        # a DenseBlock and a transition layer
        self.dense2 = self._make_dense_layers(block, num_planes, self.n)
        num_planes += self.n * self.growth_rate
        self.trans2 = Transition(num_planes)

        # a DenseBlock and a transition layer
        self.dense3 = self._make_dense_layers(block, num_planes, self.n)
        num_planes += self.n * self.growth_rate  # num_planes * 8 * 8

        # the last part is a linear layer as a classifier
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []

        # number of non-linear transformations in one DenseBlock depends on the parameter you set
        for i in range(nblock):
            layers.append(block(in_planes, self.k1, self.im, self.maxpool_or_not))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = checkpoint(lambda x:self.trans1(self.dense1(x)),out)
        out = checkpoint(lambda x:self.trans2(self.dense2(x)),out)
        out = checkpoint(self.dense3,out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.squeeze(out)
        out = self.linear(out)
        return out


class IDBlock(nn.Module):
    def __init__(self, input_channel, k1, im, maxpool_or_not=False):
        super(IDBlock, self).__init__()
        if maxpool_or_not:
            self.k = 4 * k1
        else:
            self.k = 3 * k1
        self.im = im
        self.maxpool_or_not = maxpool_or_not

        # 第一列
        self.columns1 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, k1, kernel_size=1, bias=False)
        )

        # 第二列
        self.columns2 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, self.k * self.im, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.k * self.im),
            nn.ReLU(),
            nn.Conv2d(self.k * self.im, k1, kernel_size=3, padding=1, bias=False)
        )

        # 第三列
        self.columns3 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, self.k * self.im, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.k * self.im),
            nn.ReLU(),
            nn.Conv2d(self.k * self.im, k1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k1),
            nn.ReLU(),
            nn.Conv2d(k1, k1, kernel_size=3, padding=1, bias=False)
        )

        if maxpool_or_not:
            self.columns4 = nn.Sequential(
                nn.MaxPool2d([3, 3], stride=1, padding=1),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(),
                nn.Conv2d(input_channel, k1, kernel_size=1, bias=False)
            )

    def forward(self, x):
        y1 = self.columns1(x)
        y2 = self.columns2(x)
        y3 = self.columns3(x)

        if self.maxpool_or_not:
            y4 = self.columns4(x)
            y = torch.cat([x, y1, y2, y3, y4], 1)
        else:
            y = torch.cat([x, y1, y2, y3], 1)
        return y


class Transition(nn.Module):  # factor压缩因子
    def __init__(self, input_shape):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(input_shape)
        self.conv = nn.Conv2d(input_shape, input_shape, kernel_size=1, bias=False)
        '''
            transition layer is used for down sampling the feature
            when compress rate is 0.5, out_planes is a half of in_planes
        '''

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        # use average pooling change the size of feature map here
        out = F.avg_pool2d(out, 2, 2)
        return out

m=Transition(10)
print(m.to(m.bn.bias.device))
print(m.device)