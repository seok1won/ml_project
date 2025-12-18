import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_type='B'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if shortcut_type == 'A':
                # Option A: Parameter-free padding
                self.shortcut = LambdaLayer(lambda x: 
                                            F.pad(x if stride == 1 else x[:, :, ::2, ::2], 
                                                  (0, 0, 0, 0, (self.expansion*planes - in_planes)//2, (self.expansion*planes - in_planes)//2), 
                                                  "constant", 0))
            elif shortcut_type == 'B':
                # Option B: 1x1 convolution
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut_type='B'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if shortcut_type == 'A':
                 # Option A: Parameter-free padding
                self.shortcut = LambdaLayer(lambda x: 
                                            F.pad(x if stride == 1 else x[:, :, ::2, ::2], 
                                                  (0, 0, 0, 0, (self.expansion*planes - in_planes)//2, (self.expansion*planes - in_planes)//2), 
                                                  "constant", 0))
            elif shortcut_type == 'B':
                # Option B: 1x1 convolution
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, shortcut_type='B', num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.shortcut_type = shortcut_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.shortcut_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet_cifar(n, **kwargs):
    return ResNet(ResidualBlock, [n,n,n,n], **kwargs)

def resnet34(**kwargs):
    return ResNet(ResidualBlock, [3,4,6,3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)


class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, shortcut_type='B'): # shortcut_type is ignored
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

def plainnet_cifar(n, **kwargs):
    return ResNet(PlainBlock, [n,n,n,n], **kwargs)

class ResNetWithDropout(ResNet):
    def __init__(self, block, num_blocks, shortcut_type='B', dropout_p=0.5, num_classes=10):
        super(ResNetWithDropout, self).__init__(block, num_blocks, shortcut_type, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def resnet_with_dropout_cifar(n, dropout_p=0.5, **kwargs):
    return ResNetWithDropout(ResidualBlock, [n,n,n,n], dropout_p=dropout_p, **kwargs)

# (Other blocks like ReluBnBlock, NoBnBlock, NoReluBlock can be updated similarly if needed)
