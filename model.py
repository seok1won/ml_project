import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2],
                                              (0, 0, 0, 0, (out_channels-in_channels)//2, (out_channels-in_channels)//2),
                                              "constant", 0)
        else:
            self.shortcut = lambda x: x

        # option B
        # if stride !=1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=stride,bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self,block,num_blocks,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self,block,out_channels,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers =[]
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out,out.size()[3])
        out = out.view(out.size(0),-1)
        out = self.linear(out)

        return out
    
def resnet_cifar(n):
    return ResNet(ResidualBlock, [n,n,n])

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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

def plainnet_cifar(n):
    return ResNet(PlainBlock, [n,n,n])

if __name__ == '__main__':
    print("--- ResNet ---")
    net = resnet_cifar(n=3)
    # print(net)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(f'ResNet-20 output size: {y.size()}')

    print("\n--- PlainNet ---")
    net = plainnet_cifar(n=3)
    # print(net)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(f'PlainNet-20 output size: {y.size()}')

class ReluBnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReluBnBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2],
                                              (0, 0, 0, 0, (out_channels-in_channels)//2, (out_channels-in_channels)//2),
                                              "constant", 0)
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        # Swapped order: Conv -> ReLU -> BN
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.bn2(F.relu(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NoBnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(NoBnBlock, self).__init__()
        # Use bias=True since we are not using BatchNorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2],
                                              (0, 0, 0, 0, (out_channels-in_channels)//2, (out_channels-in_channels)//2),
                                              "constant", 0)
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def relubn_resnet_cifar(n):
    return ResNet(ReluBnBlock, [n,n,n])

def nobn_resnet_cifar(n):
    return ResNet(NoBnBlock, [n,n,n])


class NoReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(NoReluBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2],
                                              (0, 0, 0, 0, (out_channels-in_channels)//2, (out_channels-in_channels)//2),
                                              "constant", 0)
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        # Forward pass without ReLU activations
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

def norelu_resnet_cifar(n):
    return ResNet(NoReluBlock, [n,n,n])

