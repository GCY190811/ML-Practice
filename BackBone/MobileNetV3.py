import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x+3, inplace=True) / 6

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x+3, input=True) / 6

class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k, stride, use_bn, act_type):
        super(ConvBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=stride, padding=k // 2)

        self.bn1 = None
        if use_bn:
            self.bn1 = nn.BatchNorm2d(out_c)

        self.act1 = None
        self.act_type = act_type
        if act_type == "HS":
            self.act1 = nn.ReLU6(inplace=True)
        elif act_type == "RE":
            self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        if self.act_type == "HS":
            x = x * self.act1(x+3) / 6
        else:
            x = self.act1(x)

        return x


class SE(nn.Module):
    def __init__(self, in_c, reduce_ratio):
        super(SE, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_channels=in_c, out_channels=in_c // reduce_ratio, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_c // reduce_ratio)
        self.act1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(in_channels=in_c // reduce_ratio, out_channels=in_c, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.act2 = HSigmoid()

    def forward(self, x):
        weights = self.act2(self.bn2(self.fc2(self.act1(self.bn1(self.fc1(self.avg_pool(x)))))))
        return x * weights


class BNeck(nn.Module):
    def __init__(self, in_c, exp_size, out_c, k, stride, use_se, act_type):
        super(BNeck, self).__init__()

        # inverted block
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=exp_size, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.act1 = nn.ReLU(inplace=True)
        if act_type == "HS":
            self.act1 = HSigmoid()
        elif act_type == "RE":
            self.act1 = nn.ReLU(inplace=True)

        # depthwise block
        self.conv2 = nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=k, stride=stride,
                               padding=k // 2, groups=exp_size)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.act2 = nn.ReLU(inplace=True)
        if act_type == "HS":
            self.act2 = HSigmoid()
        elif act_type == "RE":
            self.act2 = nn.ReLU(inplace=True)

        # se
        self.se = None
        if use_se:
            self.se = SE(in_c=exp_size, reduce_ratio=4)

        # pointwise block
        self.conv3 = nn.Conv2d(in_channels=exp_size, out_channels=out_c, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_c)

        # link
        self.use_res_connect = False
        if stride == 1 and in_c == out_c:
            self.use_res_connect = True

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.act2(self.bn2(self.conv2(y)))
        if self.se:
            y = self.se(y)
        y = self.bn3(self.conv3(y))

        if self.use_res_connect:
            y = y + x

        return y

class MobileNetV3(nn.Module):
    def __init__(self, num_class):
        super(MobileNetV3, self).__init__()
        self.net = nn.Sequential(
            ConvBN(1, 16, k=3, stride=2, use_bn=True, act_type="HS"),
            BNeck(in_c=16, exp_size=16, out_c=16, k=3, stride=1, use_se=False, act_type="RE"),
            BNeck(in_c=16, exp_size=64, out_c=24, k=3, stride=2, use_se=False, act_type="RE"),
            BNeck(in_c=24, exp_size=72, out_c=24, k=3, stride=1, use_se=False, act_type="RE"),
            BNeck(in_c=24, exp_size=72, out_c=40, k=5, stride=2, use_se=True, act_type="RE"),
            BNeck(in_c=40, exp_size=120, out_c=40, k=5, stride=1, use_se=True, act_type="RE"),
            BNeck(in_c=40, exp_size=120, out_c=40, k=5, stride=1, use_se=True, act_type="RE"),
            BNeck(in_c=40, exp_size=240, out_c=80, k=3, stride=2, use_se=False, act_type="HS"),
            BNeck(in_c=80, exp_size=200, out_c=80, k=3, stride=1, use_se=False, act_type="HS"),
            BNeck(in_c=80, exp_size=184, out_c=80, k=3, stride=1, use_se=False, act_type="HS"),
            BNeck(in_c=80, exp_size=184, out_c=80, k=3, stride=1, use_se=False, act_type="HS"),
            BNeck(in_c=80, exp_size=480, out_c=112, k=3, stride=1, use_se=True, act_type="HS"),
            BNeck(in_c=112, exp_size=672, out_c=112, k=3, stride=1, use_se=True, act_type="HS"),
            BNeck(in_c=112, exp_size=672, out_c=160, k=5, stride=2, use_se=True, act_type="HS"),
            BNeck(in_c=160, exp_size=672, out_c=160, k=5, stride=1, use_se=True, act_type="HS"),
            BNeck(in_c=160, exp_size=960, out_c=160, k=5, stride=1, use_se=True, act_type="HS"),
            ConvBN(in_c=160, out_c=960, k=1, stride=1, use_bn=True, act_type="HS"),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1, padding=0),
            HSigmoid(),
            nn.Conv2d(in_channels=1280, out_channels=num_class, kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    num_class = 10
    mobile_net_v3 = MobileNetV3(num_class)
    x = torch.rand((5, 1, 224, 224))

    print("fake input:", x.size())
    for layer in mobile_net_v3.net:
        x = layer(x)
        print(layer.__class__.__name__, 'output size():\t', x.size())

